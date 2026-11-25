"""Stage 1: leakage-free ingestion of JPL SBDB + Sentry data and target formulation.

Outputs a :class:`DatasetBundle` whose feature matrix ``X`` is guaranteed (by
:func:`assert_no_leakage`) to contain no label-defining quantity (MOID, H, PHA flag),
no Sentry output, and no column numerically equivalent to one.

Mathematical notes
------------------
* PHA  <=>  MOID <= 0.05 au  AND  H <= 22.0.  We use JPL's ``pha`` flag (computed at full
  precision) as the label and cross-check it against the rounded audit columns.
* H = 5 log10(1329 km / (D sqrt(p_v)))  ->  diameter+albedo reconstruct H.  Controlled by
  :class:`PhysicalPolicy`.
* MOID is a deterministic function of (a, e, i, Omega, omega) given Earth's orbit, so removing
  the column does NOT remove the information.  :func:`leakage_probe` quantifies this.
* Angles are encoded as (sin, cos) to remove the 0/360 deg discontinuity.
* Full covariances from ``sbdb.api`` are in *cometary* elements (e, q, tp, node, peri, i),
  angles in deg, tp in days, and are referenced to their own epoch, which may differ from the
  SBDB-query orbit epoch.  They are stored with their own nominal state for Stage 3 sampling.
* Missing physical values are left as NaN with explicit ``has_*`` masks (no imputation).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import requests

from .config import (
    COV_LABELS, FORBIDDEN_NAME_REGEX, GATE_NAME_ALLOWLIST, LABEL_DEFINING, MAX_ABS_CORR_WITH_AUDIT, ORBIT_CORE,
    PHA_H_MAX, PHA_MOID_AU, SBDB_FIELDS, SBDB_OBJECT_URL, SBDB_QUERY_URL, SENTRY_OUTPUTS,
    SENTRY_URL, PhysicalPolicy, PipelineConfig, TargetMode,
)

log = logging.getLogger(__name__)

JsonGetter = Callable[[str, dict[str, Any]], dict[str, Any]]


class LeakageError(RuntimeError):
    """Raised when a forbidden quantity reaches the feature matrix."""


# --------------------------------------------------------------------------------------
# HTTP layer (injectable for deterministic tests)
# --------------------------------------------------------------------------------------
class JPLClient:
    def __init__(self, timeout_s: float = 120.0, max_retries: int = 4,
                 session: requests.Session | None = None) -> None:
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "neo-risk-research-pipeline/0.1"})

    def get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            wait = 2.0 ** attempt * (1.0 + 0.25 * np.random.random())     # jittered backoff
            try:
                r = self.session.get(url, params=params, timeout=self.timeout_s)
                if r.status_code in (429, 500, 502, 503, 504):
                    ra = r.headers.get("Retry-After")
                    if ra and ra.isdigit():
                        wait = max(wait, float(ra))
                    raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
                r.raise_for_status()
                return r.json()
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(wait)
        raise RuntimeError(f"GET {url} failed after {self.max_retries + 1} attempts: "
                           f"{last_exc!r}") from last_exc


def _sha256(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------------------
# Raw fetchers
# --------------------------------------------------------------------------------------
def fetch_sbdb_neos(get_json: JsonGetter, cfg: PipelineConfig) -> tuple[pd.DataFrame, dict]:
    """Fetch all NEO asteroids (sb-kind=a, sb-group=neo) with paging."""
    frames: list[pd.DataFrame] = []
    offset, total, version = 0, None, None
    hashes: list[str] = []
    while True:
        limit = cfg.sbdb_page_size
        if cfg.max_objects is not None:
            limit = min(limit, cfg.max_objects - offset)
            if limit <= 0:
                break
        params = {"fields": ",".join(SBDB_FIELDS), "sb-kind": "a", "sb-group": "neo",
                  "full-prec": "1", "limit": str(limit), "limit-from": str(offset),
                  **cfg.extra_query_params}
        payload = get_json(SBDB_QUERY_URL, params)
        if "data" not in payload or "fields" not in payload:
            raise ValueError(f"Unexpected SBDB payload keys: {list(payload)}")
        version = payload.get("signature", {}).get("version")
        total = int(payload.get("count", 0))
        hashes.append(_sha256(payload))
        page = pd.DataFrame(payload["data"], columns=payload["fields"])
        frames.append(page)
        offset += len(page)
        if len(page) == 0 or offset >= total:
            break
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SBDB_FIELDS)
    meta = {"url": SBDB_QUERY_URL, "api_version": version, "count_reported": total,
            "rows": len(df), "sha256_pages": hashes}
    return df, meta


def fetch_sentry(get_json: JsonGetter) -> tuple[pd.DataFrame, dict]:
    payload = get_json(SENTRY_URL, {})
    df = pd.DataFrame(payload.get("data", []))
    meta = {"url": SENTRY_URL, "api_version": payload.get("signature", {}).get("version"),
            "count_reported": int(payload.get("count", 0)), "rows": len(df),
            "sha256": _sha256(payload)}
    return df, meta


# --------------------------------------------------------------------------------------
# Covariance (per object, sbdb.api cov=mat)
# --------------------------------------------------------------------------------------
def repair_covariance(cov: np.ndarray, rel_eps: float = 1e-12) -> tuple[np.ndarray, dict]:
    """Symmetrize and clip eigenvalues to make a valid PSD covariance.

    Returns the repaired matrix and diagnostics (asymmetry, min eigenvalue, condition number,
    whether clipping was required).  Non-finite input yields an all-NaN matrix.
    """
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (6, 6) or not np.all(np.isfinite(cov)):
        return np.full((6, 6), np.nan), {"cov_valid": False, "cov_asym": np.nan,
                                         "cov_min_eig": np.nan, "cov_cond": np.nan,
                                         "cov_repaired": False}
    asym = float(np.max(np.abs(cov - cov.T)) / max(np.max(np.abs(cov)), 1e-300))
    sym = 0.5 * (cov + cov.T)
    # Scale to unit diagonal before eigen-decomposition: elements span ~10 orders of magnitude.
    d = np.sqrt(np.clip(np.diag(sym), 1e-300, None))
    corr = sym / np.outer(d, d)
    w, v = np.linalg.eigh(corr)
    floor = rel_eps * max(w.max(), 1e-300)
    repaired = bool(w.min() < floor)
    w_c = np.clip(w, floor, None)
    corr_c = (v * w_c) @ v.T
    out = corr_c * np.outer(d, d)
    out = 0.5 * (out + out.T)
    return out, {"cov_valid": True, "cov_asym": asym, "cov_min_eig": float(w.min()),
                 "cov_cond": float(w_c.max() / w_c.min()), "cov_repaired": repaired}


def parse_covariance(payload: dict[str, Any]) -> dict[str, Any]:
    """Parse a sbdb.api ``cov=mat`` payload into a flat record."""
    rec: dict[str, Any] = {"has_cov": False}
    orbit = payload.get("orbit") or {}
    cov = orbit.get("covariance")
    if not cov or "data" not in cov:
        return rec
    # Two observed JPL formats: long-arc orbits give `elements` (label + nominal value);
    # short-arc orbits give bare `labels` and no nominal values (fall back to orbit elements,
    # flagged via cov_nominal_from_orbit because the epochs may then differ).
    if "elements" in cov:
        labels = [el.get("label", el.get("name")) for el in cov["elements"]]
        nominal = {lab: float(el["value"]) for lab, el in zip(labels, cov["elements"]) if lab in COV_LABELS}
        rec["cov_nominal_from_orbit"] = False
    else:
        labels = list(cov.get("labels", []))
        name_map = {"node": "om", "peri": "w"}
        orb = {el["name"]: el["value"] for el in orbit.get("elements", [])}
        nominal = {lab: float(orb.get(name_map.get(lab, lab), np.nan)) for lab in labels}
        rec["cov_nominal_from_orbit"] = True
    if tuple(labels[:6]) != COV_LABELS:
        raise ValueError(f"Unexpected covariance element order {labels}; expected {COV_LABELS}")
    full = np.array([[float(x) for x in row] for row in cov["data"]], dtype=float)
    # Orbits fitted with non-gravitational / SRP parameters (e.g. A1, A2, RHO, AMRAT) give an
    # (6+k)x(6+k) matrix.  The Gaussian marginal over the orbital elements is the leading block.
    mat = full[:6, :6]
    rec["cov_extra_params"] = ",".join(labels[6:])
    mat_r, diag = repair_covariance(mat)
    rec.update(diag)
    rec["has_cov"] = bool(diag["cov_valid"])
    rec["cov_epoch"] = float(cov.get("epoch", np.nan))
    rec["orbit_epoch"] = float(orbit.get("epoch", np.nan))
    for lab in COV_LABELS:
        rec[f"cov_nom_{lab}"] = nominal[lab]
    iu = np.triu_indices(6)
    for r, c in zip(*iu):
        rec[f"cov_{COV_LABELS[r]}_{COV_LABELS[c]}"] = mat_r[r, c]
    return rec


def fetch_covariances(designations: Iterable[str], get_json: JsonGetter,
                      cfg: PipelineConfig, use_cache: bool = True) -> pd.DataFrame:
    """Fetch full 6x6 covariances for a subset of objects (one request each, cached)."""
    cache = cfg.cache_dir / "cov"
    cache.mkdir(parents=True, exist_ok=True)

    def one(des: str) -> dict[str, Any]:
        fp = cache / (re.sub(r"[^A-Za-z0-9_-]", "_", des) + ".json")
        if use_cache and fp.exists():
            payload = json.loads(fp.read_text())
        else:
            try:
                payload = get_json(SBDB_OBJECT_URL, {"sstr": des, "cov": "mat", "full-prec": "1"})
            except Exception as exc:  # noqa: BLE001 - record failure, keep going
                log.warning("covariance fetch failed for %s: %s", des, exc)
                return {"pdes": des, "has_cov": False}
            fp.write_text(json.dumps(payload))
        try:
            rec = parse_covariance(payload)
        except Exception as exc:  # noqa: BLE001 - one malformed payload must not kill the run
            log.warning("covariance parse failed for %s: %s", des, exc)
            rec = {"has_cov": False, "cov_parse_error": str(exc)[:200]}
        rec["pdes"] = des
        return rec

    des_list = list(dict.fromkeys(designations))
    with ThreadPoolExecutor(max_workers=max(1, cfg.cov_n_jobs)) as ex:
        recs = list(ex.map(one, des_list))
    return pd.DataFrame(recs).set_index("pdes") if recs else pd.DataFrame()


def _cov_cache_path(cache_dir: Path, des: str) -> Path:
    return cache_dir / "cov" / (re.sub(r"[^A-Za-z0-9_-]", "_", des) + ".json")


def covariance_fetch_order(designations: Iterable[str], seed: int = 2026) -> list[str]:
    """Seeded random permutation: every prefix is a uniform (label-independent) random sample."""
    des = sorted(set(designations))
    rng = np.random.default_rng(seed)
    return [des[i] for i in rng.permutation(len(des))]


def cached_prefix(order: list[str], cache_dir: Path) -> list[str]:
    """Longest prefix of ``order`` whose covariance payloads are all cached."""
    out = []
    for d in order:
        if not _cov_cache_path(cache_dir, d).exists():
            break
        out.append(d)
    return out


def load_cached_covariances(designations: Iterable[str], cache_dir: Path) -> pd.DataFrame:
    """Parse cached sbdb.api payloads (no network)."""
    recs = []
    for d in designations:
        fp = _cov_cache_path(cache_dir, d)
        if not fp.exists():
            recs.append({"pdes": d, "has_cov": False})
            continue
        try:
            rec = parse_covariance(json.loads(fp.read_text()))
        except Exception as exc:  # noqa: BLE001
            rec = {"has_cov": False, "cov_parse_error": str(exc)[:200]}
        rec["pdes"] = d
        recs.append(rec)
    return pd.DataFrame(recs).set_index("pdes")


# --------------------------------------------------------------------------------------
# Cleaning & feature construction
# --------------------------------------------------------------------------------------
NUMERIC_SBDB = [f for f in SBDB_FIELDS if f not in
                {"spkid", "pdes", "full_name", "class", "first_obs", "last_obs",
                 "spec_B", "spec_T", "pha"}]


def clean_sbdb(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    for c in NUMERIC_SBDB:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["pdes"] = df["pdes"].astype(str).str.strip()
    df["full_name"] = df["full_name"].astype(str).str.strip()
    n0 = len(df)
    dup = df.duplicated("pdes")
    df = df[~dup]
    core_ok = df[list(ORBIT_CORE)].notna().all(axis=1)
    phys_ok = (df["a"] > 0) & (df["e"] >= 0) & (df["e"] < 1) & (df["i"] >= 0) & (df["i"] <= 180)
    df = df[core_ok & phys_ok].set_index("pdes")
    report = {"rows_in": n0, "dropped_duplicate": int(dup.sum()),
              "dropped_missing_core_orbit": int((~core_ok).sum()),
              "dropped_unphysical": int((core_ok & ~phys_ok).sum()), "rows_out": len(df)}
    return df, report


def clean_sentry(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["ip", "ps_cum"]).rename_axis("pdes")
    s = raw.copy()
    s["pdes"] = s["des"].astype(str).str.strip()
    s = s.rename(columns={"diameter": "sentry_diameter"})
    for c in ["ip", "ps_cum", "ps_max", "ts_max", "n_imp", "v_inf", "h", "sentry_diameter"]:
        if c in s:
            s[c] = pd.to_numeric(s[c], errors="coerce")
    keep = [c for c in s.columns if c in SENTRY_OUTPUTS]
    return s.drop_duplicates("pdes").set_index("pdes")[keep]


def _tax_complex(spec_b: Any, spec_t: Any) -> str | None:
    """Map SMASS (Bus) / Tholen classes to coarse complexes C/S/X/other."""
    for s in (spec_b, spec_t):
        if isinstance(s, str) and s.strip():
            t = s.strip().upper()
            if t[0] in "CBFG":
                return "C"
            if t[0] in "SQAVKLR" or t.startswith("O"):
                return "S"
            if t[0] in "XMEP":
                return "X"
            return "other"
    return None


def _date_to_jd(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", format="%Y-%m-%d")
    return dt.map(lambda d: d.to_julian_date() if pd.notna(d) else np.nan).astype(float)


def build_features(df: pd.DataFrame, policy: PhysicalPolicy,
                   snapshot_jd: float | None = None) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Construct the leakage-free feature matrix from cleaned SBDB rows."""
    X = pd.DataFrame(index=df.index)
    groups: dict[str, list[str]] = {"static_orbit": [], "obs_meta": [], "physical": [], "masks": []}

    def add(group: str, name: str, values: pd.Series | np.ndarray) -> None:
        X[name] = np.asarray(values, dtype=float)
        groups[group].append(name)

    # --- static orbit (NaN-free by construction of clean_sbdb, except derived t_jup/q/ad)
    add("static_orbit", "a", df["a"])
    add("static_orbit", "e", df["e"])
    add("static_orbit", "i_deg", df["i"])
    add("static_orbit", "q", df["a"] * (1 - df["e"]))         # recompute: guarantees no NaN
    add("static_orbit", "Q", df["a"] * (1 + df["e"]))
    for ang in ("om", "w", "ma"):
        rad = np.deg2rad(df[ang])
        add("static_orbit", f"sin_{ang}", np.sin(rad))
        add("static_orbit", f"cos_{ang}", np.cos(rad))
    # Tisserand w.r.t. Jupiter, recomputed (a_J = 5.2026 au; inclination wrt ecliptic approx.)
    aj = 5.2026
    add("static_orbit", "t_jup",
        aj / df["a"] + 2 * np.cos(np.deg2rad(df["i"])) * np.sqrt(df["a"] / aj * (1 - df["e"] ** 2)))

    # --- observation metadata
    add("obs_meta", "log10_data_arc", np.log10(df["data_arc"].clip(lower=0) + 1))
    add("obs_meta", "log10_n_obs", np.log10(df["n_obs_used"].clip(lower=0) + 1))
    add("obs_meta", "n_radar_obs", df["n_del_obs_used"].fillna(0) + df["n_dop_obs_used"].fillna(0))
    add("obs_meta", "condition_code", df["condition_code"])
    add("obs_meta", "rms", df["rms"])
    for s in ("a", "e", "i", "om", "w", "ma"):
        add("obs_meta", f"log10_sigma_{s}", np.log10(df[f"sigma_{s}"].where(df[f"sigma_{s}"] > 0)))
    ref = snapshot_jd if snapshot_jd is not None else float(df["epoch"].max())
    add("obs_meta", "obs_staleness_yr", (ref - _date_to_jd(df["last_obs"])) / 365.25)

    # --- sparse physical (NaN preserved; masks explicit)
    if policy is not PhysicalPolicy.STRICT:
        add("physical", "log10_diameter_km", np.log10(df["diameter"].where(df["diameter"] > 0)))
        add("masks", "has_diameter", df["diameter"].notna())
        add("physical", "log10_rot_per_h", np.log10(df["rot_per"].where(df["rot_per"] > 0)))
        add("masks", "has_rot_per", df["rot_per"].notna())
        tax = [_tax_complex(b, t) for b, t in zip(df["spec_B"], df["spec_T"])]
        tax_s = pd.Series(tax, index=df.index)
        for k in ("C", "S", "X", "other"):
            add("physical", f"tax_{k}", (tax_s == k))
        add("masks", "has_taxonomy", tax_s.notna())
    if policy is PhysicalPolicy.FULL:
        add("physical", "albedo_UNSAFE", df["albedo"])  # name intentionally trips the gate

    # --- automatic missingness masks for every remaining NaN-bearing non-physical column
    for c in list(groups["obs_meta"]):
        if X[c].isna().any():
            add("masks", f"{c}_missing", X[c].isna())
    return X, groups


def make_target(df: pd.DataFrame, sentry: pd.DataFrame, mode: TargetMode,
                cfg: PipelineConfig) -> pd.Series:
    if mode is TargetMode.PHA:
        return (df["pha"].astype(str).str.upper() == "Y").astype(int).rename("y")
    in_sentry = df.index.isin(sentry.index)
    if mode is TargetMode.SENTRY_BINARY:
        return pd.Series(in_sentry.astype(int), index=df.index, name="y")
    if mode is TargetMode.LOG10_IP:
        ip = sentry["ip"].reindex(df.index)
        v = np.log10(ip.where(ip > 0)).fillna(cfg.log10_ip_floor)
        return v.clip(lower=cfg.log10_ip_floor).rename("y")
    if mode is TargetMode.PS_CUM:
        ps = sentry["ps_cum"].reindex(df.index).fillna(cfg.ps_cum_floor)
        return ps.clip(lower=cfg.ps_cum_floor).rename("y")
    raise ValueError(mode)


# --------------------------------------------------------------------------------------
# Anti-leakage gate
# --------------------------------------------------------------------------------------
def assert_no_leakage(X: pd.DataFrame, audit_df: pd.DataFrame,
                      max_abs_corr: float = MAX_ABS_CORR_WITH_AUDIT) -> dict[str, float]:
    """Raise LeakageError if X contains a forbidden column name or a numeric near-copy.

    Returns the max |Spearman rho| of each X column against the audit columns (for reporting).
    """
    forbidden = (LABEL_DEFINING | SENTRY_OUTPUTS | {"albedo"})
    bad = [c for c in X.columns if c in forbidden
           or (re.search(FORBIDDEN_NAME_REGEX, c, re.I) and c not in GATE_NAME_ALLOWLIST)]
    if bad:
        raise LeakageError(f"Forbidden columns in X: {bad}")
    num_audit = audit_df.select_dtypes(include=[np.number])
    num_audit = num_audit.loc[:, (num_audit.notna().sum() >= 30) & (num_audit.nunique() > 1)]
    rx = X.rank()
    ra = num_audit.reindex(X.index).rank()
    max_corr: dict[str, float] = {}
    offenders: list[tuple[str, str, float]] = []
    for c in X.columns:
        if X[c].nunique(dropna=True) < 3:     # binary masks / one-hots: skip rank corr
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            corrs = ra.corrwith(rx[c]).abs()
        corrs = corrs.dropna()
        if corrs.empty:
            continue
        max_corr[c] = float(corrs.max())
        if corrs.max() > max_abs_corr:
            offenders.append((c, str(corrs.idxmax()), float(corrs.max())))
    if offenders:
        raise LeakageError(f"Numeric near-copies of audit columns in X: {offenders}")
    return max_corr


def leakage_probe(X: pd.DataFrame, audit_df: pd.DataFrame, targets: Iterable[str] = ("H", "moid"),
                  n_splits: int = 3, seed: int = 0, max_rows: int = 20_000) -> dict[str, float]:
    """Cross-validated R^2 of a GBDT predicting each forbidden quantity from X.

    High R^2 means the information is still recoverable from X even though the column is gone.
    This is a *reported* diagnostic, not a gate: MOID is a function of the orbital elements.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold, cross_val_score

    out: dict[str, float] = {}
    for t in targets:
        if t not in audit_df:
            continue
        y = audit_df[t].reindex(X.index)
        m = y.notna()
        Xm, ym = X[m], y[m]
        if t == "moid":
            ym = np.log10(ym.clip(lower=1e-6))
        if len(ym) > max_rows:
            idx = ym.sample(max_rows, random_state=seed).index
            Xm, ym = Xm.loc[idx], ym.loc[idx]
        if len(ym) < 50:
            continue
        model = HistGradientBoostingRegressor(max_iter=200, random_state=seed)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        out[f"r2_{t}"] = float(np.mean(cross_val_score(model, Xm, ym, cv=cv, scoring="r2")))
    return out


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
@dataclass
class DatasetBundle:
    X: pd.DataFrame
    y: pd.Series
    audit_df: pd.DataFrame
    orbit_df: pd.DataFrame
    meta_df: pd.DataFrame
    feature_groups: dict[str, list[str]]
    provenance: dict[str, Any] = field(default_factory=dict)
    cov_df: pd.DataFrame | None = None

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ("X", "audit_df", "orbit_df", "meta_df"):
            getattr(self, name).to_parquet(out_dir / f"{name}.parquet")
        self.y.to_frame().to_parquet(out_dir / "y.parquet")
        if self.cov_df is not None and not self.cov_df.empty:
            self.cov_df.to_parquet(out_dir / "cov_df.parquet")
        (out_dir / "provenance.json").write_text(json.dumps(
            {**self.provenance, "feature_groups": self.feature_groups}, indent=2, default=str))


def build_dataset(cfg: PipelineConfig, get_json: JsonGetter | None = None,
                  cov_designations: Iterable[str] | None = None) -> DatasetBundle:
    if get_json is None:
        get_json = JPLClient(cfg.request_timeout_s, cfg.max_retries).get_json
    snapshot_utc = datetime.now(timezone.utc).isoformat()

    raw, sbdb_meta = fetch_sbdb_neos(get_json, cfg)
    sraw, sentry_meta = fetch_sentry(get_json)
    df, clean_report = clean_sbdb(raw)
    sentry = clean_sentry(sraw)

    # --- audit frame: everything label-defining or label-derived
    audit = df[["moid", "moid_ld", "H", "albedo"]].copy()
    audit["pha"] = (df["pha"].astype(str).str.upper() == "Y").astype(int)
    audit["pha_recomputed"] = ((df["moid"] <= PHA_MOID_AU) & (df["H"] <= PHA_H_MAX)).astype(int)
    audit = audit.join(sentry, how="left")
    audit["in_sentry"] = audit.index.isin(sentry.index).astype(int)

    X, groups = build_features(df, cfg.physical_policy, cfg.snapshot_jd)
    y = make_target(df, sentry, cfg.target_mode, cfg)
    orbit_df = df[list(ORBIT_CORE) + [f"sigma_{s}" for s in ("a", "e", "i", "om", "w", "ma")]].copy()
    meta_df = df[["spkid", "full_name", "class", "first_obs", "last_obs", "data_arc",
                  "spec_B", "spec_T"]].copy()
    meta_df["short_arc"] = (df["data_arc"].fillna(0) < 30).astype(int)

    if cfg.physical_policy is not PhysicalPolicy.FULL:
        corr_report = assert_no_leakage(X, audit)
    else:
        corr_report = {}
        log.warning("PhysicalPolicy.FULL: leakage gate bypassed (ablation only)")

    cov_df = None
    if cov_designations is not None:
        cov_df = fetch_covariances(cov_designations, get_json, cfg)

    n_sentry_matched = int(audit["in_sentry"].sum())
    provenance = {
        "snapshot_utc": snapshot_utc, "sbdb": sbdb_meta, "sentry": sentry_meta,
        "cleaning": clean_report, "target_mode": cfg.target_mode.value,
        "physical_policy": cfg.physical_policy.value,
        "pha_flag_vs_recomputed_disagree": int((audit["pha"] != audit["pha_recomputed"]).sum()),
        "sentry_rows": len(sentry), "sentry_matched_in_sbdb": n_sentry_matched,
        "max_abs_rank_corr_X_vs_audit": max(corr_report.values()) if corr_report else None,
        "class_balance": {str(k): int(v) for k, v in y.round(6).value_counts().head(5).items()},
    }
    return DatasetBundle(X=X, y=y, audit_df=audit, orbit_df=orbit_df, meta_df=meta_df,
                         feature_groups=groups, provenance=provenance, cov_df=cov_df)
