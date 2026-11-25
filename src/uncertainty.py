"""Stage 3: covariance Monte-Carlo clones and physical (taxonomic) priors.

A. Orbital uncertainty
----------------------
* Clones are drawn from N(mu, Sigma) in JPL's *cometary* elements (e, q, tp, Omega, omega, i) at
  the covariance's own epoch (often years older than the SBDB-query orbit epoch).
* Sampling uses an eigen-decomposition in correlation space (median condition number ~1e8;
  Cholesky is fragile there) with antithetic pairs (z, -z): the clone mean equals mu exactly.
* Per-object RNG seeded from (global seed, CRC32(designation)): results are independent of
  batching and ordering.
* Clones with e >= 1, q <= 0 or i outside [0, 180] deg are *counted* (``unc_frac_unbound``),
  not resampled: resampling would truncate the distribution.
* Valid clones are aligned to t0 and propagated with the Stage 2 engine (stride=1).
* Limitation: a Gaussian in elements is a linearised approximation; for very short arcs the
  true uncertainty region is curved (banana-shaped).  The unbound fraction flags these cases.

B. Physical priors
------------------
* Albedo (lognormal) and bulk density priors by taxonomic complex; unknown taxonomy uses a
  population mixture.  NO size is inferred from H (H is a label-defining quantity in Mode A).
* Specific impact energy from the Opik encounter velocity, U^2 = 3 - T_E (a, e, i only).
* Mass / kinetic-energy distributions only where a diameter is *measured* (mask otherwise).
* Prior values for the S and C complexes come from Thomas et al. 2011 (albedo) and Carry 2012
  (density); the X complex, "other" and the unknown-taxonomy mixture are stated assumptions (see
  ``TAXON_PRIORS``).  All are configurable.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import COV_LABELS
from .dynamics import (MU_SUN, YEAR, DynamicsConfig, _run_task, align_elements, needs_ias15)
from .orbital import cartesian_to_aei

AU_KM = 1.495978707e8
V_EARTH_KMS = 29.7847          # mean heliocentric orbital speed of Earth
V_ESC_KMS = 11.186             # Earth escape speed at the surface
MT_TNT_J = 4.184e15


# ======================================================================================
# A. Covariance clones
# ======================================================================================
@dataclass
class UncertaintyConfig:
    n_clones: int = 32
    seed: int = 20261122
    objects_per_batch: int = 16             # 16 x 32 = 512 particles per MERCURIUS sim
    dyn: DynamicsConfig = field(default_factory=lambda: DynamicsConfig(horizons=(50.0,)))
    recenter: bool = True     # shift clone cloud onto the Stage-2 nominal state at t0

    def __post_init__(self) -> None:
        if self.n_clones % 2:
            raise ValueError("n_clones must be even (antithetic sampling)")


def cov_matrix(row: pd.Series) -> np.ndarray:
    """Rebuild the 6x6 cometary covariance from the flattened ``cov_df`` columns."""
    M = np.empty((6, 6))
    for a in range(6):
        for b in range(a, 6):
            M[a, b] = M[b, a] = row[f"cov_{COV_LABELS[a]}_{COV_LABELS[b]}"]
    return M


def cov_nominal(row: pd.Series) -> np.ndarray:
    return np.array([row[f"cov_nom_{lab}"] for lab in COV_LABELS], float)


def object_rng(pdes: str, seed: int) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(pdes.encode())])


def sample_cometary(mu: np.ndarray, cov: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k antithetic samples from N(mu, cov); robust to near-singular (PSD) covariances."""
    d = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    safe = np.where(d > 0, d, 1.0)
    corr = cov / np.outer(safe, safe)
    w, V = np.linalg.eigh(0.5 * (corr + corr.T))
    L = V * np.sqrt(np.clip(w, 0.0, None))              # corr = L L^T
    z = rng.standard_normal((k // 2, 6))
    z = np.vstack([z, -z])
    return mu + (z @ L.T) * d


def cometary_to_keplerian(x: np.ndarray, epoch_jd: float) -> tuple[np.ndarray, np.ndarray]:
    """(k, 6) [e, q, tp, node, peri, i] (deg, JD) -> Keplerian (a, e, i, Omega, omega, M) [rad].

    Returns elements and a validity mask (bound ellipse, q > 0, 0 <= i <= 180 deg).
    """
    e, q, tp, node, peri, inc = x.T
    valid = (e >= 0) & (e < 1) & (q > 0) & (inc >= 0) & (inc <= 180) & np.all(np.isfinite(x), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = q / (1.0 - e)
        n = np.sqrt(MU_SUN / a ** 3)                    # rad / day
        M = np.mod(n * (epoch_jd - tp), 2 * np.pi)
    el = np.column_stack([a, e, np.deg2rad(inc), np.deg2rad(np.mod(node, 360)),
                          np.deg2rad(np.mod(peri, 360)), M])
    return el, valid


def covariance_shape_features(row: pd.Series) -> dict[str, float]:
    """Cheap (no propagation) descriptors of the 6x6 covariance."""
    C = cov_matrix(row)
    mu = cov_nominal(row)
    e, q = mu[0], mu[1]
    a = q / (1 - e) if e < 1 else np.nan
    n = np.sqrt(MU_SUN / a ** 3) if np.isfinite(a) else np.nan
    # natural units: e, q [au], along-track angle n*tp [rad], angles [rad]
    J = np.diag([1.0, 1.0, n, *(np.pi / 180,) * 3])
    Cn = J @ C @ J
    with np.errstate(divide="ignore", invalid="ignore"):
        lov = float(np.sqrt(np.max(np.linalg.eigvalsh(0.5 * (Cn + Cn.T))))) if np.isfinite(n) else np.nan
        return {
            "unc_log10_sig_e": np.log10(np.sqrt(C[0, 0])),
            "unc_log10_sig_q": np.log10(np.sqrt(C[1, 1])),
            "unc_log10_sig_tp_days": np.log10(np.sqrt(C[2, 2])),
            "unc_log10_sig_i": np.log10(np.sqrt(C[5, 5])),
            "unc_log10_lov_sigma": np.log10(lov),
            "unc_log10_cov_cond": np.log10(row["cov_cond"]),
        }


def draw_clones(cov_df: pd.DataFrame, cfg: UncertaintyConfig) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Sample K clones for every object with a valid covariance.

    Returns a long frame (owner, k, cov_epoch, a, e, i, Omega, omega, M) of *valid* clones and
    per-object summary dicts (covariance-shape features + unbound fraction).
    """
    rows, summ = [], {}
    for pdes, r in cov_df[cov_df["has_cov"].astype(bool)].iterrows():
        x = sample_cometary(cov_nominal(r), cov_matrix(r), cfg.n_clones, object_rng(pdes, cfg.seed))
        el, valid = cometary_to_keplerian(x, float(r["cov_epoch"]))
        s = covariance_shape_features(r)
        s["unc_frac_unbound"] = 1.0 - valid.mean()
        summ[pdes] = s
        for k in np.flatnonzero(valid):
            rows.append((pdes, int(k), float(r["cov_epoch"]), *el[k]))
        c_el, c_ok = cometary_to_keplerian(cov_nominal(r)[None, :], float(r["cov_epoch"]))
        if c_ok[0]:                                          # k = -1: covariance nominal (centre)
            rows.append((pdes, -1, float(r["cov_epoch"]), *c_el[0]))
    clones = pd.DataFrame(rows, columns=["owner", "k", "cov_epoch", "a", "e", "i", "Omega", "omega", "M"])
    return clones, summ


def _align_chunk(groups: list[tuple[float, np.ndarray]], t0: float, dcfg: DynamicsConfig):
    return [align_elements(ep, t0, el, dcfg) for ep, el in groups]


def align_many(epochs: np.ndarray, el: np.ndarray, t0: float,
               dcfg: DynamicsConfig) -> tuple[np.ndarray, np.ndarray]:
    """Align particles with arbitrary epochs to t0 (grouped by epoch, parallel chunks)."""
    from joblib import Parallel, delayed

    order = np.argsort(epochs, kind="stable")
    uniq, starts = np.unique(epochs[order], return_index=True)
    bounds = list(starts) + [len(order)]
    groups = [(float(uniq[g]), order[bounds[g]:bounds[g + 1]]) for g in range(len(uniq))]
    n_chunks = max(1, min(len(groups), 64))
    idx_chunks = [groups[i::n_chunks] for i in range(n_chunks)]
    res = Parallel(n_jobs=dcfg.n_jobs)(
        delayed(_align_chunk)([(e_, el[idx]) for e_, idx in ch], t0, dcfg) for ch in idx_chunks)
    states = np.empty((len(el), 6))
    lost = np.zeros(len(el), bool)
    for ch, rch in zip(idx_chunks, res):
        for (_, idx), (st, lo) in zip(ch, rch):
            states[idx], lost[idx] = st, lo
    return states, lost


def extract_uncertainty_features(cov_df: pd.DataFrame, index: pd.Index, cfg: UncertaintyConfig | None = None,
                                 t0_jd: float | None = None,
                                 orbit_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Uncertainty features for every designation in ``index`` (NaN + mask where no covariance).

    With ``cfg.recenter`` and ``orbit_df`` (Stage-1 orbits), each clone's offset from the
    propagated covariance nominal at t0 is added to the Stage-2 nominal state at t0, so the
    dispersion comes from the covariance while the centre matches JPL's (full-force-model) orbit.
    """
    from joblib import Parallel, delayed

    cfg = cfg or UncertaintyConfig()
    dcfg = cfg.dyn
    t0 = t0_jd if t0_jd is not None else (dcfg.t0_jd or 2461200.5)
    sub = cov_df.reindex(index)
    sub = sub[sub["has_cov"].fillna(False).astype(bool)]
    clones, summ = draw_clones(sub, cfg)
    out = pd.DataFrame(index=index)
    if clones.empty:
        out["has_unc_clones"] = 0.0
        return out

    # ---- align clones (and covariance-nominal centres, k = -1) from cov epochs to t0
    el = clones[["a", "e", "i", "Omega", "omega", "M"]].to_numpy()
    states, lost0 = align_many(clones["cov_epoch"].to_numpy(), el, t0, dcfg)

    # ---- optional re-centring onto the Stage-2 nominal state at t0
    is_c = (clones["k"] == -1).to_numpy()
    if cfg.recenter and orbit_df is not None:
        from .dynamics import to_rad_elements
        own = clones["owner"].to_numpy()
        cen = pd.Series(np.flatnonzero(is_c), index=own[is_c])
        orb = orbit_df.reindex(cen.index)
        ok_orb = orb[["epoch", "a", "e", "i", "om", "w", "ma"]].notna().all(axis=1).to_numpy()
        nom_st = np.full((len(cen), 6), np.nan)
        nom_lost = np.ones(len(cen), bool)
        if ok_orb.any():
            nom_st[ok_orb], nom_lost[ok_orb] = align_many(
                orb["epoch"].to_numpy()[ok_orb], to_rad_elements(orb[ok_orb]), t0, dcfg)
        shift = pd.DataFrame(nom_st - states[cen.to_numpy()], index=cen.index)
        bad = pd.Series(nom_lost | lost0[cen.to_numpy()], index=cen.index)
        has_c = np.isin(own, cen.index)
        rows = np.flatnonzero(has_c & ~is_c)
        states[rows] += shift.loc[own[rows]].to_numpy()
        lost0[rows] |= bad.loc[own[rows]].to_numpy()
        lost0[~has_c] = True                     # no valid centre -> cannot re-centre
    keep = ~is_c
    clones, states, lost0 = clones[keep].reset_index(drop=True), states[keep], lost0[keep]

    # ---- propagation tasks: bulk MERCURIUS batches of whole objects; IAS15 per object
    a0, e0, _ = cartesian_to_aei(states[:, :3], states[:, 3:], MU_SUN)
    bound = (e0 < 1) & (a0 > 0)
    el0 = np.column_stack([np.where(bound, a0, 1.0), np.where(bound, e0, 0.0)])
    needy = needs_ias15(el0, dcfg) | (dcfg.integrator == "ias15")
    lost0 |= ~bound
    ids = (clones["owner"] + "#" + clones["k"].astype(str)).to_numpy()
    owners = clones["owner"].to_numpy()
    tasks: list[dict] = []

    def task(sel: np.ndarray, integ: str) -> dict:
        return {"pdes": list(ids[sel]), "states": states[sel], "has_clone": np.zeros(len(sel), bool),
                "lost0": lost0[sel], "t0": t0, "integ": integ}

    uo = pd.unique(owners)
    bulk_mask = ~needy
    for i in range(0, len(uo), cfg.objects_per_batch):
        sel = np.flatnonzero(np.isin(owners, uo[i:i + cfg.objects_per_batch]) & bulk_mask)
        if len(sel):
            tasks.append(task(sel, dcfg.integrator if dcfg.integrator != "ias15" else "ias15"))
    for o in pd.unique(owners[needy]):
        tasks.append(task(np.flatnonzero((owners == o) & needy), "ias15"))
    tasks.sort(key=lambda t: -len(t["pdes"]))
    per = pd.concat(Parallel(n_jobs=dcfg.n_jobs)(delayed(_run_task)(t, dcfg) for t in tasks))
    per["owner"] = per.index.str.rsplit("#", n=1).str[0]

    # ---- aggregate per object
    h = f"{max(dcfg.horizons):g}"
    ok = per[per["dyn_lost"] == 0]
    g = ok.groupby("owner")
    dmin = g["dyn_dmin_earth_au"]
    agg = pd.DataFrame({
        "unc_dmin_mean_au": dmin.mean(),
        "unc_dmin_std_au": dmin.std(ddof=1),
        "unc_log10_dmin_mean": g["dyn_log10_dmin_earth"].mean(),
        "unc_log10_dmin_std": g["dyn_log10_dmin_earth"].std(ddof=1),
        "unc_dmin_q05_au": dmin.quantile(0.05),
        "unc_p_dmin_lt_005": g["dyn_dmin_earth_au"].apply(lambda s: float((s < 0.05).mean())),
        "unc_p_dmin_lt_001": g["dyn_dmin_earth_au"].apply(lambda s: float((s < 0.01).mean())),
        "unc_tmin_std_yr": g["dyn_tmin_earth_yr"].std(ddof=1),
        f"unc_da{h}_std": g[f"dyn_da_{h}"].std(ddof=1),
        "unc_n_valid_clones": g.size().astype(float),
    })
    n_tot = per.groupby("owner").size()
    agg["unc_frac_lost"] = 1.0 - agg["unc_n_valid_clones"] / n_tot.reindex(agg.index)
    shape = pd.DataFrame.from_dict(summ, orient="index")
    out = out.join(shape).join(agg)
    out["has_unc_clones"] = (out["unc_n_valid_clones"].fillna(0) >= 2).astype(float)
    feat = [c for c in out.columns if c.startswith("unc_")]
    out.loc[out["has_unc_clones"] == 0, [c for c in feat if c in agg.columns]] = np.nan
    return out


# ======================================================================================
# B. Physical priors (taxonomy; no H)
# ======================================================================================
# Approximate literature values -- VERIFY before publication:
#   albedo: NEO thermal surveys (e.g. Thomas et al. 2011 AJ 142:85; Mainzer et al. 2011 ApJ 743:156)
#   density: Carry (2012) PSS 73:98 bulk-density compilation
# Sources
#   Albedo (NEO population): Thomas et al. 2011, AJ 142, 85 (ExploreNEOs V): S 0.26 (+0.04/-0.03),
#     C 0.13 (+0.06/-0.05; only 5 objects, hence the wide spread), Q 0.29, V 0.42, D 0.02.
#   Density: Carry 2012, Planet. Space Sci. 73, 98, Table 3 (estimates better than 20%):
#     S 2.72 +- 0.54, C 1.33 +- 0.58 g/cm3.
#   X complex: deliberately broad, an ASSUMPTION spanning its sub-types. Albedo goes from P (~0.05) to
#     E (~0.5); Carry 2012 densities are X 1.85, Xe 2.60, Xk 4.22, Xc 4.86.
#   "other" and UNKNOWN_MIXTURE are assumptions (not from a single source); they only set the
#     prior for objects without a taxonomy and are reported as such.
# The tree models depend on these only through per-class constants (albedo order C < X < S) and
# through the mass and energy priors of the ~2% of objects with a measured diameter.
TAXON_PRIORS: dict[str, dict[str, float]] = {
    "C": {"albedo_median": 0.13, "albedo_sigma_ln": 0.45, "density_mean": 1.33, "density_sd": 0.58},
    "S": {"albedo_median": 0.26, "albedo_sigma_ln": 0.30, "density_mean": 2.72, "density_sd": 0.54},
    "X": {"albedo_median": 0.15, "albedo_sigma_ln": 0.70, "density_mean": 2.6, "density_sd": 1.0},
    "other": {"albedo_median": 0.15, "albedo_sigma_ln": 0.80, "density_mean": 2.0, "density_sd": 0.8},
}
UNKNOWN_MIXTURE: dict[str, float] = {"C": 0.35, "S": 0.45, "X": 0.15, "other": 0.05}


@dataclass
class PhysicalConfig:
    n_samples: int = 4000
    seed: int = 7
    diam_frac_sigma: float = 0.15           # assumed 1-sigma fractional diameter error
    use_diameter: bool = True               # False under PhysicalPolicy.STRICT


def opik_encounter_velocity(a: np.ndarray, e: np.ndarray, i_deg: np.ndarray) -> np.ndarray:
    """Opik/Tisserand encounter speed U (units of Earth's orbital speed), clipped at 0."""
    t_e = 1.0 / a + 2.0 * np.sqrt(np.clip(a * (1 - e ** 2), 0, None)) * np.cos(np.deg2rad(i_deg))
    return np.sqrt(np.clip(3.0 - t_e, 0.0, None))


def _lognormal_from_mean_sd(mean: float, sd: float) -> tuple[float, float]:
    s2 = np.log(1 + (sd / mean) ** 2)
    return np.log(mean) - 0.5 * s2, np.sqrt(s2)


def _class_samples(cls: str, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Samples of (albedo, density[g/cm3]) for a complex, or the population mixture if unknown."""
    if cls not in TAXON_PRIORS:
        keys = list(UNKNOWN_MIXTURE)
        pick = rng.choice(len(keys), size=n, p=np.array(list(UNKNOWN_MIXTURE.values())))
        alb, rho = np.empty(n), np.empty(n)
        for j, k in enumerate(keys):
            m = pick == j
            alb[m], rho[m] = _class_samples(k, int(m.sum()), rng)
        return alb, rho
    p = TAXON_PRIORS[cls]
    alb = p["albedo_median"] * np.exp(p["albedo_sigma_ln"] * rng.standard_normal(n))
    mu, s = _lognormal_from_mean_sd(p["density_mean"], p["density_sd"])
    return alb, np.exp(mu + s * rng.standard_normal(n))


def physical_prior_features(X: pd.DataFrame, cfg: PhysicalConfig | None = None) -> pd.DataFrame:
    """Prior/derived physical features from the Stage 1 feature matrix (no H, no imputation).

    Needs: a, e, i_deg, tax_{C,S,X,other}, has_taxonomy; optionally log10_diameter_km.
    """
    cfg = cfg or PhysicalConfig()
    out = pd.DataFrame(index=X.index)
    cls = np.full(len(X), "unknown", dtype=object)
    for k in ("C", "S", "X", "other"):
        if f"tax_{k}" in X:
            cls[X[f"tax_{k}"].to_numpy() == 1] = k

    # class-level prior summaries (computed once per class with a fixed RNG)
    summ = {}
    for c in ("C", "S", "X", "other", "unknown"):
        alb, rho = _class_samples(c, 200_000, np.random.default_rng(cfg.seed))
        summ[c] = (np.median(alb), np.std(np.log(alb)), rho.mean(), rho.std())
    S = np.array([summ[c] for c in cls])
    out["phys_albedo_prior_median"], out["phys_albedo_prior_logsd"] = S[:, 0], S[:, 1]
    out["phys_density_prior_mean"], out["phys_density_prior_sd"] = S[:, 2], S[:, 3]

    U = opik_encounter_velocity(X["a"].to_numpy(), X["e"].to_numpy(), X["i_deg"].to_numpy())
    v_inf = U * V_EARTH_KMS
    v_imp = np.sqrt(v_inf ** 2 + V_ESC_KMS ** 2)
    out["phys_v_inf_kms"] = v_inf
    out["phys_v_imp_kms"] = v_imp
    out["phys_log10_e_spec_jkg"] = np.log10(0.5 * (v_imp * 1e3) ** 2)

    has_d = np.zeros(len(X), bool)
    if cfg.use_diameter and "log10_diameter_km" in X:
        has_d = X["log10_diameter_km"].notna().to_numpy()
    lm_mu, lm_sd = np.full(len(X), np.nan), np.full(len(X), np.nan)
    lk_mu, lk_sd = np.full(len(X), np.nan), np.full(len(X), np.nan)
    for j in np.flatnonzero(has_d):
        rng = np.random.default_rng([cfg.seed, j])
        d_km = 10 ** X["log10_diameter_km"].iat[j]
        D = d_km * np.exp(cfg.diam_frac_sigma * rng.standard_normal(cfg.n_samples)) * 1e3   # m
        _, rho = _class_samples(cls[j], cfg.n_samples, rng)
        mass = np.pi / 6 * D ** 3 * rho * 1e3                                           # kg
        ke_mt = 0.5 * mass * (v_imp[j] * 1e3) ** 2 / MT_TNT_J
        lm, lk = np.log10(mass), np.log10(ke_mt)
        lm_mu[j], lm_sd[j], lk_mu[j], lk_sd[j] = lm.mean(), lm.std(), lk.mean(), lk.std()
    out["phys_log10_mass_kg_mean"], out["phys_log10_mass_kg_sd"] = lm_mu, lm_sd
    out["phys_log10_ke_mt_mean"], out["phys_log10_ke_mt_sd"] = lk_mu, lk_sd
    out["has_phys_mass"] = has_d.astype(float)
    return out


def uncertainty_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("unc_", "phys_", "has_unc", "has_phys"))]
