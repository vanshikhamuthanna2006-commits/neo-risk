"""Export data for the interactive GitHub Pages site (docs/).

python scripts/export_web.py [n_folds]

Every object gets an OUT-OF-FOLD score: K-fold (stratified by label x short-arc); inside each
fold the remaining objects are split train / val / cal, the model is trained on train (early
stopping on val), Platt-calibrated on val and conformal-calibrated on cal, then applied to the
held-out fold.  So no object is ever scored by a model that saw it.

For the conformal sets we export per-object Mondrian conformal p-values (one per class) computed
against that fold's calibration set; the browser then forms sets for any alpha exactly:
class y is in the set iff p_value_y > alpha_y.

Writes docs/data/objects.json (columnar), docs/data/summary.json.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from src.model import LGBMConfig, LGBMModel, PlattCalibrator, assemble_features, conformal_pvalues, recall_at_fpr

root = Path(__file__).resolve().parents[1]
proc = root / "data" / "processed" / "pha_default"
out_dir = root / "docs" / "data"
out_dir.mkdir(parents=True, exist_ok=True)
K = int(sys.argv[1]) if len(sys.argv) > 1 else 5

X = pd.read_parquet(proc / "X.parquet"); audit = pd.read_parquet(proc / "audit_df.parquet")
meta = pd.read_parquet(proc / "meta_df.parquet"); orbit = pd.read_parquet(proc / "orbit_df.parquet")
dyn = pd.read_parquet(proc / "dyn_df_50yr.parquet"); unc = pd.read_parquet(proc / "unc_df.parquet")
groups = json.load(open(proc / "provenance.json"))["feature_groups"]
MODELS = {"naive": "naive", "static": "static", "full": "static+dyn+unc"}
frames = {k: assemble_features(fs, X, groups, audit, dyn, unc).F for k, fs in MODELS.items()}
targets = {"pha": pd.read_parquet(proc / "y.parquet")["y"].astype(int),
           "sentry": audit["in_sentry"].astype(int)}
idx = X.index
short = meta["short_arc"].reindex(idx).fillna(0).astype(int).to_numpy()


def strat_folds(y: np.ndarray, k: int, rng) -> np.ndarray:
    fold = np.empty(len(y), int)
    strata = y * 2 + short
    for s in np.unique(strata):
        m = np.flatnonzero(strata == s)
        fold[rng.permutation(m)] = np.arange(len(m)) % k
    return fold


def strat_split(pos: np.ndarray, y: np.ndarray, fracs, rng):
    parts = [[] for _ in fracs]
    strata = y[pos] * 2 + short[pos]
    for s in np.unique(strata):
        m = rng.permutation(pos[strata == s])
        cuts = np.round(np.cumsum(fracs)[:-1] * len(m)).astype(int)
        for j, p in enumerate(np.split(m, cuts)):
            parts[j] += p.tolist()
    return [np.sort(np.array(p, int)) for p in parts]


res: dict[str, dict] = {}
cols: dict[str, np.ndarray] = {}
explain: dict[str, dict] = {}

# Plain-language feature names shown on the site (unknown names fall back to the column name).
LABELS = {
    "a": "Semi-major axis (au)", "e": "Eccentricity", "i_deg": "Inclination (°)", "q": "Perihelion distance (au)",
    "Q": "Aphelion distance (au)", "t_jup": "Tisserand parameter (Jupiter)", "sin_om": "sin node", "cos_om": "cos node",
    "sin_w": "sin arg. perihelion", "cos_w": "cos arg. perihelion", "sin_ma": "sin mean anomaly", "cos_ma": "cos mean anomaly",
    "log10_data_arc": "log10 data arc (days)", "log10_n_obs": "log10 number of observations", "n_radar_obs": "Radar observations",
    "condition_code": "Orbit condition code (0 best, 9 worst)", "rms": "Orbit-fit RMS (arcsec)", "obs_staleness_yr": "Years since last observation",
    "log10_sigma_a": "log10 σ semi-major axis", "log10_sigma_e": "log10 σ eccentricity", "log10_sigma_i": "log10 σ inclination",
    "log10_sigma_om": "log10 σ node", "log10_sigma_w": "log10 σ arg. perihelion", "log10_sigma_ma": "log10 σ mean anomaly",
    "log10_diameter_km": "log10 measured diameter (km)", "has_diameter": "Has measured diameter", "log10_rot_per_h": "log10 rotation period (h)",
    "has_rot_per": "Has rotation period", "tax_C": "Taxonomy C", "tax_S": "Taxonomy S", "tax_X": "Taxonomy X", "tax_other": "Taxonomy other",
    "has_taxonomy": "Has taxonomy",
    "dyn_da_10": "Δa after 10 yr (N-body)", "dyn_de_10": "Δe after 10 yr", "dyn_di_10": "Δi after 10 yr", "dyn_da_25": "Δa after 25 yr",
    "dyn_de_25": "Δe after 25 yr", "dyn_di_25": "Δi after 25 yr", "dyn_da_50": "Δa after 50 yr", "dyn_de_50": "Δe after 50 yr",
    "dyn_di_50": "Δi after 50 yr", "dyn_dmin_earth_au": "Closest Earth approach, 50 yr (au)", "dyn_log10_dmin_earth": "log10 closest Earth approach",
    "dyn_tmin_earth_yr": "Years until closest approach", "dyn_n_enc_005": "Earth encounters < 0.05 au", "dyn_div_log10_growth": "Orbit divergence growth",
    "dyn_lyap_proxy_per_yr": "Chaos (Lyapunov) proxy", "dyn_log10_div_max_au": "log10 max divergence (au)", "dyn_lost": "Ejected or collided",
    "unc_log10_sig_e": "log10 σ e (full covariance)", "unc_log10_sig_q": "log10 σ q (full covariance)", "unc_log10_sig_tp_days": "log10 σ perihelion time (days)",
    "unc_log10_sig_i": "log10 σ i (full covariance)", "unc_log10_lov_sigma": "log10 line-of-variations σ", "unc_log10_cov_cond": "log10 covariance condition no.",
    "unc_frac_unbound": "Unbound clone fraction", "unc_dmin_mean_au": "Clones: mean closest approach (au)", "unc_dmin_std_au": "Clones: spread of closest approach (au)",
    "unc_log10_dmin_mean": "Clones: log10 mean closest approach", "unc_log10_dmin_std": "Clones: log10 spread of closest approach",
    "unc_dmin_q05_au": "Clones: 5th-percentile closest approach (au)", "unc_p_dmin_lt_005": "Clones: P(closest < 0.05 au)",
    "unc_p_dmin_lt_001": "Clones: P(closest < 0.01 au)", "unc_tmin_std_yr": "Clones: spread of encounter time (yr)", "unc_da50_std": "Clones: spread of Δa at 50 yr",
    "unc_n_valid_clones": "Valid clones", "unc_frac_lost": "Lost clone fraction", "has_unc_clones": "Has covariance clones",
    "phys_albedo_prior_median": "Albedo prior (taxonomy)", "phys_albedo_prior_logsd": "Albedo prior spread", "phys_density_prior_mean": "Density prior (g/cm³)",
    "phys_density_prior_sd": "Density prior spread", "phys_v_inf_kms": "Encounter speed v∞ (km/s)", "phys_v_imp_kms": "Impact speed (km/s)",
    "phys_log10_e_spec_jkg": "log10 specific impact energy", "phys_log10_mass_kg_mean": "log10 mass (kg)", "phys_log10_mass_kg_sd": "log10 mass spread",
    "phys_log10_ke_mt_mean": "log10 impact energy (Mt)", "phys_log10_ke_mt_sd": "log10 impact energy spread", "has_phys_mass": "Has mass estimate",
}
for tname, ys in targets.items():
    y = ys.reindex(idx).to_numpy()
    rng = np.random.default_rng(2026)
    fold = strat_folds(y, K, rng)
    res[tname] = {"importance": {}, "curves": {}, "metrics": {}}
    for mname, F in frames.items():
        t = time.time()
        p_oof = np.full(len(y), np.nan); pv_h = np.full(len(y), np.nan); pv_b = np.full(len(y), np.nan)
        imp = pd.Series(0.0, index=F.columns)
        for f in range(K):
            te = np.flatnonzero(fold == f)
            rest = np.flatnonzero(fold != f)
            tr, va, ca = strat_split(rest, y, (0.70, 0.15, 0.15), rng)
            m = LGBMModel(LGBMConfig(seed=f)).fit(F.iloc[tr], y[tr], F.iloc[va], y[va])
            cal = PlattCalibrator().fit(m.predict_logit(F.iloc[va]), y[va])
            p_ca = cal(m.predict_logit(F.iloc[ca])); p_te = cal(m.predict_logit(F.iloc[te]))
            p_oof[te] = p_te
            # nonconformity: hazardous s = 1 - p ; benign s = p
            pv_h[te] = conformal_pvalues(1 - p_ca[y[ca] == 1], 1 - p_te)
            pv_b[te] = conformal_pvalues(p_ca[y[ca] == 0], p_te)
            g = pd.Series(m.booster.feature_importance("gain"), index=m.columns)
            imp += g / g.sum() / K
            if mname == "full":
                # per-object explanation from THIS fold's model (the one that scored these objects):
                # LightGBM TreeSHAP contributions; they plus the bias sum exactly to the raw logit
                Xte = F.iloc[te][m.columns]
                c = m.booster.predict(Xte, pred_contrib=True, num_iteration=m.booster.best_iteration)
                z = m.predict_logit(F.iloc[te])
                assert np.allclose(c.sum(axis=1) + m.init, z, atol=1e-6), "contributions do not sum to the logit"
                ex = explain.setdefault(tname, {"names": list(m.columns), "idx": np.zeros((len(y), 5), int),
                                                "val": np.zeros((len(y), 5)), "fx": np.full((len(y), 5), np.nan),
                                                "base": np.zeros(len(y)), "logit": np.zeros(len(y))})
                assert ex["names"] == list(m.columns)
                top = np.argsort(-np.abs(c[:, :-1]), axis=1)[:, :5]
                ex["idx"][te] = top
                ex["val"][te] = np.take_along_axis(c[:, :-1], top, axis=1)
                ex["fx"][te] = np.take_along_axis(Xte.to_numpy(float), top, axis=1)
                ex["base"][te] = c[:, -1] + m.init
                ex["logit"][te] = z
                ex.setdefault("fold", np.zeros(len(y), int))[te] = f
                ex.setdefault("platt", [None] * K)[f] = [cal.a, cal.b]   # p = sigmoid(a * logit + b)
        cols[f"p_{tname}_{mname}"] = p_oof
        if mname == "full":
            cols[f"pvh_{tname}"] = pv_h; cols[f"pvb_{tname}"] = pv_b
        res[tname]["importance"][mname] = imp.sort_values(ascending=False).head(15).round(4).to_dict()
        prec, rec, _ = precision_recall_curve(y, p_oof)
        keep = np.unique(np.r_[np.linspace(0, len(rec) - 1, 300).astype(int), len(rec) - 1])
        res[tname]["curves"][mname] = {"recall": rec[keep].round(4).tolist(), "precision": prec[keep].round(4).tolist()}
        res[tname]["metrics"][mname] = {"pr_auc": round(float(average_precision_score(y, p_oof)), 4),
                                        "roc_auc": round(float(roc_auc_score(y, p_oof)), 4),
                                        "recall_at_1fpr": round(recall_at_fpr(y, p_oof, 0.01), 4)}
        print(f"{tname}/{mname}: {res[tname]['metrics'][mname]}  ({time.time() - t:.0f}s)", flush=True)
    res[tname]["prevalence"] = round(float(y.mean()), 5)
    res[tname]["n_pos"] = int(y.sum())

# ---------------------------------------------------------------- per-object table (columnar)
def r(x, d):
    a = np.asarray(x, float)
    return [None if not np.isfinite(v) else round(float(v), d) for v in a]

tbl = {
    "pdes": idx.astype(str).tolist(),
    "name": meta["full_name"].reindex(idx).fillna("").str.strip().tolist(),
    "cls": meta["class"].reindex(idx).fillna("").tolist(),
    "a": r(orbit["a"].reindex(idx), 4), "e": r(orbit["e"].reindex(idx), 4), "i": r(orbit["i"].reindex(idx), 2),
    "q": r(X["q"], 4),
    "arc": r(meta["data_arc"].reindex(idx), 0), "short": short.tolist(),
    "first": meta["first_obs"].reindex(idx).astype(str).str[:10].tolist(),
    "diam": r(10 ** X["log10_diameter_km"].where(X["has_diameter"] == 1), 3),
    "H": r(audit["H"], 2), "moid": r(audit["moid"], 5),
    "pha": audit["pha"].fillna(0).astype(int).tolist(), "sentry": audit["in_sentry"].astype(int).tolist(),
    "ip": [None if not np.isfinite(v) else float(f"{v:.3g}") for v in audit["ip"].to_numpy(float)],
    "ps": r(audit["ps_cum"], 2),
    "dmin": r(dyn["dyn_dmin_earth_au"].reindex(idx), 5), "tmin": r(dyn["dyn_tmin_earth_yr"].reindex(idx), 2),
    "uq05": r(unc["unc_dmin_q05_au"].reindex(idx), 5), "up05": r(unc["unc_p_dmin_lt_005"].reindex(idx), 3),
    "usig": r(unc["unc_dmin_std_au"].reindex(idx), 5), "hasu": unc["has_unc_clones"].reindex(idx).fillna(0).astype(int).tolist(),
    "vimp": r(unc["phys_v_imp_kms"].reindex(idx), 2),
}
for k, v in cols.items():
    tbl[k] = r(v, 5)
json.dump(tbl, open(out_dir / "objects.json", "w"), separators=(",", ":"), allow_nan=False)

# ---------------------------------------------------------------- per-object explanations
def sig4(a):
    return [None if not np.isfinite(v) else float(f"{v:.4g}") for v in np.asarray(a, float).ravel()]

xout = {"note": "TreeSHAP contributions to the full model's log-odds (before Platt calibration), from the "
                "cross-validation fold model that scored each object; base + all contributions = logit.",
        "labels": {}}
for tname, ex in explain.items():
    xout["labels"] = {c: LABELS.get(c, c) for c in ex["names"]}
    xout[tname] = {"names": ex["names"], "idx": ex["idx"].ravel().tolist(), "val": np.round(ex["val"], 4).ravel().tolist(),
                   "fx": sig4(ex["fx"]), "base": np.round(ex["base"], 4).tolist(), "logit": np.round(ex["logit"], 6).tolist(),
                   "fold": ex["fold"].tolist(), "platt": [[float(a), float(b)] for a, b in ex["platt"]]}
# split into 1,000-object chunks so a card loads ~200 KB instead of the whole file
CH = 1000
xd = out_dir / "explain"; xd.mkdir(exist_ok=True)
for old in xd.glob("*.json"):
    old.unlink()
index = {"note": xout["note"], "labels": xout["labels"], "chunk": CH, "n": len(idx), "modes": {}}
for tname in explain:
    xm = xout[tname]
    index["modes"][tname] = {"names": xm["names"], "platt": xm["platt"]}
    for c0 in range(0, len(idx), CH):
        c1 = min(len(idx), c0 + CH)
        part = {"start": c0, "idx": xm["idx"][c0 * 5:c1 * 5], "val": xm["val"][c0 * 5:c1 * 5], "fx": xm["fx"][c0 * 5:c1 * 5],
                "base": xm["base"][c0:c1], "logit": xm["logit"][c0:c1], "fold": xm["fold"][c0:c1]}
        json.dump(part, open(xd / f"{tname}_{c0 // CH}.json", "w"), separators=(",", ":"), allow_nan=False)
json.dump(index, open(xd / "index.json", "w"), separators=(",", ":"), allow_nan=False)
(out_dir / "explain.json").unlink(missing_ok=True)

# ---------------------------------------------------------------- official test-split tables
tabs = {}
for tname in ("pha", "sentry"):
    for scheme in ("stratified", "temporal"):
        for stratum in ("all", "short_arc"):
            f = root / "results" / "tables" / f"summary_{tname}_{scheme}_{stratum}.csv"
            if f.exists():
                s = pd.read_csv(f, index_col=0)
                keep = [c for c in ("pr_auc", "pr_auc_lo", "pr_auc_hi", "roc_auc", "recall_at_fpr", "cost_per_1k",
                                    "brier", "d_pr_auc", "d_pr_auc_lo", "d_pr_auc_hi") if c in s]
                tabs[f"{tname}_{scheme}_{stratum}"] = s[keep].round(4).reset_index().to_dict("records")
res["tables"] = tabs
res["stage3"] = json.load(open(root / "results" / "stage3_report.json"))
res["n_objects"] = len(idx); res["n_folds"] = K
# plain-language risk context from the same JPL Sentry snapshot the labels came from
prov = json.load(open(proc / "provenance.json"))
sen = audit[audit["in_sentry"] == 1]
pmax = sen["ps_cum"].idxmax()
res["sentry_context"] = {"snapshot_utc": prov.get("snapshot_utc"), "n_on_list": int(len(sen)),
                         "n_torino_rated": int(sen["ts_max"].notna().sum()), "torino_max": float(sen["ts_max"].max()),
                         "palermo_cum_max": float(sen["ps_cum"].max()), "palermo_cum_max_des": str(pmax),
                         "palermo_cum_max_name": str(meta.loc[pmax, "full_name"]).strip()}
def clean(o):
    """JSON for browsers: NaN/inf -> null (json.dump would emit bare NaN, which JSON.parse rejects)."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


json.dump(clean(res), open(out_dir / "summary.json", "w"), separators=(",", ":"), allow_nan=False)
print("wrote", out_dir, {p.name: f"{p.stat().st_size / 1e6:.1f} MB" for p in out_dir.iterdir()})
