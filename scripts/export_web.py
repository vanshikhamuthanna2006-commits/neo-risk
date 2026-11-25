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
