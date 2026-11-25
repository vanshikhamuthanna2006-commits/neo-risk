"""Measure the effect of the R1 prior update on the full model (no full re-run needed if tiny).

Recomputes the physical prior features with the cited priors, retrains the full LightGBM model
(seed 0, both modes, stratified split) with old and new features, and compares test PR-AUC and
predictions.  Writes results/prior_update_impact.json.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.metrics import average_precision_score
from src.model import LGBMConfig, LGBMModel, assemble_features, make_splits
from src.uncertainty import physical_prior_features

root = Path(__file__).resolve().parents[1]; proc = root / "data" / "processed" / "pha_default"
X = pd.read_parquet(proc / "X.parquet"); audit = pd.read_parquet(proc / "audit_df.parquet")
meta = pd.read_parquet(proc / "meta_df.parquet"); dyn = pd.read_parquet(proc / "dyn_df_50yr.parquet")
unc_old = pd.read_parquet(proc / "unc_df.parquet")
groups = json.load(open(proc / "provenance.json"))["feature_groups"]
new_phys = physical_prior_features(X)
unc_new = unc_old.copy(); unc_new[new_phys.columns] = new_phys
changed = {c: float(np.nanmax(np.abs(unc_new[c] - unc_old[c]))) for c in new_phys.columns}
out = {"max_abs_feature_change": changed, "modes": {}}
for tname, y in {"pha": pd.read_parquet(proc / "y.parquet")["y"], "sentry": audit["in_sentry"].astype(int)}.items():
    sp = make_splits(y, meta, "stratified", seed=0)
    r = {}
    for tag, unc in (("old", unc_old), ("new", unc_new)):
        F = assemble_features("static+dyn+unc", X, groups, audit, dyn, unc).F.loc[y.index]
        yv = y.to_numpy().astype(float)
        m = LGBMModel(LGBMConfig(seed=0)).fit(F.iloc[sp["train"]], yv[sp["train"]], F.iloc[sp["val"]], yv[sp["val"]])
        z = m.predict_logit(F.iloc[sp["test"]]); r[tag] = (z, average_precision_score(yv[sp["test"]], z))
    out["modes"][tname] = {"pr_auc_old": round(r["old"][1], 4), "pr_auc_new": round(r["new"][1], 4),
                           "pr_auc_delta": round(r["new"][1] - r["old"][1], 4),
                           "logit_corr": round(float(np.corrcoef(r["old"][0], r["new"][0])[0, 1]), 5)}
json.dump(out, open(root / "results" / "prior_update_impact.json", "w"), indent=1)
print(json.dumps(out["modes"], indent=1)); print({k: round(v, 3) for k, v in changed.items() if v > 0})
