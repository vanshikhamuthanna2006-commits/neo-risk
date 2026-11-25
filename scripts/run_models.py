"""Stage 4 production: train the full ablation ladder (stratified + temporal splits)."""
from __future__ import annotations
import json, logging, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.metrics import average_precision_score, roc_auc_score
from src.model import ExperimentConfig, MondrianConformal, SET_BOTH, SET_EMPTY, assemble_features, make_splits, run_ablations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
root = Path(__file__).resolve().parents[1]
proc = root / "data" / "processed" / "pha_default"
X = pd.read_parquet(proc / "X.parquet"); audit = pd.read_parquet(proc / "audit_df.parquet")
target = sys.argv[1] if len(sys.argv) > 1 else "pha"          # "pha" (Mode A) | "sentry" (Mode B)
if target == "pha":
    y = pd.read_parquet(proc / "y.parquet")["y"]
elif target == "sentry":
    y = audit["in_sentry"].astype(int).rename("y")              # Mode B: on the Sentry risk list
else:
    raise SystemExit(f"unknown target {target}")
meta = pd.read_parquet(proc / "meta_df.parquet")
dyn = pd.read_parquet(proc / "dyn_df_50yr.parquet"); unc = pd.read_parquet(proc / "unc_df.parquet")
groups = json.load(open(proc / "provenance.json"))["feature_groups"]
frames = {fs: assemble_features(fs, X, groups, audit, dyn, unc)
          for fs in ("naive", "static_orbit", "static", "static+dyn", "static+dyn+unc")}
cfg = ExperimentConfig()
out_dir = root / "results"; out_dir.mkdir(exist_ok=True)
summary = {"target": target, "unc_coverage": float(unc["has_unc_clones"].mean()), "n": len(y), "prevalence": float(y.mean())}
for scheme in ("stratified", "temporal"):
    t = time.time()
    splits = make_splits(y, meta, scheme, seed=0)
    preds, arte = run_ablations(frames, y, splits, cfg)
    preds["scheme"] = scheme
    preds.to_parquet(out_dir / f"predictions_{target}_{scheme}.parquet")
    json.dump(arte, open(out_dir / f"artefacts_{target}_{scheme}.json", "w"), indent=1, default=str)
    te = preds[preds.split == "test"]
    rows = {}
    for m, d in te.groupby("model", sort=False):
        cov = MondrianConformal.covers(d.conf_set.to_numpy(), d.y.to_numpy())
        rows[m] = {"PR-AUC": average_precision_score(d.y, d.p), "ROC-AUC": roc_auc_score(d.y, d.p),
                   "haz_cov": cov[d.y == 1].mean(), "ben_cov": cov[d.y == 0].mean(),
                   "frac_uncertain": float(np.isin(d.conf_set, [SET_BOTH, SET_EMPTY]).mean())}
    tab = pd.DataFrame(rows).T
    summary[scheme] = {"wall_s": round(time.time() - t, 1), "test_n": int(len(splits["test"])),
                       "test_pos": int(y.iloc[splits["test"]].sum()), "table": tab.round(4).to_dict(orient="index")}
    print(f"\n=== {scheme} split (test n={len(splits['test'])}, positives={int(y.iloc[splits['test']].sum())}) ===")
    print(tab.round(4).to_string())
json.dump(summary, open(out_dir / f"stage4_summary_{target}.json", "w"), indent=1)
