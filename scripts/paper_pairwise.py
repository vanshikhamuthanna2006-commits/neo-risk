"""Paired bootstrap tests BETWEEN model variants (not only against the static reference).

python scripts/paper_pairwise.py   ->  results/pairwise_tests.json

Same protocol as src.evaluate (stratified bootstrap with identical indices for every model within a
seed, 500 replicates, replicate-wise averaging over the 5 seeds): Delta = A - B on PR-AUC, recall at
1% FPR and Brier score, 95% interval, and the fraction of replicates in which A is not better than B.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from src.evaluate import EvalConfig, bootstrap_seed, point_metrics

PAIRS = [("static_dyn_unc", "static_dyn", "uncertainty beyond dynamics"),
         ("static_dyn_unc_focal", "static_dyn_unc", "focal loss vs log-loss"),
         ("proposed_ensemble", "static_dyn_unc", "ensemble vs LightGBM"),
         ("dual_resmlp_focal", "static_dyn_unc", "MLP vs LightGBM")]
MODELS = sorted({m for a, b, _ in PAIRS for m in (a, b)})
meta = pd.read_parquet(root / "data" / "processed" / "pha_default" / "meta_df.parquet")
cfg = EvalConfig(n_boot=500)
out = {}
for mode in ("sentry", "pha"):
    for split in ("stratified", "temporal"):
        preds = pd.concat([pd.read_parquet(root / "results" / "preds" / f"{mode}_{split}_s{s}.parquet") for s in range(5)], ignore_index=True)
        test = preds[(preds.split == "test") & preds.model.isin(MODELS)].copy()
        for stratum in ("all", "short_arc"):
            sub = test if stratum == "all" else test[meta["short_arc"].reindex(test.pdes).to_numpy() == 1]
            point = pd.DataFrame([{"seed": s, "model": m, **point_metrics(d.y.to_numpy(), d.logit.to_numpy(), d.p.to_numpy(), cfg)}
                                  for (s, m), d in sub.groupby(["seed", "model"], sort=False)])
            boot = pd.concat([bootstrap_seed(d, cfg, s).assign(seed=s) for s, d in sub.groupby("seed")], ignore_index=True)
            rep = boot.groupby(["model", "b"])[["pr_auc", "recall_at_fpr", "brier"]].mean()
            pm = point.groupby("model")[["pr_auc", "recall_at_fpr", "brier"]].mean()
            res = {}
            for a, b, label in PAIRS:
                d = rep.xs(a, level="model") - rep.xs(b, level="model")
                r = {"label": label}
                for k, better in (("pr_auc", 1), ("recall_at_fpr", 1), ("brier", -1)):
                    r[k] = round(float(pm.loc[a, k] - pm.loc[b, k]), 4)
                    r[f"{k}_lo"], r[f"{k}_hi"] = (round(float(x), 4) for x in d[k].quantile([0.025, 0.975]))
                    r[f"{k}_p_not_better"] = round(float(((better * d[k]) <= 0).mean()), 4)
                res[f"{a}__vs__{b}"] = r
            out[f"{mode}_{split}_{stratum}"] = res
            print(mode, split, stratum, {k: (v["pr_auc"], v["pr_auc_lo"], v["pr_auc_hi"]) for k, v in res.items()}, flush=True)
json.dump(out, open(root / "results" / "pairwise_tests.json", "w"), indent=1)
print("wrote results/pairwise_tests.json")
