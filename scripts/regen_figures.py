"""Regenerate Stage 5 figures from stored predictions (no retraining)."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import plot_conformal_tradeoff, plot_cost_curve, plot_pr_curves, plot_reliability

res = Path(__file__).resolve().parents[1] / "results"
NAMES = {"pha": "Mode A (non-circular PHA)", "sentry": "Mode B (Sentry risk list)"}
for t in ("pha", "sentry"):
    for sch in ("stratified", "temporal"):
        tag = f"{t}_{sch}"
        preds = pd.concat([pd.read_parquet(f) for f in sorted((res / "preds").glob(f"{tag}_s*.parquet"))])
        conf = pd.read_csv(res / "tables" / f"conformal_{tag}.csv")
        plot_pr_curves(preds, res / "figures" / f"pr_{tag}", title=f"{NAMES[t]} - {sch} (seed 0)")
        plot_cost_curve(preds, res / "figures" / f"cost_{tag}", title=f"{NAMES[t]} - {sch}")
        plot_conformal_tradeoff(conf, res / "figures" / f"conformal_{tag}", title=f"{NAMES[t]} - {sch}")
        plot_reliability(preds, res / "figures" / f"reliability_{tag}", title=f"{NAMES[t]} - {sch}")
print("figures regenerated")
