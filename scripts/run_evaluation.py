"""Stage 5 production: multi-seed training of the ablation ladder + full evaluation suite.

python scripts/run_evaluation.py [n_seeds] [n_boot]
Writes results/preds/*.parquet, results/tables/*.tex (+ results/ablation_table.tex),
results/figures/*.{pdf,png}, results/stage5_summary.json.
"""
from __future__ import annotations
import json, logging, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import (EvalConfig, evaluate_predictions, latex_ablation_table, latex_conformal_table,
                          plot_conformal_tradeoff, plot_cost_curve, plot_pr_curves, plot_reliability)
from src.model import ExperimentConfig, LGBMConfig, MLPConfig, assemble_features, make_splits, run_ablations

logging.basicConfig(level=logging.WARNING)
root = Path(__file__).resolve().parents[1]
proc = root / "data" / "processed" / "pha_default"
res = root / "results"
n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
n_boot = int(sys.argv[2]) if len(sys.argv) > 2 else 500
X = pd.read_parquet(proc / "X.parquet"); audit = pd.read_parquet(proc / "audit_df.parquet")
meta = pd.read_parquet(proc / "meta_df.parquet")
dyn = pd.read_parquet(proc / "dyn_df_50yr.parquet"); unc = pd.read_parquet(proc / "unc_df.parquet")
groups = json.load(open(proc / "provenance.json"))["feature_groups"]
frames = {fs: assemble_features(fs, X, groups, audit, dyn, unc)
          for fs in ("naive", "static_orbit", "static", "static+dyn", "static+dyn+unc")}
targets = {"pha": pd.read_parquet(proc / "y.parquet")["y"], "sentry": audit["in_sentry"].astype(int).rename("y")}
NAMES = {"pha": "Mode A (non-circular PHA)", "sentry": "Mode B (Sentry risk list)"}
summary = {"unc_coverage": float(unc["has_unc_clones"].mean()), "n_seeds": n_seeds, "n_boot": n_boot}
(res / "preds").mkdir(parents=True, exist_ok=True)
for tname, y in targets.items():
    for scheme in ("stratified", "temporal"):
        t0 = time.time()
        parts = []
        for seed in range(n_seeds):
            fp = res / "preds" / f"{tname}_{scheme}_s{seed}.parquet"
            if fp.exists():
                parts.append(pd.read_parquet(fp)); continue
            cfg = ExperimentConfig(lgbm=LGBMConfig(seed=seed), mlp=MLPConfig(seed=seed))
            preds, _ = run_ablations(frames, y, make_splits(y, meta, scheme, seed=seed), cfg)
            preds["seed"] = seed
            preds.to_parquet(fp); parts.append(preds)
            print(f"  trained {tname}/{scheme} seed {seed}", flush=True)
        preds = pd.concat(parts, ignore_index=True)
        out = evaluate_predictions(preds, meta, EvalConfig(n_boot=n_boot))
        tag = f"{tname}_{scheme}"
        triv = float(out["trivial_cost_per_1k"].cost.mean())
        cap = f"{NAMES[tname]}, {scheme} split: test performance over {n_seeds} seeds"
        (res / "tables").mkdir(exist_ok=True)
        for stratum in ("all", "short_arc"):
            if f"summary_{stratum}" not in out:
                continue
            s = out[f"summary_{stratum}"]
            s.to_csv(res / "tables" / f"summary_{tag}_{stratum}.csv")
            title = cap + (" (short-arc NEOs, data arc $<30$ d)" if stratum == "short_arc" else "") + "."
            tex = latex_ablation_table(s, title, f"tab:{tag}_{stratum}", counts=out[f"counts_{stratum}"],
                                       trivial_cost=triv if stratum == "all" else None)
            (res / "tables" / f"ablation_{tag}_{stratum}.tex").write_text(tex)
        (res / "tables" / f"conformal_{tag}.tex").write_text(latex_conformal_table(
            out["conformal"], ("static_leakfree", "proposed_ensemble"),
            f"{NAMES[tname]}, {scheme} split: Mondrian conformal coverage vs follow-up burden.", f"tab:conf_{tag}"))
        out["conformal"].to_csv(res / "tables" / f"conformal_{tag}.csv", index=False)
        figs = res / "figures"
        plot_pr_curves(preds, figs / f"pr_{tag}", title=f"{NAMES[tname]} - {scheme} (seed 0)")
        plot_cost_curve(preds, figs / f"cost_{tag}", title=f"{NAMES[tname]} - {scheme}")
        plot_conformal_tradeoff(out["conformal"], figs / f"conformal_{tag}", title=f"{NAMES[tname]} - {scheme}")
        plot_reliability(preds, figs / f"reliability_{tag}", title=f"{NAMES[tname]} - {scheme}")
        sa = out["summary_all"]
        summary[tag] = {"wall_s": round(time.time() - t0, 1), "trivial_cost_per_1k": triv,
                        "counts": out["counts_all"].mean().round(1).to_dict(),
                        "short_arc_counts": out["counts_short_arc"].mean().round(1).to_dict() if "counts_short_arc" in out else None,
                        "table": sa[["pr_auc", "pr_auc_lo", "pr_auc_hi", "d_pr_auc", "d_pr_auc_lo", "d_pr_auc_hi",
                                     "recall_at_fpr", "cost_per_1k", "cost_per_1k_lo", "cost_per_1k_hi", "brier"]
                                    ].round(4).to_dict(orient="index")}
        print(f"done {tag} in {time.time()-t0:.0f}s", flush=True)
# the main paper table: Mode A, stratified split
(res / "ablation_table.tex").write_text((res / "tables" / "ablation_pha_stratified_all.tex").read_text())
json.dump(summary, open(res / "stage5_summary.json", "w"), indent=1, default=str)
print("wrote results/ablation_table.tex")
