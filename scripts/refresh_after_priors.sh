#!/usr/bin/env bash
# Re-run evaluation with the cited priors (loop R1), then refresh the site's model data.
cd "$(dirname "$0")/.." || exit 1
LOG=results/refresh_priors.log
echo "$(date -u +%FT%TZ) start" > $LOG
mkdir -p results/archive_pre_cited_priors && cp -r results/tables results/stage5_summary.json results/ablation_table.tex results/archive_pre_cited_priors/
python3 - >> $LOG 2>&1 <<'PY'
import pandas as pd, sys
sys.path.insert(0, ".")
from src.uncertainty import physical_prior_features
p = "data/processed/pha_default/"
X = pd.read_parquet(p + "X.parquet"); u = pd.read_parquet(p + "unc_df.parquet")
ph = physical_prior_features(X); u[ph.columns] = ph; u.to_parquet(p + "unc_df.parquet"); print("unc_df physical priors updated")
PY
rm -rf results/preds
python3 scripts/run_evaluation.py 5 500 >> $LOG 2>&1 && python3 scripts/regen_figures.py >> $LOG 2>&1 && echo "$(date -u +%FT%TZ) EVAL DONE" >> $LOG
echo "$(date -u +%FT%TZ) REFRESH EXIT $?" >> $LOG
