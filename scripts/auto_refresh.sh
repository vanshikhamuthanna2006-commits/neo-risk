#!/usr/bin/env bash
# Wait for the covariance download to finish (retrying gaps), then refresh Stages 3-5.
cd "$(dirname "$0")/.." || exit 1
LOG=results/auto_refresh.log
prefix() { python3 -c "
import sys,json; sys.path.insert(0,'.')
from pathlib import Path; from src.data_pipeline import cached_prefix
o=json.load(open('data/processed/pha_default/cov_fetch_order.json'))['order']
print(len(cached_prefix(o, Path('data/raw'))), len(o))"; }
echo "$(date -u +%FT%TZ) watcher started" >> "$LOG"
while pgrep -f fetch_covariances.py >/dev/null; do sleep 30; done
for pass in 1 2 3; do
  read -r got total <<< "$(prefix)"
  echo "$(date -u +%FT%TZ) prefix $got / $total" >> "$LOG"
  [ "$got" -ge "$total" ] && break
  echo "$(date -u +%FT%TZ) retry pass $pass" >> "$LOG"
  python3 scripts/fetch_covariances.py >> results/stage3_cov_fetch.log 2>&1
done
read -r got total <<< "$(prefix)"
echo "$(date -u +%FT%TZ) final prefix $got / $total; starting refresh" >> "$LOG"
mkdir -p "results/archive_$(date -u +%Y%m%d)_pre_full"
cp -r results/tables results/stage5_summary.json results/stage3_report.json results/ablation_table.tex "results/archive_$(date -u +%Y%m%d)_pre_full/" 2>/dev/null
python3 scripts/run_uncertainty.py > results/stage3_run_refresh.log 2>&1 \
  && rm -rf results/preds \
  && python3 scripts/run_evaluation.py 5 500 > results/stage5_run.log 2>&1 \
  && python3 scripts/regen_figures.py >> results/stage5_run.log 2>&1
echo "$(date -u +%FT%TZ) REFRESH EXIT $?" >> "$LOG"
