"""Fetch full covariances for ALL NEOs in a seeded random order (resumable, polite).

Any fully-cached prefix of the order is a uniform random sample, so partial runs give a
label-independent subset (selecting by PHA/Sentry status would leak the label via the mask).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PipelineConfig
from src.data_pipeline import JPLClient, covariance_fetch_order, fetch_covariances, _cov_cache_path

root = Path(__file__).resolve().parents[1]
proc = root / "data" / "processed" / "pha_default"
order = covariance_fetch_order(pd.read_parquet(proc / "meta_df.parquet").index, seed=2026)
json.dump({"seed": 2026, "order": order}, open(proc / "cov_fetch_order.json", "w"))
cfg = PipelineConfig(cache_dir=root / "data" / "raw", cov_n_jobs=1)   # JPL throttles concurrency
client = JPLClient(timeout_s=60, max_retries=6)
todo = [d for d in order if not _cov_cache_path(cfg.cache_dir, d).exists()]
print(f"{len(order) - len(todo)} cached, {len(todo)} to fetch", flush=True)
t = time.time()
for i in range(0, len(todo), 500):
    fetch_covariances(todo[i:i + 500], client.get_json, cfg)
    print(f"{i + 500} done, {(i + 500) / (time.time() - t):.2f}/s", flush=True)
