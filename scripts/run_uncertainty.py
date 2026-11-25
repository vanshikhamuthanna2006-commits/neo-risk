"""Stage 3 production: clone-dispersion features on the uniform random covariance subset +
physical priors on all objects; leakage gate, probe and selection-independence check."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_pipeline import assert_no_leakage, cached_prefix, leakage_probe, load_cached_covariances
from src.dynamics import DynamicsConfig
from src.uncertainty import (UncertaintyConfig, extract_uncertainty_features, physical_prior_features,
                             uncertainty_feature_columns)

root = Path(__file__).resolve().parents[1]
proc = root / "data" / "processed" / "pha_default"
X = pd.read_parquet(proc / "X.parquet"); audit = pd.read_parquet(proc / "audit_df.parquet")
orbit = pd.read_parquet(proc / "orbit_df.parquet"); y = pd.read_parquet(proc / "y.parquet")["y"]
order = json.load(open(proc / "cov_fetch_order.json"))["order"]
limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
sel = cached_prefix(order, root / "data" / "raw")[:limit]
print(f"uniform random covariance subset: {len(sel)} objects", flush=True)
cov = load_cached_covariances(sel, root / "data" / "raw")
cfg = UncertaintyConfig(dyn=DynamicsConfig(horizons=(50.0,), cache_dir=root / "data" / "unc_cache", n_jobs=-1))
t = time.time()
unc = extract_uncertainty_features(cov, X.index, cfg, orbit_df=orbit)
wall = time.time() - t
phys = physical_prior_features(X)
out = unc.join(phys)
out["diag_unc_selected"] = X.index.isin(sel).astype(float)
out.to_parquet(proc / "unc_df.parquet")
cols = uncertainty_feature_columns(out)
s = out["diag_unc_selected"] == 1
rep = {"n_selected": int(s.sum()), "wall_s": round(wall, 1),
       "has_cov_rate_in_selected": float(cov.has_cov.mean()) if len(cov) else None,
       "has_unc_clones_in_selected": float(out.loc[s, "has_unc_clones"].mean()),
       "pha_rate_selected": float(y[s].mean()), "pha_rate_not_selected": float(y[~s].mean()),
       "frac_unbound_mean": float(out.loc[s, "unc_frac_unbound"].mean()),
       "frac_lost_mean": float(out.loc[s, "unc_frac_lost"].mean()),
       "p_dmin_lt_005_mean": float(out.loc[s, "unc_p_dmin_lt_005"].mean())}
# selection independence: two-proportion z-test on PHA rate
p1, p0, n1, n0 = rep["pha_rate_selected"], rep["pha_rate_not_selected"], s.sum(), (~s).sum()
pp = (p1 * n1 + p0 * n0) / (n1 + n0)
rep["selection_z"] = float((p1 - p0) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n0)))
try:
    corr = assert_no_leakage(out[cols], audit); rep["gate"] = "PASS"
    rep["max_rank_corr_vs_audit"] = max(corr.values())
    rep["top_corr"] = dict(sorted(corr.items(), key=lambda kv: -kv[1])[:4])
except Exception as e:  # noqa: BLE001
    rep["gate"] = f"FAIL: {e}"
XS = X[s].join(out.loc[s, cols])
rep["probe_X_subset"] = leakage_probe(X[s], audit)
rep["probe_X_plus_unc_subset"] = leakage_probe(XS, audit)
print(json.dumps(rep, indent=1))
json.dump(rep, open(root / "results" / "stage3_report.json", "w"), indent=1)
