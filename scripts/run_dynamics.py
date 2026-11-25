"""Stage 2 production run: 50-yr N-body features for the full NEO catalogue + leakage audit."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dynamics import DynamicsConfig, extract_dynamical_features, dynamical_feature_columns
from src.data_pipeline import assert_no_leakage, leakage_probe, LeakageError

root = Path(__file__).resolve().parents[1]
proc = root / "data" / "processed" / "pha_default"
orbit = pd.read_parquet(proc / "orbit_df.parquet")
audit = pd.read_parquet(proc / "audit_df.parquet")
X = pd.read_parquet(proc / "X.parquet")
years = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
cfg = DynamicsConfig(years=years, horizons=tuple(h for h in (10.0, 25.0, 50.0, 100.0) if h <= years),
                     cache_dir=root / "data" / "dyn_cache", n_jobs=-1)
t = time.time()
dyn = extract_dynamical_features(orbit, cfg)
wall = time.time() - t
dyn.to_parquet(proc / f"dyn_df_{int(years)}yr.parquet")
cols = dynamical_feature_columns(dyn)
rep = {
    "years": years, "n": len(dyn), "wall_s": round(wall, 1),
    "n_ias15": int(dyn.diag_integrator_ias15.sum()), "n_lost": int(dyn.dyn_lost.sum()),
    "n_no_div": int((dyn.has_dyn_div == 0).sum()),
    "max_energy_err_bulk": float(dyn.loc[dyn.diag_integrator_ias15 == 0, "diag_energy_rel_err"].max()),
    "max_energy_err_ias15": float(dyn.loc[dyn.diag_integrator_ias15 == 1, "diag_energy_rel_err"].max()),
    "n_dmin_lt_0.05": int((dyn.dyn_dmin_earth_au < 0.05).sum()),
    "spearman_dmin_vs_moid": float(dyn.dyn_dmin_earth_au.corr(audit.moid, method="spearman")),
    "frac_dmin_ge_moid": float((dyn.dyn_dmin_earth_au >= audit.moid * 0.98).mean()),
}
try:
    assert_no_leakage(dyn[cols], audit); rep["gate"] = "PASS"
except LeakageError as e:
    rep["gate"] = f"FAIL: {e}"
XD = X.join(dyn[cols])
rep["probe_X_only"] = leakage_probe(X, audit)
rep["probe_dyn_only"] = leakage_probe(dyn[cols], audit)
rep["probe_X_plus_dyn"] = leakage_probe(XD, audit)
print(json.dumps(rep, indent=1))
print(dyn[cols].describe().T[["mean", "50%", "min", "max"]].to_string())
json.dump(rep, open(root / "results" / f"stage2_report_{int(years)}yr.json", "w"), indent=1)
