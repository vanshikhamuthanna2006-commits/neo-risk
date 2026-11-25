"""Integrity checks for the static site's data files (docs/data). Exit 1 on any problem."""
import json, sys
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parents[2] / "docs" / "data"
o = json.load(open(D / "objects.json")); s = json.load(open(D / "summary.json")); f = json.load(open(D / "flybys.json"))
E = np.fromfile(D / "orbits.bin", dtype=np.float32).reshape(-1, 10)
N = len(o["pdes"]); errs = []
def chk(c, m):
    if not c: errs.append(m)
chk(all(len(v) == N for v in o.values()), "objects.json column lengths differ")
chk(len(E) == N, f"orbits.bin rows {len(E)} != {N}")
chk(np.isfinite(E).all(), "non-finite values in orbits.bin")
chk((E[:, 0] > 0).all() and (E[:, 1] >= 0).all() and (E[:, 1] < 1).all(), "invalid a/e in orbits.bin")
chk(np.allclose(E[:, 7], np.nan_to_num(np.array(o["p_sentry_full"], float)), atol=1e-5), "risk order mismatch")
fl = E[:, 9].astype(int)
chk(((fl & 1) == np.array(o["pha"])).all() and (((fl >> 1) & 1) == np.array(o["sentry"])).all(), "flag mismatch")
for k in ("p_pha_full", "p_sentry_full", "p_pha_static", "p_sentry_static", "pvh_pha", "pvb_pha", "pvh_sentry", "pvb_sentry"):
    v = np.array(o[k], float)
    chk(np.isnan(v).sum() == 0, f"{k}: {np.isnan(v).sum()} nulls"); chk(np.nanmin(v) >= 0 and np.nanmax(v) <= 1, f"{k} outside [0,1]")
for b in f["flybys"]:
    chk(o["pdes"][b["row"]] == b["des"], f"flyby {b['name']}: row mismatch")
    chk(abs(b["horizons_min_km"] - b["dist_km"]) / b["dist_km"] < 1e-3, f"flyby {b['name']}: Horizons vs CAD")
    chk(len(b["el"]) % 6 == 0 and len(b["el"]) > 600, f"flyby {b['name']}: element sets")
for m in ("pha", "sentry"):
    t = {r["model"]: r for r in s["tables"][f"{m}_stratified_all"]}
    oof, test = s[m]["metrics"]["full"]["pr_auc"], t["static_dyn_unc"]["pr_auc"]
    chk(abs(oof - test) < 0.03, f"{m}: OOF PR-AUC {oof} vs test {test} differ > 0.03")
print("check_data:", "OK" if not errs else "\n  ".join(["FAIL"] + errs))
sys.exit(1 if errs else 0)
