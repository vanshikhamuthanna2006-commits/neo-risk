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
# W1: explanations (the exact-sum property is asserted at export time against the fold models)
# reassemble the chunked explanations (docs/data/explain/) into whole arrays
xi = json.load(open(D / "explain" / "index.json")); x = {"labels": xi["labels"]}
for m, meta in xi["modes"].items():
    parts = [json.load(open(D / "explain" / f"{m}_{c}.json")) for c in range((xi["n"] + xi["chunk"] - 1) // xi["chunk"])]
    chk([q["start"] for q in parts] == list(range(0, xi["n"], xi["chunk"])), f"explain {m}: chunk starts")
    x[m] = {"names": meta["names"], "platt": meta["platt"], **{k: sum((q[k] for q in parts), []) for k in ("idx", "val", "fx", "base", "logit", "fold")}}
for m in ("pha", "sentry"):
    xm = x[m]; nf = len(xm["names"])
    idx = np.array(xm["idx"]).reshape(-1, 5); val = np.array(xm["val"]).reshape(-1, 5)
    chk(len(idx) == N and len(xm["base"]) == N and len(xm["logit"]) == N, f"explain {m}: length")
    chk((idx >= 0).all() and (idx < nf).all(), f"explain {m}: feature index out of range")
    chk((np.diff(np.abs(val), axis=1) <= 1e-9).all(), f"explain {m}: drivers not sorted by |contribution|")
    chk(all(n in x["labels"] for n in xm["names"]), f"explain {m}: missing labels")
    # the explained logit, through that object's fold calibration, must give the published probability
    ab = np.array(xm["platt"])[np.array(xm["fold"])]
    p_rec = 1 / (1 + np.exp(-(ab[:, 0] * np.array(xm["logit"]) + ab[:, 1])))
    err = np.abs(p_rec - np.array(o[f"p_{m}_full"], float)).max()
    chk(err < 1e-4, f"explain {m}: calibrated logit differs from p_full by {err:.2e}")
# round 2: uncertainty clouds + upcoming close approaches
C = np.fromfile(D / "clouds.bin", dtype=np.float64).reshape(-1, 8)
rowsC = C[:, 0].astype(int)
chk(((rowsC >= 0) & (rowsC < N)).all(), "clouds: row index out of range")
cen = C[C[:, 1] == -1]
chk(len(np.unique(cen[:, 0])) == len(np.unique(rowsC)) == len(cen), "clouds: each object needs exactly one centre row")
chk(np.allclose(cen[:, 2:8], E[cen[:, 0].astype(int), :6], atol=1e-5), "clouds: centre != orbits.bin elements")
chk((C[:, 3] < 1).all() and (C[:, 2] > 0).all(), "clouds: unbound clone")
counts = np.bincount(rowsC)[np.unique(rowsC)]
chk(counts.min() >= 20, f"clouds: an object has only {counts.min() - 1} clones")
ca = json.load(open(D / "closeapp.json"))["items"]
jd = [i["jd"] for i in ca]
chk(len(ca) > 50 and jd == sorted(jd), "closeapp: empty or not sorted by date")
chk(all(i["dist_au"] <= 0.05 + 1e-9 for i in ca), "closeapp: approach beyond 0.05 au")
chk(all(i["row"] == -1 or o["pdes"][i["row"]] == i["des"] for i in ca), "closeapp: row link mismatch")
print("check_data:", "OK" if not errs else "\n  ".join(["FAIL"] + errs))
sys.exit(1 if errs else 0)
