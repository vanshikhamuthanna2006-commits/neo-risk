"""Capture a small, real, stratified snapshot of JPL data as deterministic test fixtures."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import SBDB_FIELDS, SBDB_QUERY_URL, SENTRY_URL, SBDB_OBJECT_URL
from src.data_pipeline import JPLClient

out = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
out.mkdir(parents=True, exist_ok=True)
c = JPLClient()
full = c.get_json(SBDB_QUERY_URL, {"fields": ",".join(SBDB_FIELDS), "sb-kind": "a",
                                   "sb-group": "neo", "full-prec": "1"})
sentry = c.get_json(SENTRY_URL, {})
f = full["fields"]; rows = full["data"]
ip, ia = f.index("pha"), f.index("data_arc")
sdes = {d["des"] for d in sentry["data"]}
rng = np.random.default_rng(42)
pha = [r for r in rows if r[ip] == "Y"]
short = [r for r in rows if r[ia] is not None and float(r[ia]) < 30]
sent = [r for r in rows if r[f.index("pdes")] in sdes]
other = rows
pick = lambda L, k: [L[j] for j in rng.choice(len(L), size=min(k, len(L)), replace=False)]
sub = {tuple(r[:2]): r for r in pick(pha, 150) + pick(short, 150) + pick(sent, 150) + pick(other, 350)}
sub = list(sub.values())
json.dump({"signature": full["signature"], "fields": f, "data": sub, "count": len(sub)},
          open(out / "sbdb_query_sample.json", "w"))
pdes = {r[f.index("pdes")] for r in sub}
sent_sub = [d for d in sentry["data"] if d["des"] in pdes] + sentry["data"][:20]
json.dump({"signature": sentry["signature"], "count": str(len(sent_sub)), "data": sent_sub},
          open(out / "sentry_sample.json", "w"))
for des in ["433", "2024 YR4", short[0][f.index("pdes")]]:
    p = c.get_json(SBDB_OBJECT_URL, {"sstr": des, "cov": "mat", "full-prec": "1"})
    json.dump(p, open(out / f"sbdb_cov_{des.replace(' ', '_')}.json", "w"))
print("full NEO rows:", len(rows), "| sentry:", len(sentry["data"]), "| fixture rows:", len(sub),
      "| short-arc in full:", len(short), "| PHA in full:", len(pha))
