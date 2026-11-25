"""Export upcoming Earth close approaches (next 24 months, <= 0.05 au) from JPL's close-approach
database (cad.api) to docs/data/closeapp.json, linked to the atlas's object rows.

python scripts/export_closeapp.py
"""
from __future__ import annotations
import json, time
from datetime import date, timedelta
from pathlib import Path
import requests

root = Path(__file__).resolve().parents[1]
out = root / "docs" / "data"


def main() -> None:
    objs = json.load(open(out / "objects.json"))
    row = {d: i for i, d in enumerate(objs["pdes"])}
    start = date.today(); stop = start + timedelta(days=730)
    for k in range(6):
        try:
            r = requests.get("https://ssd-api.jpl.nasa.gov/cad.api", timeout=90, params={
                "date-min": start.isoformat(), "date-max": stop.isoformat(), "dist-max": "0.05",
                "sort": "date", "fullname": "true", "diameter": "true"})
            r.raise_for_status(); res = r.json(); break
        except requests.RequestException:
            time.sleep(2 ** k)
    f = res["fields"]
    items = []
    for d in res.get("data", []):
        x = dict(zip(f, d))
        items.append({"des": x["des"], "name": x["fullname"].strip(), "jd": float(x["jd"]), "cd": x["cd"],
                      "dist_au": float(x["dist"]), "dist_min_au": float(x["dist_min"]), "v_rel": float(x["v_rel"]),
                      "h": None if x["h"] is None else float(x["h"]),
                      "diameter_km": None if x.get("diameter") in (None, "") else float(x["diameter"]),
                      "row": row.get(x["des"], -1)})
    json.dump({"source": "JPL SBDB close-approach data API (cad.api)", "generated": start.isoformat(),
               "window_days": 730, "dist_max_au": 0.05, "items": items},
              open(out / "closeapp.json", "w"), separators=(",", ":"))
    n_in = sum(i["row"] >= 0 for i in items)
    print(f"closeapp.json: {len(items)} approaches ({n_in} objects in the atlas, {len(items) - n_in} not in it)")


if __name__ == "__main__":
    main()
