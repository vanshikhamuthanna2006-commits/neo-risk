"""Prospective validation against the live JPL Sentry list (closes the "no future data" gap
once enough time has passed).

python scripts/prospective_check.py

The model was trained on the Sentry list as downloaded at Stage 1 (data/processed/.../audit_df).
Later, JPL adds objects (new risk found) and removes them (risk ruled out). This script
downloads the CURRENT list and asks two questions with scores the model produced BEFORE those
changes (the out-of-fold full-model scores in docs/data/objects.json):

  1. Additions: among objects NOT on the list at training time, do the ones added since score
     higher?  (ROC-AUC and the rank of each added object)
  2. Removals: among objects on the list at training time, do the ones removed since score
     lower?

Objects discovered after the snapshot are not in the model at all and are reported separately.
Results go to results/prospective_<date>.json.  With few changes (a few days after training),
the counts are too small to conclude anything, and the script says so.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, requests
from sklearn.metrics import roc_auc_score

root = Path(__file__).resolve().parents[1]
proc = root / "data" / "processed" / "pha_default"


def current_sentry() -> set[str]:
    for k in range(6):
        try:
            r = requests.get("https://ssd-api.jpl.nasa.gov/sentry.api", timeout=90)
            r.raise_for_status()
            return {d["des"] for d in r.json()["data"]}
        except requests.RequestException:
            time.sleep(2 ** k)
    raise RuntimeError("could not download the Sentry list")


def main() -> None:
    audit = pd.read_parquet(proc / "audit_df.parquet")
    prov = json.load(open(proc / "provenance.json"))
    objs = json.load(open(root / "docs" / "data" / "objects.json"))
    score = pd.Series(objs["p_sentry_full"], index=objs["pdes"], dtype=float)
    then = audit["in_sentry"].astype(bool)
    now_set = current_sentry()
    now = pd.Series(then.index.isin(list(now_set)), index=then.index)
    added, removed = then.index[~then & now], then.index[then & ~now]
    new_objects = sorted(now_set - set(then.index))
    out = {"checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "training_snapshot": prov.get("snapshot_utc", "unknown"),
           "n_added": int(len(added)), "n_removed": int(len(removed)), "n_new_discoveries_on_list": len(new_objects)}
    MIN = 10
    if len(added) >= MIN:
        y = (~then & now)[~then].astype(int)
        out["additions_roc_auc"] = round(float(roc_auc_score(y, score[y.index])), 4)
    if len(removed) >= MIN:
        y = (then & ~now)[then].astype(int)
        out["removals_roc_auc_lower_is_better"] = round(float(roc_auc_score(y, score[y.index])), 4)
    out["added"] = {d: round(float(score[d]), 4) for d in added}
    out["removed"] = {d: round(float(score[d]), 4) for d in removed}
    out["verdict"] = ("enough changes to evaluate" if max(len(added), len(removed)) >= MIN else
                      f"too few changes since training (need >= {MIN}); run again in a few months")
    f = root / "results" / f"prospective_{time.strftime('%Y%m%d')}.json"
    json.dump(out, open(f, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("added", "removed")}, indent=1)); print("wrote", f)


if __name__ == "__main__":
    sys.exit(main())
