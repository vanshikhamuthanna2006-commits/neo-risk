"""Build the full live dataset, save it, and run the information-leakage ablation."""
from __future__ import annotations
import json, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PipelineConfig, PhysicalPolicy
from src.data_pipeline import build_dataset, leakage_probe

logging.basicConfig(level=logging.INFO)
root = Path(__file__).resolve().parents[1]
cfg = PipelineConfig(cache_dir=root / "data" / "raw")
b = build_dataset(cfg)
b.save(root / "data" / "processed" / "pha_default")
print(json.dumps(b.provenance, indent=1, default=str))
print("X shape:", b.X.shape, "| PHA prevalence:", round(b.y.mean(), 4),
      "| short-arc (<30 d):", int(b.meta_df.short_arc.sum()))
g = b.feature_groups
abl = {"static_orbit": g["static_orbit"],
       "obs_meta": g["obs_meta"] + [m for m in g["masks"] if m.endswith("_missing")],
       "physical": g["physical"] + [m for m in g["masks"] if m.startswith("has_")],
       "orbit+obs": g["static_orbit"] + g["obs_meta"], "ALL": list(b.X.columns)}
res = {k: leakage_probe(b.X[v], b.audit_df) for k, v in abl.items()}
for k, v in res.items():
    print(f"  probe[{k:13s}] R2(H)={v['r2_H']:.3f}  R2(log MOID)={v['r2_moid']:.3f}")
(root / "results").mkdir(exist_ok=True)
json.dump(res, open(root / "results" / "stage1_leakage_probe.json", "w"), indent=1)
