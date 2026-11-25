"""Export covariance-clone "uncertainty clouds" for the 3-D atlas (docs/data/clouds.bin).

python scripts/export_clouds.py

For the riskiest objects (top 300 by each mode's out-of-fold score) plus the featured flyby
objects and a few well-known ones, the Stage-3 clones (same sampler, same seeds) are:
  1. propagated two-body from the covariance epoch to the common epoch T_COMMON (2026),
  2. turned into offsets from the covariance nominal at T_COMMON, and
  3. added to the object's N-body-aligned nominal state at T_COMMON (orbits_aligned.parquet),
     then converted back to osculating elements.
The viewer propagates each clone two-body and anchors the cloud to the displayed position,
so it always sits on the asteroid; spread grows along the orbit with time, as in Stage 3.

Layout: Float32, 8 values per row: object row, k (-1 = centre), a, e, i, Omega, omega, M at T_COMMON.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from src.data_pipeline import load_cached_covariances
from src.dynamics import MU_SUN
from src.orbital import cartesian_to_elements, elements_to_cartesian
from src.uncertainty import UncertaintyConfig, draw_clones

T_COMMON = 2461200.5
proc = root / "data" / "processed" / "pha_default"
out = root / "docs" / "data"


def two_body(el: np.ndarray, epoch: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    el = el.copy()
    el[:, 5] = el[:, 5] + np.sqrt(MU_SUN / el[:, 0] ** 3) * (t - epoch)
    return elements_to_cartesian(*el.T, mu=MU_SUN)


def main() -> None:
    objs = json.load(open(out / "objects.json"))
    fly = json.load(open(out / "flybys.json"))["flybys"]
    pdes = pd.Index(objs["pdes"])
    pick = set()
    for m in ("sentry", "pha"):
        p = np.array(objs[f"p_{m}_full"], float)
        pick |= set(pdes[np.argsort(-p)[:300]])
    pick |= {f["des"] for f in fly} | {"101955", "433", "99942", "2024 YR4"}
    pick = sorted(d for d in pick if d in pdes)
    cov = load_cached_covariances(pick, root / "data" / "raw")
    clones, _ = draw_clones(cov, UncertaintyConfig())
    al = pd.read_parquet(proc / "orbits_aligned.parquet")
    rows = []
    for d, g in clones.groupby("owner"):
        if (g.k == -1).sum() != 1 or d not in al.index or al.loc[d, "epoch"] != T_COMMON:
            continue
        el = g[["a", "e", "i", "Omega", "omega", "M"]].to_numpy(float).copy()
        # two-body to T_COMMON: only the mean anomaly moves
        el[:, 5] += np.sqrt(MU_SUN / el[:, 0] ** 3) * (T_COMMON - g["cov_epoch"].to_numpy(float))
        c = np.flatnonzero(g.k.to_numpy() == -1)[0]
        nom = al.loc[[d], ["a", "e", "i", "om", "w", "ma"]].to_numpy(float)
        # offsets applied in ELEMENT space (angles wrapped): stays valid even when a cloud has
        # spread far along the orbit, where Cartesian state offsets give unphysical orbits
        dl = el - el[c]
        dl[:, 3:] = np.angle(np.exp(1j * dl[:, 3:]))
        el2 = nom + dl
        el2[:, 3:] = np.mod(el2[:, 3:], 2 * np.pi)
        ok = np.isfinite(el2).all(axis=1) & (el2[:, 1] >= 0) & (el2[:, 1] < 1) & (el2[:, 0] > 0) & (el2[:, 2] >= 0) & (el2[:, 2] <= np.pi)
        k = g.k.to_numpy()
        el2[c] = nom[0]                                    # centre = the displayed nominal (exactly)
        i = pdes.get_loc(d)
        for j in np.flatnonzero(ok | (k == -1)):
            rows.append([i, k[j], *el2[j]])
    arr = np.array(rows, np.float32)
    arr.tofile(out / "clouds.bin")
    n_obj = len(np.unique(arr[:, 0]))
    print(f"clouds.bin: {n_obj} objects, {len(arr)} rows, {arr.nbytes / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
