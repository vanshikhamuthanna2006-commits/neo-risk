"""Export data for the 3-D orbit viewer (docs/index.html).

python scripts/export_orbits.py

docs/data/orbits.bin   Float32, 10 values per object, same order as docs/data/objects.json:
                       a [au], e, i, Omega, omega, M0 [rad], epoch - J2000 [days],
                       p_sentry (full model, out-of-fold), p_pha (same), flags
                       (bit0 PHA, bit1 Sentry list, bit2 short arc).
docs/data/flybys.json  Featured close approaches: JPL SBDB close-approach record (cad.api) and
                       JPL Horizons geocentric ecliptic-J2000 vectors of the asteroid and the Moon
                       around closest approach, plus the geocentric Sun direction.

Rendering note: the browser propagates every asteroid with two-body Kepler motion from its
osculating elements (fine at solar-system scale).  Close approaches are NOT drawn from that
approximation; they replay JPL's own trajectory, which is accurate to kilometres.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, requests

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
proc = root / "data" / "processed" / "pha_default"
out = root / "docs" / "data"
J2000 = 2451545.0
AU_KM = 149597870.7

FEATURED = [  # (designation as in SBDB, Horizons command, display name, one-line story)
    ("99942", "'99942;'", "Apophis",
     "A 340 m asteroid that will pass closer than geostationary satellites, visible to the naked eye from Earth."),
    ("153814", "'153814;'", "2001 WN5",
     "A near-kilometre asteroid that slips inside the Moon's orbit."),
    ("137108", "'137108;'", "1999 AN10",
     "One of the largest known objects to pass this close this century, about 1 km across."),
    ("2024 YR4", "'DES=2024 YR4;'", "2024 YR4",
     "Briefly the highest-rated impact risk ever; impact on Earth has since been ruled out."),
    ("2012 UE34", "'DES=2012 UE34;'", "2012 UE34",
     "A small asteroid that grazes past at under a third of the lunar distance."),
    ("308635", "'308635;'", "2005 YU55",
     "A dark, roughly 400 m asteroid, radar-imaged in 2011, returning inside the Moon's orbit."),
]


def get(url: str, params: dict, tries: int = 6) -> dict:
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=90)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:  # proxy resets are transient
            if k == tries - 1:
                raise
            wait = 2 ** k + np.random.rand()
            print(f"  retry {k + 1} after {type(e).__name__}; sleeping {wait:.1f}s", flush=True)
            time.sleep(wait)


def horizons_vectors(command: str, center: str, start: str, stop: str, step: str) -> np.ndarray:
    """(n, 4) array: JD_TDB, x, y, z [km], ecliptic J2000."""
    res = get("https://ssd.jpl.nasa.gov/api/horizons.api", {
        "format": "json", "COMMAND": command, "EPHEM_TYPE": "VECTORS", "CENTER": center,
        "START_TIME": f"'{start}'", "STOP_TIME": f"'{stop}'", "STEP_SIZE": f"'{step}'",
        "VEC_TABLE": "1", "OUT_UNITS": "KM-S", "REF_PLANE": "ECLIPTIC", "REF_SYSTEM": "ICRF",
        "CSV_FORMAT": "YES", "VEC_LABELS": "NO", "OBJ_DATA": "NO", "MAKE_EPHEM": "YES"})["result"]
    if "$$SOE" not in res:
        raise RuntimeError(f"Horizons returned no ephemeris for {command}:\n{res[:800]}")
    body = res[res.index("$$SOE") + 5:res.index("$$EOE")].strip().splitlines()
    rows = [[float(f) for k, f in enumerate(l.split(",")) if k in (0, 2, 3, 4)] for l in body if l.strip()]
    return np.array(rows)


T_COMMON = 2461200.5          # 2026-06-?? : the Stage-2 propagation epoch


def aligned_elements(orbit: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Osculating elements of every object at one common recent epoch.

    SBDB elements come at each orbit solution's own epoch (25% are pre-2025, back to 1980).
    Two-body motion from an old epoch drifts by up to ~0.05 au, so each object is first
    propagated to T_COMMON with the project's N-body model (Sun + Venus, EMB, Mars, Jupiter,
    Saturn) and its heliocentric state converted back to osculating elements.  Cached.
    """
    from src.dynamics import DynamicsConfig, MU_SUN, to_rad_elements
    from src.orbital import cartesian_to_elements
    from src.uncertainty import align_many
    cache = proc / "orbits_aligned.parquet"
    if cache.exists():
        c = pd.read_parquet(cache).reindex(orbit.index)
        if c.notna().all().all():
            return c[["a", "e", "i", "om", "w", "ma"]].to_numpy(), c["epoch"].to_numpy()
    el0 = to_rad_elements(orbit)
    ep = orbit["epoch"].to_numpy(float)
    t = time.time()
    states, lost = align_many(ep, el0, T_COMMON, DynamicsConfig(n_jobs=-1))
    el = cartesian_to_elements(states[:, :3], states[:, 3:], MU_SUN)
    bad = lost | ~np.isfinite(el).all(axis=1) | (el[:, 1] >= 1) | (el[:, 0] <= 0)
    el[bad] = el0[bad]                                   # keep SBDB elements where alignment failed
    epoch = np.where(bad, ep, T_COMMON)
    print(f"aligned {len(el) - bad.sum()} objects to JD {T_COMMON} in {time.time() - t:.0f}s "
          f"({bad.sum()} kept at their own epoch)", flush=True)
    pd.DataFrame(np.column_stack([el, epoch]), index=orbit.index,
                 columns=["a", "e", "i", "om", "w", "ma", "epoch"]).to_parquet(cache)
    return el, epoch


def horizons_elements(command: str, start: str, stop: str, step: str) -> np.ndarray:
    """(n, 7): JD_TDB, a [au], e, i, Omega, omega, M [rad]; heliocentric ecliptic J2000 osculating."""
    res = get("https://ssd.jpl.nasa.gov/api/horizons.api", {
        "format": "json", "COMMAND": command, "EPHEM_TYPE": "ELEMENTS", "CENTER": "'500@10'",
        "START_TIME": f"'{start}'", "STOP_TIME": f"'{stop}'", "STEP_SIZE": f"'{step}'",
        "OUT_UNITS": "'AU-D'", "REF_PLANE": "ECLIPTIC", "REF_SYSTEM": "ICRF", "CSV_FORMAT": "YES",
        "OBJ_DATA": "NO", "MAKE_EPHEM": "YES"})["result"]
    if "$$SOE" not in res:
        raise RuntimeError(f"Horizons returned no elements for {command}:\n{res[:800]}")
    rows = []
    for l in res[res.index("$$SOE") + 5:res.index("$$EOE")].strip().splitlines():
        f = [x.strip() for x in l.split(",")]
        # JDTDB, Cal, EC, QR, IN, OM, W, Tp, N, MA, TA, A, AD, PR
        rows.append([float(f[0]), float(f[11]), float(f[2]), *np.deg2rad([float(f[4]), float(f[5]), float(f[6]), float(f[9])])])
    return np.array(rows)


def main() -> None:
    objs = json.load(open(out / "objects.json"))
    idx = pd.Index(objs["pdes"])
    orbit = pd.read_parquet(proc / "orbit_df.parquet").reindex(idx)

    # ------------------------------------------------------------ all objects: elements + risk
    el, epoch = aligned_elements(orbit)
    nan0 = lambda k: np.nan_to_num(np.array(objs[k], dtype=float), nan=0.0)
    flags = (np.array(objs["pha"]) | (np.array(objs["sentry"]) << 1) | (np.array(objs["short"]) << 2)).astype(float)
    arr = np.column_stack([el, epoch - J2000,
                           nan0("p_sentry_full"), nan0("p_pha_full"), flags]).astype(np.float32)
    bad = ~np.isfinite(arr).all(axis=1) | (arr[:, 0] <= 0) | (arr[:, 1] >= 1)
    arr[bad, 0] = 0.0                          # a = 0 -> the viewer skips the object
    arr.tofile(out / "orbits.bin")
    print(f"orbits.bin: {len(arr)} objects ({bad.sum()} skipped), {arr.nbytes / 1e6:.2f} MB")

    # ------------------------------------------------------------ featured close approaches
    flybys = []
    for des, cmd, label, story in FEATURED:
        cad = get("https://ssd-api.jpl.nasa.gov/cad.api", {"des": des, "date-min": "2026-11-01",
                                                           "date-max": "2080-01-01", "dist-max": "0.01"})
        f = cad["fields"]
        rec = min((dict(zip(f, r)) for r in cad["data"]), key=lambda r: float(r["dist"]))
        jd_ca = float(rec["jd"])
        t0, t1 = jd_ca - 3.0, jd_ca + 3.0
        span = lambda jd: f"JD{jd:.5f}"
        ast = horizons_vectors(cmd, "'500@399'", span(t0), span(t1), "10m")
        moon = horizons_vectors("'301'", "'500@399'", span(t0), span(t1), "1h")
        sun = horizons_vectors("'10'", "'500@399'", span(jd_ca), span(jd_ca + 0.01), "1")
        # osculating elements every 30 days: the viewer switches to the nearest set, so the
        # orbit change at the encounter (e.g. Apophis 2029) is shown correctly
        hel = horizons_elements(cmd, "2026-06-01", "2080-01-01", "30d")
        d = np.linalg.norm(ast[:, 1:], axis=1)
        k = int(np.argmin(d))
        H = float(rec["h"])
        i = int(idx.get_loc(des))
        flybys.append({
            "des": des, "name": label, "story": story, "row": i,
            "jd_ca": jd_ca, "cd": rec["cd"], "dist_au": float(rec["dist"]), "dist_km": float(rec["dist"]) * AU_KM,
            "v_rel_kms": float(rec["v_rel"]), "H": H,
            # measured diameter when SBDB has one; otherwise the H-based range for albedo 0.25-0.05
            "diam_m": ([round(objs["diam"][i] * 1000)] * 2 if objs["diam"][i]
                       else [round(1329e3 / np.sqrt(a) * 10 ** (-H / 5)) for a in (0.25, 0.05)]),
            "diam_measured": bool(objs["diam"][i]),
            "p_sentry": objs["p_sentry_full"][i], "p_pha": objs["p_pha_full"][i],
            "t0": t0, "dt_min": 10.0,
            "ast_km": np.round(ast[:, 1:], 1).ravel().tolist(),
            "moon_t0": float(moon[0, 0]), "moon_dt_h": 1.0, "moon_km": np.round(moon[:, 1:], 0).ravel().tolist(),
            "sun_dir": (sun[0, 1:] / np.linalg.norm(sun[0, 1:])).round(6).tolist(),
            "horizons_min_km": float(d[k]),
            "el_t0": float(hel[0, 0]), "el_step": float(hel[1, 0] - hel[0, 0]),
            "el": np.round(hel[:, 1:], 9).ravel().tolist(),
        })
        print(f"{label:10s} {rec['cd']}  CAD {float(rec['dist']) * AU_KM:,.0f} km  "
              f"Horizons 10-min grid min {d[k]:,.0f} km  ({len(ast)} pts)", flush=True)
    json.dump({"generated": time.strftime("%Y-%m-%d"), "flybys": flybys},
              open(out / "flybys.json", "w"), separators=(",", ":"), allow_nan=False)
    print("flybys.json", f"{(out / 'flybys.json').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
