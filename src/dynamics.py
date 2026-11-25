"""Stage 2: N-body propagation and dynamical feature extraction (REBOUND).

Pipeline
--------
1. Perturbation: each asteroid gets a 1-sigma clone (diagonal covariance, see ``perturb``).
2. Epoch alignment: objects whose osculating epoch differs from the common start ``t0`` are
   integrated (with planets) from their epoch to ``t0`` in per-epoch-group simulations.
3. Propagation over ``years`` from ``t0``:
   * MERCURIUS (WHFast + IAS15 during close encounters), fixed dt, many test particles per
     simulation: test particles do not interact, so batching does not change results.
   * IAS15 per object for orbits whose perihelion passage is under-resolved by dt
     (tau_p = q / v_p < peri_resolution_steps * dt).  This isolates adaptive-step collapse to
     the object that causes it.
4. Features per object (prefix ``dyn_``), all measured from the *integrated* heliocentric
   state at t0 so that frame/epoch conventions cancel.

Assumptions / limitations (documented for the paper)
---------------------------------------------------
* Planets: Sun + Venus, Earth-Moon barycentre (EMB), Mars, Jupiter, Saturn; Standish (1800-2050)
  initial conditions.  Mercury, Uranus, Neptune, GR, Yarkovsky and non-gravitational forces are
  omitted.  Distances to "Earth" are to the EMB (offset from geocentre <= 4,700 km).
* d_min uses step-bracketing: when the Earth-relative radial velocity changes sign inside
  Earth's Hill sphere the Earth-centred two-body periapsis is used (captures gravitational
  focusing); outside it, the straight-line closest approach.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import rebound

from .ephemeris import G_AU_DAY, MASS_MSUN, PLANETS, planet_elements
from .orbital import cartesian_to_aei, elements_to_cartesian, periapsis_distance

log = logging.getLogger(__name__)

YEAR = 365.25
N_ACTIVE = 1 + len(PLANETS)
EARTH_IDX = 1 + PLANETS.index("EMB")
MU_SUN = G_AU_DAY
MU_EARTH = G_AU_DAY * MASS_MSUN["EMB"]
HILL_EARTH = 1.0 * (MASS_MSUN["EMB"] / 3.0) ** (1.0 / 3.0)
EL_COLS = ("a", "e", "i", "om", "w", "ma")


@dataclass
class DynamicsConfig:
    years: float = 50.0
    horizons: tuple[float, ...] = (10.0, 25.0, 50.0)
    dt_days: float = 1.0
    integrator: str = "mercurius"            # bulk integrator: "mercurius" | "whfast" | "ias15"
    peri_resolution_steps: float = 3.0       # route to IAS15 if tau_p < this * dt
    ias15_min_dt: float = 1e-4               # days; prevents adaptive-step stall
    ias15_max_steps: int = 3_000_000
    batch_size: int = 256
    n_jobs: int = -1
    t0_jd: float | None = None               # default: modal epoch of the input
    r_lost_min: float = 0.005                # au (Sun-grazing)
    r_lost_max: float = 2000.0               # au (ejection)
    enc_threshold_au: float = 0.05
    div_sample_days: float = 10.0
    max_e_clone: float = 0.99
    cache_dir: Path | None = None

    def key(self) -> dict:
        d = asdict(self)
        d.pop("cache_dir"), d.pop("n_jobs"), d.pop("batch_size")
        return d


# --------------------------------------------------------------------------------------
# Perturbation (1-sigma clone)
# --------------------------------------------------------------------------------------
def to_rad_elements(df: pd.DataFrame) -> np.ndarray:
    """(n, 6) array a[au], e, i, Omega, omega, M [rad] from SBDB degree columns."""
    el = df[list(EL_COLS)].to_numpy(float).copy()
    el[:, 2:] = np.deg2rad(el[:, 2:])
    return el


def _valid(el: np.ndarray, max_e: np.ndarray | float) -> np.ndarray:
    return (el[:, 0] > 0) & (el[:, 1] >= 0) & (el[:, 1] < max_e) & (el[:, 2] >= 0) & (el[:, 2] <= np.pi)


def perturb(df: pd.DataFrame, max_e: float = 0.99) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nominal elements, clone elements and the applied scale s in [0, 1].

    delta = s * sigma * (1,1,1,1,1,1)/sqrt(6): a point on the s-sigma ellipsoid of the diagonal
    covariance (Mahalanobis norm = s).  s is halved until the clone is a valid ellipse;
    s = 0 (clone == nominal) when sigmas are missing.
    """
    nom = to_rad_elements(df)
    sig = df[[f"sigma_{c}" for c in EL_COLS]].to_numpy(float).copy()
    sig[:, 2:] = np.deg2rad(sig[:, 2:])
    has = np.all(np.isfinite(sig), axis=1)
    sig = np.where(np.isfinite(sig), sig, 0.0)
    s = np.where(has, 1.0, 0.0)
    delta = sig / np.sqrt(6.0)
    # per-object eccentricity cap: never below the nominal orbit's own eccentricity
    max_e = np.maximum(max_e, 0.5 * (1.0 + nom[:, 1]))
    for _ in range(40):
        clone = nom + s[:, None] * delta
        bad = ~_valid(clone, max_e)
        if not bad.any():
            break
        s = np.where(bad, s * 0.5, s)
    clone = nom + s[:, None] * delta
    return nom, clone, s


def needs_ias15(nom: np.ndarray, cfg: DynamicsConfig) -> np.ndarray:
    a, e = nom[:, 0], nom[:, 1]
    q = a * (1 - e)
    v_p = np.sqrt(MU_SUN * (1 + e) / q)
    return (q / v_p) < cfg.peri_resolution_steps * cfg.dt_days


# --------------------------------------------------------------------------------------
# Simulation construction
# --------------------------------------------------------------------------------------
def _add_helio(sim: rebound.Simulation, r: np.ndarray, v: np.ndarray, m: float = 0.0) -> None:
    """Add bodies by heliocentric Cartesian state (Sun must be particle 0 at the origin).

    NB: never pass ``primary=sim.particles[0]`` held across ``sim.add`` calls: it is a ctypes
    view into REBOUND's particle array, which is reallocated as particles are added, leaving a
    dangling pointer (silent, non-deterministic garbage states).  Converting elements ourselves
    avoids the pattern entirely.
    """
    for ri, vi in zip(np.atleast_2d(r), np.atleast_2d(v)):
        sim.add(m=m, x=ri[0], y=ri[1], z=ri[2], vx=vi[0], vy=vi[1], vz=vi[2])


def _massive_sim(jd: float) -> rebound.Simulation:
    sim = rebound.Simulation()
    sim.G = G_AU_DAY
    sim.add(m=1.0)
    for name in PLANETS:
        el = planet_elements(name, jd)
        m = MASS_MSUN[name]
        r, v = elements_to_cartesian(el["a"], el["e"], el["inc"], el["Omega"], el["omega"], el["M"],
                                     mu=G_AU_DAY * (1.0 + m))
        _add_helio(sim, r, v, m)
    sim.N_active = N_ACTIVE
    sim.testparticle_type = 0
    return sim


def _set_integrator(sim: rebound.Simulation, name: str, dt: float, cfg: DynamicsConfig) -> None:
    sim.integrator = name
    if name == "ias15":
        sim.integrator.min_dt = cfg.ias15_min_dt
        sim.dt = np.sign(dt) * min(abs(dt), 1.0)
    else:
        sim.dt = dt


def _read(sim: rebound.Simulation, xyz: np.ndarray, vel: np.ndarray) -> None:
    sim.serialize_particle_data(xyz=xyz, vxvyvz=vel)


# --------------------------------------------------------------------------------------
# Phase 1: epoch alignment
# --------------------------------------------------------------------------------------
def align_elements(epoch_jd: float, t0_jd: float, el: np.ndarray,
                   cfg: DynamicsConfig) -> tuple[np.ndarray, np.ndarray]:
    """Propagate independent test particles given by elements (m, 6) at ``epoch_jd`` to t0.

    Returns heliocentric states (m, 6) and a per-particle lost flag (m,).
    """
    m = len(el)
    r, v = elements_to_cartesian(*el.T, mu=MU_SUN)
    if abs(epoch_jd - t0_jd) < 1e-9:
        return np.hstack([r, v]), np.zeros(m, bool)
    needy = needs_ias15(el, cfg)
    if needy.any() and not needy.all() and cfg.integrator != "ias15":
        # Split so that one under-resolved orbit cannot force the whole group onto IAS15.
        st, lost = np.empty((m, 6)), np.zeros(m, bool)
        for mask in (~needy, needy):
            st[mask], lost[mask] = align_elements(epoch_jd, t0_jd, el[mask], cfg)
        return st, lost
    sim = _massive_sim(epoch_jd)
    _add_helio(sim, r, v)
    sim.move_to_com()
    span = t0_jd - epoch_jd
    integ = "ias15" if needy.any() else cfg.integrator
    _set_integrator(sim, integ, np.sign(span) * cfg.dt_days, cfg)
    sim.exact_finish_time = 1
    sim.integrate(span)
    xyz, vel = np.empty((sim.N, 3)), np.empty((sim.N, 3))
    _read(sim, xyz, vel)
    st = np.concatenate([xyz[N_ACTIVE:] - xyz[0], vel[N_ACTIVE:] - vel[0]], axis=1)
    rn = np.linalg.norm(st[:, :3], axis=1)
    lost = (rn < cfg.r_lost_min) | (rn > cfg.r_lost_max) | ~np.isfinite(rn)
    return st, lost


def align_group(epoch_jd: float, t0_jd: float, nom: np.ndarray, clone: np.ndarray,
                cfg: DynamicsConfig) -> tuple[np.ndarray, np.ndarray]:
    """Heliocentric states at t0, shape (n, 2, 6) [nominal, clone]; plus a lost flag (n,)."""
    n = len(nom)
    both = np.stack([nom, clone], axis=1).reshape(2 * n, 6)       # nom0, clone0, nom1, ...
    st, lost = align_elements(epoch_jd, t0_jd, both, cfg)
    return st.reshape(n, 2, 6), lost.reshape(n, 2)[:, 0]


# --------------------------------------------------------------------------------------
# Phase 2: propagation with on-the-fly feature recording
# --------------------------------------------------------------------------------------
class _Recorder:
    def __init__(self, n: int, cfg: DynamicsConfig, has_clone: np.ndarray, lost0: np.ndarray,
                 stride: int = 2):
        """stride=2: interleaved (nominal, clone) pairs; stride=1: independent particles."""
        self.cfg, self.n, self.stride = cfg, n, stride
        self.active = ~lost0.copy()
        self.div_ok = has_clone & self.active
        self.lost_t = np.where(lost0, 0.0, np.nan)
        self.dmin = np.full(n, np.inf)
        self.tmin = np.full(n, np.nan)
        self.n_enc = np.zeros(n, int)
        self.horizon_days = [h * YEAR for h in cfg.horizons]
        self.snap: dict[float, np.ndarray] = {}
        self.el0: np.ndarray | None = None
        self.div_t: list[float] = []
        self.div_ln: list[np.ndarray] = []
        self.next_div = 0.0
        self.prev: tuple | None = None

    def _helio(self, xyz, vel):
        r = xyz[N_ACTIVE:] - xyz[0]
        v = vel[N_ACTIVE:] - vel[0]
        if self.stride == 1:
            return r, v, r, v
        return r[0::2], v[0::2], r[1::2], v[1::2]

    def update(self, t: float, xyz: np.ndarray, vel: np.ndarray) -> None:
        cfg = self.cfg
        rn_, vn_, rc_, vc_ = self._helio(xyz, vel)
        # lost detection (nominal and clone)
        rnorm = np.linalg.norm(rn_, axis=1)
        newly = self.active & ((rnorm < cfg.r_lost_min) | (rnorm > cfg.r_lost_max) | ~np.isfinite(rnorm))
        if newly.any():
            self.active &= ~newly
            self.lost_t[newly] = t / YEAR
        rcn = np.linalg.norm(rc_, axis=1)
        self.div_ok &= self.active & (rcn > cfg.r_lost_min) & (rcn < cfg.r_lost_max) & np.isfinite(rcn)

        if self.el0 is None:
            self.el0 = np.stack(cartesian_to_aei(rn_, vn_, MU_SUN), axis=1)
        for hd in self.horizon_days:
            if hd not in self.snap and t >= hd - 1e-9:
                el = np.stack(cartesian_to_aei(rn_, vn_, MU_SUN), axis=1)
                el[~self.active] = np.nan
                self.snap[hd] = el

        # Earth-relative closest approach (step bracketing)
        r_rel = xyz[N_ACTIVE::self.stride] - xyz[EARTH_IDX]
        v_rel = vel[N_ACTIVE::self.stride] - vel[EARTH_IDX]
        d = np.linalg.norm(r_rel, axis=1)
        rv = np.einsum("ij,ij->i", r_rel, v_rel)
        cand = d.copy()
        if self.prev is not None:
            pr, pv, pd_, prv = self.prev
            passed = (prv < 0) & (rv >= 0)
            if passed.any():
                use_prev = pd_ < d
                rr = np.where(use_prev[:, None], pr, r_rel)
                vv = np.where(use_prev[:, None], pv, v_rel)
                dd = np.minimum(pd_, d)
                inside = dd < HILL_EARTH
                with np.errstate(invalid="ignore", divide="ignore"):
                    q2b = periapsis_distance(rr, vv, MU_EARTH)
                    v2 = np.einsum("ij,ij->i", vv, vv)
                    lin = np.sqrt(np.maximum(dd ** 2 - np.einsum("ij,ij->i", rr, vv) ** 2 / v2, 0.0))
                ref = np.where(inside, q2b, lin)
                ref = np.where(np.isfinite(ref), np.minimum(ref, dd), dd)
                cand = np.where(passed, np.minimum(ref, cand), cand)
                enc = passed & (ref < cfg.enc_threshold_au) & self.active
                self.n_enc += enc
        better = self.active & (cand < self.dmin)
        self.dmin = np.where(better, cand, self.dmin)
        self.tmin = np.where(better, t / YEAR, self.tmin)
        self.prev = (r_rel.copy(), v_rel.copy(), d, rv)

        # divergence sampling
        if t >= self.next_div - 1e-9:
            sep = np.linalg.norm(rc_ - rn_, axis=1)
            with np.errstate(divide="ignore"):
                ln = np.where(self.div_ok & (sep > 0), np.log(sep), np.nan)
            self.div_t.append(t / YEAR)
            self.div_ln.append(ln)
            self.next_div += cfg.div_sample_days

    def features(self) -> dict[str, np.ndarray]:
        cfg = self.cfg
        f: dict[str, np.ndarray] = {}
        el0 = self.el0
        for h, hd in zip(cfg.horizons, self.horizon_days):
            el = self.snap.get(hd, np.full_like(el0, np.nan))
            tag = f"{h:g}"
            f[f"dyn_da_{tag}"] = el[:, 0] - el0[:, 0]
            f[f"dyn_de_{tag}"] = el[:, 1] - el0[:, 1]
            f[f"dyn_di_{tag}"] = np.rad2deg(el[:, 2] - el0[:, 2])
        dmin = np.where(np.isfinite(self.dmin), self.dmin, np.nan)
        f["dyn_dmin_earth_au"] = dmin
        f["dyn_log10_dmin_earth"] = np.log10(dmin)
        f["dyn_tmin_earth_yr"] = self.tmin
        f["dyn_n_enc_005"] = self.n_enc.astype(float)
        T = np.asarray(self.div_t)
        L = np.vstack(self.div_ln) if self.div_ln else np.full((0, self.n), np.nan)
        ok = self.div_ok
        growth = (L[-1] - L[0]) / np.log(10) if len(T) > 1 else np.full(self.n, np.nan)
        half = T >= T[-1] / 2 if len(T) else np.zeros(0, bool)
        Th, Lh = T[half], L[half]
        tc = Th - Th.mean()
        slope = (tc[:, None] * (Lh - Lh.mean(axis=0))).sum(0) / (tc ** 2).sum() if len(Th) > 2 else np.full(self.n, np.nan)
        with np.errstate(invalid="ignore"):
            dmax = np.nanmax(np.where(np.isfinite(L), L, -np.inf), axis=0) / np.log(10) if len(T) else np.full(self.n, np.nan)
        f["dyn_div_log10_growth"] = np.where(ok, growth, np.nan)
        f["dyn_lyap_proxy_per_yr"] = np.where(ok, slope, np.nan)
        f["dyn_log10_div_max_au"] = np.where(ok, dmax, np.nan)
        f["dyn_lost"] = (~self.active).astype(float)
        f["diag_lost_time_yr"] = self.lost_t
        return f


def propagate_batch(states: np.ndarray, has_clone: np.ndarray, lost0: np.ndarray, t0_jd: float,
                    integrator: str, cfg: DynamicsConfig) -> dict[str, np.ndarray]:
    """Propagate heliocentric states at t0 for ``cfg.years`` and return features.

    ``states`` is (n, 2, 6) for nominal/clone pairs or (n, 6) for independent particles
    (divergence features are then NaN).
    """
    n = len(states)
    stride = 2 if states.ndim == 3 else 1
    sim = _massive_sim(t0_jd)
    st = states.copy()
    st[lost0] = np.array([1.0, 0, 0, 0, np.sqrt(MU_SUN), 0])      # inert placeholder orbit
    flat = st.reshape(stride * n, 6)
    _add_helio(sim, flat[:, :3], flat[:, 3:])
    sim.move_to_com()
    _set_integrator(sim, integrator, cfg.dt_days, cfg)
    e0 = sim.energy()
    rec = _Recorder(n, cfg, has_clone if stride == 2 else np.zeros(n, bool), lost0, stride)
    xyz, vel = np.empty((sim.N, 3)), np.empty((sim.N, 3))
    _read(sim, xyz, vel)
    rec.update(0.0, xyz, vel)
    T = cfg.years * YEAR
    n_steps = 0
    while sim.t < T - 1e-9:
        if integrator == "ias15":
            sim.dt = min(sim.dt, T - sim.t) if sim.dt > 0 else sim.dt
        sim.steps(1)
        n_steps += 1
        _read(sim, xyz, vel)
        rec.update(sim.t, xyz, vel)
        if n_steps > cfg.ias15_max_steps:
            log.warning("step budget exceeded; marking batch objects lost at t=%.2f yr", sim.t / YEAR)
            rec.active[:] = False
            break
    feats = rec.features()
    feats["diag_energy_rel_err"] = np.full(n, abs((sim.energy() - e0) / e0))
    feats["diag_n_steps"] = np.full(n, float(n_steps))
    feats["diag_integrator_ias15"] = np.full(n, float(integrator == "ias15"))
    return feats


# --------------------------------------------------------------------------------------
# Orchestration (parallel, cached)
# --------------------------------------------------------------------------------------
def _cache_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:24]


def _run_task(task: dict, cfg: DynamicsConfig) -> pd.DataFrame:
    fp = None
    if cfg.cache_dir is not None:
        key = _cache_key({"cfg": cfg.key(), "pdes": task["pdes"], "t0": task["t0"],
                          "integ": task["integ"],
                          "states": hashlib.sha256(task["states"].tobytes()).hexdigest()})
        fp = Path(cfg.cache_dir) / f"dyn_{key}.parquet"
        if fp.exists():
            return pd.read_parquet(fp)
    feats = propagate_batch(task["states"], task["has_clone"], task["lost0"], task["t0"],
                            task["integ"], cfg)
    df = pd.DataFrame(feats, index=pd.Index(task["pdes"], name="pdes"))
    if fp is not None:
        fp.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(fp)
    return df


def _align_chunk(groups: list[tuple[float, np.ndarray, np.ndarray]], t0: float,
                 cfg: DynamicsConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    return [align_group(ep, t0, nom, cl, cfg) for ep, nom, cl in groups]


def extract_dynamical_features(orbit_df: pd.DataFrame, cfg: DynamicsConfig | None = None,
                               ) -> pd.DataFrame:
    """Main entry point.  ``orbit_df`` must contain epoch, a, e, i, om, w, ma, sigma_* columns."""
    from joblib import Parallel, delayed

    cfg = cfg or DynamicsConfig()
    if max(cfg.horizons) > cfg.years + 1e-9:
        raise ValueError("horizons must not exceed integration length")
    df = orbit_df.copy()
    t0 = cfg.t0_jd if cfg.t0_jd is not None else float(df["epoch"].mode().iloc[0])
    nom, clone, scale = perturb(df, cfg.max_e_clone)
    has_clone = scale > 0
    pdes = df.index.to_numpy()

    # ---- phase 1: epoch alignment (grouped by epoch, chunked for parallelism)
    epochs = df["epoch"].to_numpy(float)
    order = np.argsort(epochs, kind="stable")
    uniq, starts = np.unique(epochs[order], return_index=True)
    bounds = list(starts) + [len(order)]
    groups = [(float(uniq[g]), order[bounds[g]:bounds[g + 1]]) for g in range(len(uniq))]
    n_jobs = cfg.n_jobs
    n_chunks = max(1, min(len(groups), 64))
    idx_chunks = [groups[i::n_chunks] for i in range(n_chunks)]
    chunks = [[(ep, nom[idx], clone[idx]) for ep, idx in ch] for ch in idx_chunks]
    res = Parallel(n_jobs=n_jobs)(delayed(_align_chunk)(c, t0, cfg) for c in chunks)
    states = np.empty((len(df), 2, 6))
    lost0 = np.zeros(len(df), bool)
    for gch, rch in zip(idx_chunks, res):
        for (ep, idx), (st, lo) in zip(gch, rch):
            states[idx], lost0[idx] = st, lo

    # ---- phase 2: propagation tasks
    ias = needs_ias15(nom, cfg) | needs_ias15(clone, cfg)
    if cfg.integrator == "ias15":
        ias[:] = True
    tasks: list[dict] = []
    bulk = np.flatnonzero(~ias)
    for i in range(0, len(bulk), cfg.batch_size):
        b = bulk[i:i + cfg.batch_size]
        tasks.append({"pdes": list(pdes[b]), "states": states[b], "has_clone": has_clone[b],
                      "lost0": lost0[b], "t0": t0, "integ": cfg.integrator})
    for k in np.flatnonzero(ias):
        tasks.append({"pdes": [pdes[k]], "states": states[[k]], "has_clone": has_clone[[k]],
                      "lost0": lost0[[k]], "t0": t0, "integ": "ias15"})
    tasks.sort(key=lambda t: -len(t["pdes"]))
    out = Parallel(n_jobs=n_jobs)(delayed(_run_task)(t, cfg) for t in tasks)
    dyn = pd.concat(out).reindex(df.index)
    dyn["dyn_perturb_scale"] = scale
    dyn["has_dyn_div"] = (has_clone & dyn["dyn_div_log10_growth"].notna().to_numpy()).astype(float)
    dyn["diag_t0_jd"] = t0
    dyn["diag_align_years"] = (t0 - epochs) / YEAR
    return dyn


def dynamical_feature_columns(dyn: pd.DataFrame) -> list[str]:
    """Columns usable as model features (excludes ``diag_*`` bookkeeping)."""
    return [c for c in dyn.columns if c.startswith(("dyn_", "has_dyn"))]
