"""Stage 2 verification: ephemeris, conversions, integrator accuracy, real-object close approaches,
batching invariance, edge cases (old epochs, small q, missing sigma, sun-grazers, huge a)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import rebound

from src.dynamics import (EARTH_IDX, HILL_EARTH, MU_SUN, N_ACTIVE, YEAR, DynamicsConfig, _Recorder,
                          align_group, dynamical_feature_columns, extract_dynamical_features,
                          needs_ias15, perturb, propagate_batch)
from src.ephemeris import G_AU_DAY, J2000_JD, planet_elements
from src.orbital import cartesian_to_aei, elements_to_cartesian, periapsis_distance
from tests.conftest import FIX

T0 = 2461200.5


@pytest.fixture(scope="module")
def orbits() -> pd.DataFrame:
    return pd.read_json(FIX / "orbits_sample.json", dtype={"pdes": str}).set_index("pdes")


@pytest.fixture(scope="module")
def dyn(orbits) -> pd.DataFrame:
    return extract_dynamical_features(orbits, DynamicsConfig(n_jobs=4))


# ---------------- ephemeris & conversions ----------------
def test_emb_position_at_j2000():
    el = planet_elements("EMB", J2000_JD)
    r, _ = elements_to_cartesian(*[el[k] for k in ("a", "e", "inc", "Omega", "omega", "M")],
                                 mu=G_AU_DAY * (1 + 3.04e-6))
    # JPL Horizons, Earth-Moon barycentre, heliocentric ecliptic J2000 at 2000-01-01.5 TDB
    np.testing.assert_allclose(r[0], [-0.17713, 0.96724, 0.0], atol=2e-3)


def test_standish_validity_guard():
    with pytest.raises(ValueError):
        planet_elements("EMB", 2470000.5)


def test_elements_to_cartesian_matches_rebound():
    rng = np.random.default_rng(1)
    n = 200
    el = np.column_stack([rng.uniform(0.5, 6, n), np.r_[rng.uniform(0, 0.9, n - 2), 0.99, 0.9964],
                          rng.uniform(0, np.pi, n), *rng.uniform(0, 2 * np.pi, (3, n))])
    r, v = elements_to_cartesian(*el.T, mu=MU_SUN)
    sim = rebound.Simulation()
    sim.G = G_AU_DAY
    sim.add(m=1.0)
    for row in el:
        sim.add(a=row[0], e=row[1], inc=row[2], Omega=row[3], omega=row[4], M=row[5], primary=sim.particles[0])
    ref = np.array([[p.x, p.y, p.z, p.vx, p.vy, p.vz] for p in sim.particles[1:]])
    np.testing.assert_allclose(np.hstack([r, v]), ref, rtol=1e-9, atol=1e-12)
    a, e, i = cartesian_to_aei(r, v, MU_SUN)
    np.testing.assert_allclose(np.c_[a, e, i], el[:, :3], rtol=1e-9, atol=1e-11)


def test_periapsis_distance_hyperbola():
    mu, b, vinf = 1.0, 2.0, 1.0       # analytic: q = -mu/v^2 + sqrt(mu^2/v^4 + b^2)
    r = np.array([[-1e6, b, 0.0]])
    v = np.array([[1.0, 0.0, 0.0]]) * np.sqrt(vinf ** 2 + 2 * mu / np.linalg.norm(r))
    q_exp = -mu / vinf ** 2 + np.sqrt(mu ** 2 / vinf ** 4 + b ** 2)
    np.testing.assert_allclose(periapsis_distance(r, v, mu), q_exp, rtol=1e-5)


# ---------------- closest-approach refinement (synthetic) ----------------
def _straight_pass(b: float, speed: float = 0.01, dt: float = 1.0) -> np.ndarray:
    cfg = DynamicsConfig(div_sample_days=1e9)
    rec = _Recorder(1, cfg, np.array([False]), np.array([False]))
    earth = np.array([1.0, 0.0, 0.0])
    for k in range(-5, 6):
        t = (k + 0.37) * dt                         # closest approach falls mid-step
        xyz = np.zeros((N_ACTIVE + 2, 3))
        vel = np.zeros_like(xyz)
        xyz[EARTH_IDX] = earth
        xyz[N_ACTIVE] = xyz[N_ACTIVE + 1] = earth + [speed * t, b, 0.0]
        vel[N_ACTIVE] = vel[N_ACTIVE + 1] = [speed, 0.0, 0.0]
        rec.update(t + 10, xyz, vel)
    return rec


def test_dmin_between_steps_outside_hill():
    rec = _straight_pass(0.02)
    assert 0.02 > HILL_EARTH
    np.testing.assert_allclose(rec.dmin, 0.02, rtol=1e-9)    # sampled minimum would be ~0.0201
    assert rec.n_enc[0] == 1


def test_dmin_inside_hill_uses_focused_periapsis():
    rec = _straight_pass(0.002, speed=0.002)
    assert 0 < rec.dmin[0] < 0.002


# ---------------- perturbation edge cases ----------------
def test_perturbation_is_one_sigma_and_valid(orbits):
    nom, clone, s = perturb(orbits)
    sig = orbits[[f"sigma_{c}" for c in ("a", "e", "i", "om", "w", "ma")]].to_numpy(float).copy()
    sig[:, 2:] = np.deg2rad(sig[:, 2:])
    ok = s > 0
    maha = np.sqrt((((clone - nom)[ok] / sig[ok]) ** 2).sum(1))
    # (nom + delta) - nom loses ~1e-16/sigma_rel digits (sigma_a ~1e-10 au): 1e-5 is exact enough
    np.testing.assert_allclose(maha, s[ok], rtol=1e-5)
    assert (clone[:, 1] < 1).all() and (clone[:, 0] > 0).all()
    assert (s[orbits["sigma_a"].isna().to_numpy()] == 0).all()
    assert s[orbits.index.get_loc("2016 XK24")] == 1.0         # e=0.9964 nominal not collapsed


def test_huge_sigma_is_scaled_down():
    df = pd.DataFrame({"epoch": [T0], "a": [1.5], "e": [0.6], "i": [5.0], "om": [10.0], "w": [20.0],
                       "ma": [30.0], "sigma_a": [5.0], "sigma_e": [2.0], "sigma_i": [1.0],
                       "sigma_om": [1.0], "sigma_w": [1.0], "sigma_ma": [10.0]})
    _, clone, s = perturb(df)
    assert 0 < s[0] < 1 and clone[0, 1] < 0.99


# ---------------- integration quality ----------------
def test_energy_conserved(dyn):
    bulk = dyn[dyn.diag_integrator_ias15 == 0]
    assert (bulk.diag_energy_rel_err < 1e-8).all()
    assert (dyn.diag_energy_rel_err < 1e-6).all()


def test_small_q_routed_to_ias15(orbits, dyn):
    nom, clone, _ = perturb(orbits)
    flag = needs_ias15(nom, DynamicsConfig())
    assert flag.sum() >= 3
    assert (dyn.loc[orbits.index[flag], "diag_integrator_ias15"] == 1).all()


def test_mercurius_agrees_with_ias15(orbits):
    std = orbits[(orbits.epoch == T0) & ~needs_ias15(perturb(orbits)[0], DynamicsConfig())].iloc[:16]
    kw = dict(years=10.0, horizons=(10.0,), n_jobs=4)
    m = extract_dynamical_features(std, DynamicsConfig(**kw))
    i = extract_dynamical_features(std, DynamicsConfig(integrator="ias15", **kw))
    rel = (m.dyn_dmin_earth_au - i.dyn_dmin_earth_au).abs() / i.dyn_dmin_earth_au
    assert rel.median() < 1e-3, rel.describe()
    assert (m.dyn_da_10 - i.dyn_da_10).abs().median() < 1e-5


def test_batching_does_not_change_results(orbits):
    cfg = DynamicsConfig(years=5.0, horizons=(5.0,))
    sub = orbits[(orbits.epoch == T0) & ~needs_ias15(perturb(orbits)[0], cfg)].iloc[:8]
    nom, clone, s = perturb(sub)
    st, lost = align_group(T0, T0, nom, clone, cfg)
    full = propagate_batch(st, s > 0, lost, T0, "mercurius", cfg)
    one = propagate_batch(st[[3]], s[[3]] > 0, lost[[3]], T0, "mercurius", cfg)
    for k in ("dyn_dmin_earth_au", "dyn_da_5", "dyn_div_log10_growth"):
        np.testing.assert_allclose(full[k][3], one[k][0], rtol=1e-8)


# ---------------- real-object physical validation ----------------
def test_apophis_2029_flyby(dyn):
    r = dyn.loc["99942"]
    jd = T0 + r.dyn_tmin_earth_yr * YEAR
    assert abs(jd - 2462240.407) < 1.0              # 2029-04-13 21:46 UTC
    assert r.dyn_dmin_earth_au < 0.0008             # true geocentric: 0.000254 au (38,000 km)
    assert r.dyn_div_log10_growth > 3               # keyhole encounter amplifies uncertainty


def test_bennu_2060_and_eros_2056(dyn):
    b, e = dyn.loc["101955"], dyn.loc["433"]
    assert 0.003 < b.dyn_dmin_earth_au < 0.008 and 2060 <= 2026.44 + b.dyn_tmin_earth_yr < 2061
    assert 0.14 < e.dyn_dmin_earth_au < 0.16 and 2055.9 < 2026.44 + e.dyn_tmin_earth_yr < 2056.2
    assert abs(e.dyn_da_50) < 1e-3                   # Eros: no close encounters, tiny drift


def test_old_epoch_objects_aligned(orbits, dyn):
    old = orbits.index[orbits.epoch < 2455000]
    assert (dyn.loc[old, "diag_align_years"] > 15).all()
    assert dyn.loc[old, "dyn_dmin_earth_au"].notna().all()


# ---------------- output contract ----------------
def test_nan_policy_and_columns(dyn):
    cols = dynamical_feature_columns(dyn)
    assert not any(c.startswith("diag_") for c in cols)
    X = dyn[cols]
    assert not np.isinf(X.to_numpy(float)).any()
    div = ["dyn_div_log10_growth", "dyn_lyap_proxy_per_yr", "dyn_log10_div_max_au"]
    assert X[div].isna().any(axis=1).equals(dyn.has_dyn_div == 0)
    rest = [c for c in cols if c not in div]
    assert X.loc[dyn.dyn_lost == 0, rest].notna().all().all()
    assert {"dyn_da_10", "dyn_de_25", "dyn_di_50", "dyn_dmin_earth_au", "dyn_lyap_proxy_per_yr"} <= set(cols)


def test_sun_grazer_flagged_lost_without_crash():
    df = pd.DataFrame({"epoch": [T0, T0], "a": [1.0, 1.3], "e": [0.997, 0.2], "i": [10.0, 3.0],
                       "om": [0.0, 50.0], "w": [0.0, 60.0], "ma": [300.0, 10.0],
                       **{f"sigma_{c}": [1e-6, 1e-6] for c in ("a", "e", "i", "om", "w", "ma")}},
                      index=pd.Index(["grazer", "normal"], name="pdes"))
    d = extract_dynamical_features(df, DynamicsConfig(years=2.0, horizons=(1.0, 2.0), n_jobs=1))
    assert d.loc["grazer", "dyn_lost"] == 1 and d.loc["normal", "dyn_lost"] == 0
    assert np.isnan(d.loc["grazer", "dyn_da_2"]) and np.isfinite(d.loc["normal", "dyn_da_2"])


def test_configurable_horizon_and_cache(orbits, tmp_path):
    sub = orbits.iloc[:6]
    cfg = DynamicsConfig(years=3.0, horizons=(1.0, 3.0), cache_dir=tmp_path, n_jobs=1)
    a = extract_dynamical_features(sub, cfg)
    assert any(tmp_path.iterdir())
    b = extract_dynamical_features(sub, cfg)
    pd.testing.assert_frame_equal(a, b)
    assert {"dyn_da_1", "dyn_da_3"} <= set(a.columns)
    with pytest.raises(ValueError):
        extract_dynamical_features(sub, DynamicsConfig(years=3.0, horizons=(5.0,)))


def test_deterministic(orbits):
    sub = orbits.iloc[:10]
    cfg = DynamicsConfig(years=2.0, horizons=(1.0, 2.0), n_jobs=2)
    pd.testing.assert_frame_equal(extract_dynamical_features(sub, cfg),
                                  extract_dynamical_features(sub, cfg))


def test_large_alignment_group_regression():
    """Regression: holding ``primary=sim.particles[0]`` across sim.add() left a dangling ctypes
    pointer after REBOUND reallocated its particle array, silently corrupting ~10% of objects
    (all at non-standard epochs).  A large group must match one-at-a-time alignment exactly."""
    rng = np.random.default_rng(3)
    n = 150
    df = pd.DataFrame({"epoch": 2456771.5, "a": rng.uniform(1.0, 2.5, n), "e": rng.uniform(0.05, 0.6, n),
                       "i": rng.uniform(0, 30, n), "om": rng.uniform(0, 360, n), "w": rng.uniform(0, 360, n),
                       "ma": rng.uniform(0, 360, n),
                       **{f"sigma_{c}": 1e-4 for c in ("a", "e", "i", "om", "w", "ma")}})
    cfg = DynamicsConfig()
    nom, clone, _ = perturb(df)
    for _ in range(3):                                   # failure was non-deterministic
        st, lost = align_group(2456771.5, T0, nom, clone, cfg)
        assert not lost.any()
        assert np.abs(st[:, :, :3]).max() < 50
    for k in (0, 77, 149):
        st1, _ = align_group(2456771.5, T0, nom[[k]], clone[[k]], cfg)
        np.testing.assert_allclose(st[k], st1[0], rtol=1e-9, atol=1e-12)
