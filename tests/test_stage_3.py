"""Stage 3 verification: covariance sampling, cometary->Keplerian conversion, clone propagation,
re-centring, physical priors, leakage."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.config import COV_LABELS
from src.data_pipeline import assert_no_leakage, build_dataset, parse_covariance
from src.dynamics import DynamicsConfig
from src.uncertainty import (PhysicalConfig, UncertaintyConfig, cometary_to_keplerian, cov_matrix,
                             cov_nominal, draw_clones, extract_uncertainty_features, object_rng,
                             opik_encounter_velocity, physical_prior_features, sample_cometary,
                             uncertainty_feature_columns)
from tests.conftest import FIX

NAMES = {"433": "433", "2024_YR4": "2024 YR4", "1979_XB": "1979 XB", "nongrav": "101955", "99942": "99942"}


@pytest.fixture(scope="module")
def cov_df() -> pd.DataFrame:
    recs = {des: parse_covariance(json.loads((FIX / f"sbdb_cov_{f}.json").read_text()))
            for f, des in NAMES.items()}
    return pd.DataFrame.from_dict(recs, orient="index")


@pytest.fixture(scope="module")
def orbits() -> pd.DataFrame:
    return pd.read_json(FIX / "orbits_sample.json", dtype={"pdes": str}).set_index("pdes")


@pytest.fixture(scope="module")
def unc(cov_df, orbits) -> pd.DataFrame:
    idx = pd.Index(list(NAMES.values()) + ["NO_COV"], name="pdes")
    return extract_uncertainty_features(cov_df, idx, UncertaintyConfig(), orbit_df=orbits)


# ---------------- covariance parsing ----------------
def test_nongrav_covariance_marginalised(cov_df):
    raw = json.loads((FIX / "sbdb_cov_nongrav.json").read_text())["orbit"]["covariance"]
    full = np.array(raw["data"], float)
    assert full.shape == (8, 8)
    np.testing.assert_allclose(cov_matrix(cov_df.loc["101955"]), full[:6, :6], rtol=1e-12)
    assert cov_df.loc["101955", "cov_extra_params"] == "RHO,AMRAT"


# ---------------- sampling ----------------
@pytest.mark.parametrize("des", ["433", "1979 XB", "99942"])
def test_sampling_reproduces_mean_and_covariance(cov_df, des):
    r = cov_df.loc[des]
    mu, C = cov_nominal(r), cov_matrix(r)
    x = sample_cometary(mu, C, 40_000, np.random.default_rng(0))
    np.testing.assert_allclose(x.mean(0), mu, rtol=0, atol=1e-12 * np.abs(mu).max() + 1e-14)  # antithetic
    d = np.sqrt(np.diag(C))
    emp = np.cov(((x - mu) / d).T)
    np.testing.assert_allclose(emp, C / np.outer(d, d), atol=0.03)            # correlation matrix


def test_sampling_singular_covariance():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 2))
    C = A @ A.T * 1e-6                                          # rank 2
    x = sample_cometary(np.zeros(6), C, 1000, rng)
    assert np.isfinite(x).all()
    assert np.linalg.matrix_rank(x - x.mean(0), tol=1e-9) == 2


def test_seed_is_per_object_and_order_independent(cov_df):
    a, _ = draw_clones(cov_df, UncertaintyConfig())
    b, _ = draw_clones(cov_df.iloc[::-1], UncertaintyConfig())
    key = ["owner", "k"]
    pd.testing.assert_frame_equal(a.sort_values(key).reset_index(drop=True),
                                  b.sort_values(key).reset_index(drop=True))
    assert not np.allclose(object_rng("433", 1).standard_normal(3), object_rng("434", 1).standard_normal(3))


# ---------------- conversion ----------------
@pytest.mark.parametrize("fname", ["1979_XB", "nongrav"])
def test_cometary_to_keplerian_matches_jpl_elements(fname):
    """At the covariance epoch, (e, q, tp, ...) -> (a, M) must equal JPL's separately reported a, ma."""
    p = json.loads((FIX / f"sbdb_cov_{fname}.json").read_text())
    orb = p["orbit"]
    assert float(orb["covariance"]["epoch"]) == float(orb["epoch"])
    rec = parse_covariance(p)
    nom = np.array([[rec[f"cov_nom_{lab}"] for lab in COV_LABELS]])
    el, ok = cometary_to_keplerian(nom, rec["cov_epoch"])
    jpl = {e["name"]: float(e["value"]) for e in orb["elements"]}
    assert ok[0]
    np.testing.assert_allclose(el[0, 0], jpl["a"], rtol=1e-9)
    dM = (np.rad2deg(el[0, 5]) - jpl["ma"] + 180) % 360 - 180
    assert abs(dM) < 1e-5, dM


def test_unbound_clones_counted_not_resampled():
    x = np.array([[0.5, 1.0, 2460000.0, 10, 20, 5], [1.2, 1.0, 2460000.0, 10, 20, 5],
                  [0.5, -0.1, 2460000.0, 10, 20, 5], [0.5, 1.0, 2460000.0, 10, 20, 190]])
    _, ok = cometary_to_keplerian(x, 2460000.0)
    assert ok.tolist() == [True, False, False, False]


# ---------------- propagated features ----------------
def test_dispersion_orders_with_orbit_quality(unc):
    eros, xb, bennu = unc.loc["433"], unc.loc["1979 XB"], unc.loc["101955"]
    assert eros.unc_dmin_std_au < 1e-6 and bennu.unc_dmin_std_au < 1e-6
    assert xb.unc_dmin_std_au > 1e4 * eros.unc_dmin_std_au          # 4-day arc, lost object
    assert xb.unc_log10_lov_sigma > eros.unc_log10_lov_sigma + 4
    assert 0 < xb.unc_p_dmin_lt_005 < 1 and xb.unc_tmin_std_yr > 1
    assert eros.unc_p_dmin_lt_005 == 0


def test_recentring_matches_stage2_nominal(unc):
    # Stage-2 nominal d_min values (tests/test_stage_2 + production run)
    np.testing.assert_allclose(unc.loc["433", "unc_dmin_mean_au"], 0.149819, rtol=1e-4)
    np.testing.assert_allclose(unc.loc["99942", "unc_dmin_mean_au"], 0.000358, rtol=5e-3)
    np.testing.assert_allclose(unc.loc["2024 YR4", "unc_dmin_mean_au"], 0.002176, rtol=5e-3)


def test_yr4_2032_encounter_certain(unc):
    assert unc.loc["2024 YR4", "unc_p_dmin_lt_001"] == 1.0


def test_missing_covariance_is_masked(unc):
    feats = [c for c in uncertainty_feature_columns(unc) if c.startswith("unc_")]
    assert unc.loc["NO_COV", "has_unc_clones"] == 0 and unc.loc["NO_COV", feats].isna().all()
    have = unc[unc.has_unc_clones == 1]
    assert len(have) == 5 and have[feats].notna().all().all()
    assert (have.unc_n_valid_clones == 32).all()


def test_batching_invariance(cov_df, orbits):
    one = extract_uncertainty_features(cov_df.loc[["433"]], pd.Index(["433"]),
                                       UncertaintyConfig(dyn=DynamicsConfig(years=5.0, horizons=(5.0,))),
                                       orbit_df=orbits)
    many = extract_uncertainty_features(cov_df, cov_df.index,
                                        UncertaintyConfig(dyn=DynamicsConfig(years=5.0, horizons=(5.0,))),
                                        orbit_df=orbits)
    cols = ["unc_dmin_mean_au", "unc_dmin_std_au", "unc_da5_std"]
    np.testing.assert_allclose(one.loc["433", cols].to_numpy(float), many.loc["433", cols].to_numpy(float),
                               rtol=1e-8)


# ---------------- physical priors ----------------
def test_opik_velocity():
    np.testing.assert_allclose(opik_encounter_velocity(np.array([1.0]), np.array([0.0]), np.array([0.0])), 0.0,
                               atol=1e-12)
    # perpendicular-ish fast orbit
    assert opik_encounter_velocity(np.array([2.5]), np.array([0.7]), np.array([30.0]))[0] > 0.4


def test_opik_agrees_with_sentry_vinf(tmp_path, getter):
    b = build_dataset(__import__("src.config", fromlist=["PipelineConfig"]).PipelineConfig(
        cache_dir=tmp_path), get_json=getter)
    ph = physical_prior_features(b.X)
    s = b.audit_df["v_inf"].dropna()
    common = s.index.intersection(ph.index)
    rho = ph.loc[common, "phys_v_inf_kms"].corr(s.loc[common], method="spearman")
    rel = ((ph.loc[common, "phys_v_inf_kms"] - s.loc[common]).abs() / s.loc[common]).median()
    assert len(common) > 100 and rho > 0.8 and rel < 0.3, (rho, rel)
    assert_no_leakage(ph, b.audit_df)                     # Opik v_inf is not a copy of Sentry's


def test_kinetic_energy_known_case():
    X = pd.DataFrame({"a": [2.0], "e": [0.6], "i_deg": [20.0], "tax_C": [0], "tax_S": [1], "tax_X": [0],
                      "tax_other": [0], "has_taxonomy": [1], "log10_diameter_km": [0.0]})
    ph = physical_prior_features(X, PhysicalConfig(diam_frac_sigma=1e-6, n_samples=20000))
    v = ph.phys_v_imp_kms.iat[0] * 1e3
    rho = 2.7e3                                            # S-type prior mean, kg/m^3
    ke_mt = 0.5 * (np.pi / 6 * 1e9 * rho) * v ** 2 / 4.184e15
    np.testing.assert_allclose(ph.phys_log10_ke_mt_mean.iat[0], np.log10(ke_mt), atol=0.05)
    assert 0.05 < ph.phys_log10_ke_mt_sd.iat[0] < 0.2       # density prior spread only


def test_unknown_taxonomy_uses_mixture_and_no_diameter_no_mass():
    X = pd.DataFrame({"a": [1.5, 1.5], "e": [0.3, 0.3], "i_deg": [5.0, 5.0], "tax_C": [1, 0], "tax_S": [0, 0],
                      "tax_X": [0, 0], "tax_other": [0, 0], "has_taxonomy": [1, 0],
                      "log10_diameter_km": [np.nan, np.nan]})
    ph = physical_prior_features(X)
    c, u = ph.iloc[0], ph.iloc[1]
    assert c.phys_albedo_prior_median < u.phys_albedo_prior_median < 0.25
    assert u.phys_albedo_prior_logsd > c.phys_albedo_prior_logsd           # mixture is broader
    assert ph.has_phys_mass.sum() == 0 and ph.phys_log10_mass_kg_mean.isna().all()
    assert ph.phys_log10_e_spec_jkg.notna().all()


def test_strict_policy_disables_mass():
    X = pd.DataFrame({"a": [1.5], "e": [0.3], "i_deg": [5.0], "tax_C": [0], "tax_S": [1], "tax_X": [0],
                      "tax_other": [0], "has_taxonomy": [1], "log10_diameter_km": [-0.5]})
    ph = physical_prior_features(X, PhysicalConfig(use_diameter=False))
    assert ph.has_phys_mass.iat[0] == 0


def test_no_h_used_anywhere():
    import inspect
    import src.uncertainty as U
    src = inspect.getsource(U)
    assert '"H"' not in src and "abs_mag" not in src
