"""Stage 1 verification: ingestion, leakage gate, targets, covariance parsing."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.config import (COV_LABELS, LABEL_DEFINING, SENTRY_OUTPUTS, PhysicalPolicy,
                        PipelineConfig, TargetMode)
from src.data_pipeline import (LeakageError, assert_no_leakage, build_dataset, leakage_probe,
                               parse_covariance, repair_covariance)
from tests.conftest import FIX


def _cfg(tmp_path, **kw) -> PipelineConfig:
    return PipelineConfig(cache_dir=tmp_path / "raw", sbdb_page_size=150, **kw)


@pytest.fixture
def bundle(tmp_path, getter):
    return build_dataset(_cfg(tmp_path), get_json=getter,
                         cov_designations=["433", "2024 YR4", "1979 XB", "NOT_AN_OBJECT"])


# ---------------- ingestion ----------------
def test_paging_retrieves_all_rows_once(bundle, getter):
    n_fixture = len(getter.sbdb["data"])
    sbdb_calls = [c for c in getter.calls if "sbdb_query" in c[0]]
    assert len(sbdb_calls) == int(np.ceil(n_fixture / 150))
    assert bundle.provenance["sbdb"]["rows"] == n_fixture
    assert bundle.X.index.is_unique
    assert len(bundle.X) == bundle.provenance["cleaning"]["rows_out"]


def test_frames_aligned(bundle):
    for df in (bundle.y, bundle.audit_df, bundle.orbit_df, bundle.meta_df):
        assert df.index.equals(bundle.X.index)


# ---------------- leakage gate ----------------
def test_no_forbidden_columns_in_X(bundle):
    forbidden = LABEL_DEFINING | SENTRY_OUTPUTS | {"albedo"}
    assert not forbidden & set(bundle.X.columns)
    assert {"moid", "moid_ld", "H"} <= set(bundle.audit_df.columns)


def test_gate_catches_renamed_copy_of_moid(bundle):
    X = bundle.X.copy()
    X["orbital_feature_17"] = bundle.audit_df["moid"] * 3.7 + 1.0
    with pytest.raises(LeakageError, match="near-copies"):
        assert_no_leakage(X, bundle.audit_df)


def test_gate_catches_monotone_transform_of_H(bundle):
    X = bundle.X.copy()
    X["brightness_proxy"] = np.exp(-0.4 * bundle.audit_df["H"])   # rank-identical to H
    with pytest.raises(LeakageError):
        assert_no_leakage(X, bundle.audit_df)


@pytest.mark.parametrize("name", ["moid", "H", "pha", "abs_mag", "MOID_au", "ps_cum", "albedo"])
def test_gate_catches_forbidden_names(bundle, name):
    X = bundle.X.copy()
    X[name] = 0.0
    with pytest.raises(LeakageError, match="Forbidden"):
        assert_no_leakage(X, bundle.audit_df)


def test_full_policy_albedo_would_trip_gate(tmp_path, getter):
    b = build_dataset(_cfg(tmp_path, physical_policy=PhysicalPolicy.FULL), get_json=getter)
    with pytest.raises(LeakageError):
        assert_no_leakage(b.X, b.audit_df)


def test_strict_policy_has_no_size_information(tmp_path, getter):
    b = build_dataset(_cfg(tmp_path, physical_policy=PhysicalPolicy.STRICT), get_json=getter)
    assert not any("diam" in c or "tax_" in c for c in b.X.columns)


def test_diameter_plus_albedo_reconstructs_H(bundle, getter):
    """Justifies PhysicalPolicy.NO_H_RECONSTRUCTION: D & p_v give H almost exactly."""
    from src.data_pipeline import clean_sbdb
    df, _ = clean_sbdb(pd.DataFrame(getter.sbdb["data"], columns=getter.sbdb["fields"]))
    m = df["diameter"].gt(0) & df["albedo"].gt(0) & df["H"].notna()
    assert m.sum() >= 20
    h_hat = 5 * np.log10(1329.0 / (df.loc[m, "diameter"] * np.sqrt(df.loc[m, "albedo"])))
    resid = (h_hat - df.loc[m, "H"]).abs()
    assert np.median(resid) < 0.3, np.median(resid)


# ---------------- NaN / physical plausibility ----------------
def test_static_orbit_block_nan_free(bundle):
    cols = bundle.feature_groups["static_orbit"]
    assert np.isfinite(bundle.X[cols].to_numpy()).all()


def test_nans_only_in_sparse_columns_with_masks(bundle):
    X, g = bundle.X, bundle.feature_groups
    nan_cols = [c for c in X.columns if X[c].isna().any()]
    for c in nan_cols:
        assert c not in g["static_orbit"] and c not in g["masks"]
        if c in g["obs_meta"]:
            assert f"{c}_missing" in X.columns
            assert (X[f"{c}_missing"] == X[c].isna()).all()
    assert np.isfinite(X[g["masks"]].to_numpy()).all()
    assert not np.isinf(X.to_numpy(dtype=float)).any()


def test_physical_plausibility(bundle):
    X = bundle.X
    assert ((X["e"] >= 0) & (X["e"] < 1)).all()
    assert (X["q"] > 0).all() and (X["q"] <= 1.3 + 1e-6).all()          # NEO definition
    assert ((X["i_deg"] >= 0) & (X["i_deg"] <= 180)).all()
    for ang in ("om", "w", "ma"):
        np.testing.assert_allclose(X[f"sin_{ang}"] ** 2 + X[f"cos_{ang}"] ** 2, 1.0, atol=1e-12)
    np.testing.assert_allclose(X["q"], bundle.orbit_df["a"] * (1 - bundle.orbit_df["e"]))


# ---------------- targets ----------------
def test_mode_a_pha_target(bundle):
    y, audit = bundle.y, bundle.audit_df
    assert set(y.unique()) <= {0, 1}
    assert (y == audit["pha"]).all()
    agree = (audit["pha"] == audit["pha_recomputed"]).mean()
    assert agree > 0.99, agree          # JPL flag vs definition from rounded audit cols
    assert 0.05 < y.mean() < 0.5        # fixture is PHA-enriched; real prevalence ~5.9%


def test_mode_b_targets(tmp_path, getter):
    yb = build_dataset(_cfg(tmp_path, target_mode=TargetMode.SENTRY_BINARY), getter).y
    yip = build_dataset(_cfg(tmp_path, target_mode=TargetMode.LOG10_IP), getter).y
    yps = build_dataset(_cfg(tmp_path, target_mode=TargetMode.PS_CUM), getter).y
    assert yb.sum() >= 100
    assert ((yip > -12) == (yb == 1)).mean() > 0.99
    assert yip.max() < 0 and yip.min() == -12.0
    assert (yps[yb == 1] > -12).all() and yps.max() < 1.0
    assert not yip.isna().any() and not yps.isna().any()


# ---------------- covariance ----------------
def test_parse_covariance_fixtures(bundle):
    cov = bundle.cov_df
    assert cov.loc[["433", "2024 YR4", "1979 XB"], "has_cov"].all()
    assert not cov.loc["NOT_AN_OBJECT", "has_cov"]
    for des in ("433", "2024 YR4", "1979 XB"):
        r = cov.loc[des]
        M = np.zeros((6, 6))
        for a in range(6):
            for b in range(a, 6):
                M[a, b] = M[b, a] = r[f"cov_{COV_LABELS[a]}_{COV_LABELS[b]}"]
        assert np.all(np.linalg.eigvalsh(M / np.sqrt(np.outer(np.diag(M), np.diag(M)))) > 0)
        assert r["cov_asym"] < 1e-10


def test_cov_epoch_differs_from_orbit_epoch_is_preserved(bundle):
    raw = json.loads((FIX / "sbdb_cov_2024_YR4.json").read_text())
    assert float(raw["orbit"]["covariance"]["epoch"]) == bundle.cov_df.loc["2024 YR4", "cov_epoch"]
    assert float(raw["orbit"]["epoch"]) != bundle.cov_df.loc["2024 YR4", "cov_epoch"]


def test_repair_singular_and_asymmetric_covariance():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 3)) * np.array([1e-8, 1e-4, 1e2])     # rank 3, wild scales
    S = A @ A.T
    S[0, 1] += 1e-20                                              # tiny asymmetry
    R, d = repair_covariance(S)
    assert d["cov_repaired"] and d["cov_valid"]
    np.testing.assert_allclose(R, R.T)
    np.linalg.cholesky(R / np.sqrt(np.outer(np.diag(R), np.diag(R))))   # must not raise
    _, d_nan = repair_covariance(np.full((6, 6), np.nan))
    assert not d_nan["cov_valid"]


def test_parse_covariance_missing_block():
    assert parse_covariance({"orbit": {}}) == {"has_cov": False}


# ---------------- determinism & reporting ----------------
def test_deterministic(tmp_path, getter):
    h = [hashlib.sha256(pd.util.hash_pandas_object(
        build_dataset(_cfg(tmp_path), get_json=getter).X, index=True).values).hexdigest()
        for _ in range(2)]
    assert h[0] == h[1]


def test_leakage_probe_reports(bundle, capsys):
    r = leakage_probe(bundle.X, bundle.audit_df)
    assert set(r) == {"r2_H", "r2_moid"} and all(np.isfinite(list(r.values())))
    with capsys.disabled():
        print(f"\n[leakage probe on fixture, n={len(bundle.X)}] {r}")


def test_save_roundtrip(bundle, tmp_path):
    bundle.save(tmp_path / "out")
    X2 = pd.read_parquet(tmp_path / "out" / "X.parquet")
    pd.testing.assert_frame_equal(X2, bundle.X)


# ---------------- live (opt-in) ----------------
@pytest.mark.network
def test_live_jpl_small(tmp_path):
    b = build_dataset(PipelineConfig(cache_dir=tmp_path, max_objects=500))
    assert len(b.X) > 400 and b.provenance["sentry_rows"] > 1000
