"""Stage 4 verification: loss maths, LightGBM/torch models, calibration, Mondrian conformal coverage,
feature-set leakage gate, splits, and an end-to-end ablation run on real fixture data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig
from src.data_pipeline import LeakageError, build_dataset
from src.dynamics import DynamicsConfig, extract_dynamical_features
from src.model import (SET_BENIGN, SET_BOTH, SET_EMPTY, SET_HAZ, AsymFocalConfig, DualBranchMLP, ExperimentConfig,
                       LGBMConfig, LGBMModel, MLPConfig, MondrianConformal, PlattCalibrator, _sigmoid,
                       assemble_features, asym_focal_grad, asym_focal_grad_hess, asym_focal_loss,
                       asym_focal_loss_torch, bayes_threshold, make_splits, run_ablations)
from src.uncertainty import physical_prior_features


def _synthetic(n=6000, prev=0.06, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    logit = 2.0 * X.f0 - 1.5 * X.f1 + X.f2 * X.f3 + np.log(prev / (1 - prev)) - 1.0
    y = (rng.random(n) < _sigmoid(logit.to_numpy())).astype(float)
    return X, y


# ---------------- loss ----------------
@pytest.mark.parametrize("gamma,lfn,lfp", [(0.0, 1.0, 1.0), (2.0, 10.0, 1.0), (0.5, 3.0, 2.0)])
def test_focal_gradient_matches_finite_difference(gamma, lfn, lfp):
    cfg = AsymFocalConfig(gamma, lfn, lfp)
    z = np.linspace(-8, 8, 41)
    for yv in (0.0, 1.0):
        y = np.full_like(z, yv)
        fd = (asym_focal_loss(z + 1e-6, y, cfg) - asym_focal_loss(z - 1e-6, y, cfg)) / 2e-6
        np.testing.assert_allclose(asym_focal_grad(z, y, cfg), fd, rtol=1e-5, atol=1e-7)


def test_focal_reduces_to_logloss():
    cfg = AsymFocalConfig(0.0, 1.0, 1.0)
    z = np.linspace(-6, 6, 25)
    y = (np.arange(25) % 2).astype(float)
    p = _sigmoid(z)
    g, h = asym_focal_grad_hess(z, y, cfg)
    np.testing.assert_allclose(g, p - y, atol=1e-10)
    np.testing.assert_allclose(h, p * (1 - p), rtol=1e-6)
    np.testing.assert_allclose(asym_focal_loss(z, y, cfg), -(y * np.log(p) + (1 - y) * np.log(1 - p)), rtol=1e-10)


def test_focal_hessian_always_positive():
    _, h = asym_focal_grad_hess(np.linspace(-30, 30, 601), np.ones(601), AsymFocalConfig(4.0, 50.0, 1.0))
    assert (h > 0).all() and np.isfinite(h).all()


def test_torch_loss_matches_numpy():
    import torch
    cfg = AsymFocalConfig(2.0, 10.0, 1.0)
    z = np.linspace(-7, 7, 29)
    y = (np.arange(29) % 3 == 0).astype(float)
    zt = torch.tensor(z, requires_grad=True)
    lt = asym_focal_loss_torch(zt, torch.tensor(y), cfg)
    np.testing.assert_allclose(lt.detach().numpy(), asym_focal_loss(z, y, cfg), rtol=1e-9)
    lt.sum().backward()
    np.testing.assert_allclose(zt.grad.numpy(), asym_focal_grad(z, y, cfg), rtol=1e-6, atol=1e-9)


# ---------------- models ----------------
def test_lgbm_focal_lowers_false_negatives():
    X, y = _synthetic()
    tr, va, te = slice(0, 4000), slice(4000, 5000), slice(5000, None)
    base = LGBMModel(LGBMConfig(objective="binary")).fit(X[tr], y[tr], X[va], y[va])
    foc = LGBMModel(LGBMConfig(objective="asym_focal", focal=AsymFocalConfig(2.0, 20.0, 1.0))).fit(X[tr], y[tr], X[va], y[va])
    fn = lambda m: int(((m.predict_proba(X[te]) < 0.5) & (y[te] == 1)).sum())
    assert fn(foc) < fn(base)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y[te], foc.predict_logit(X[te])) > 0.85


def test_lgbm_logit_includes_init_score():
    X, y = _synthetic(n=3000)
    m = LGBMModel(LGBMConfig(objective="binary")).fit(X[:2000], y[:2000], X[2000:], y[2000:])
    assert abs(m.predict_proba(X).mean() - y.mean()) < 0.02      # unbiased level => init added back


def test_mlp_deterministic_uses_both_branches_and_handles_nan():
    X, y = _synthetic(n=3000)
    X.loc[::7, "f4"] = np.nan
    cfg = MLPConfig(max_epochs=8, patience=3)
    m1 = DualBranchMLP(["f0", "f1", "f4"], ["f2", "f3", "f5"], cfg).fit(X[:2000], y[:2000], X[2000:], y[2000:])
    m2 = DualBranchMLP(["f0", "f1", "f4"], ["f2", "f3", "f5"], cfg).fit(X[:2000], y[:2000], X[2000:], y[2000:])
    z = m1.predict_logit(X)
    np.testing.assert_array_equal(z, m2.predict_logit(X))
    assert np.isfinite(z).all()
    X2 = X.copy()
    X2[["f2", "f3"]] = 0.0
    assert np.abs(m1.predict_logit(X2) - z).max() > 1e-3          # branch 2 is used


def test_platt_improves_calibration():
    rng = np.random.default_rng(0)
    z = rng.normal(size=20000) * 2
    y = (rng.random(20000) < _sigmoid(0.4 * z - 1)).astype(float)   # raw scores over-confident & biased
    cal = PlattCalibrator().fit(z[:10000], y[:10000])
    brier = lambda p: np.mean((p - y[10000:]) ** 2)
    assert brier(cal(z[10000:])) < brier(_sigmoid(z[10000:]))
    np.testing.assert_allclose([cal.a, cal.b], [0.4, -1.0], atol=0.08)


# ---------------- conformal ----------------
def test_mondrian_coverage_per_class():
    rng = np.random.default_rng(1)
    n = 60000
    z = rng.normal(size=n) * 1.5 - 2.8
    y = (rng.random(n) < _sigmoid(z)).astype(float)
    p = _sigmoid(z + rng.normal(scale=0.5, size=n))                  # imperfect but exchangeable
    cal, te = slice(0, 30000), slice(30000, None)
    conf = MondrianConformal(alpha_benign=0.10, alpha_hazardous=0.01).fit(p[cal], y[cal])
    sets = conf.predict_sets(p[te])
    cov = MondrianConformal.covers(sets, y[te])
    assert cov[y[te] == 1].mean() >= 0.985                            # 1 - alpha_H (finite-sample slack)
    assert cov[y[te] == 0].mean() >= 0.885
    assert set(np.unique(sets)) <= {SET_EMPTY, SET_BENIGN, SET_HAZ, SET_BOTH}
    assert (sets == SET_BOTH).any()                                   # uncertain/follow-up set occurs


def test_mondrian_small_class_always_included():
    p = np.array([0.1] * 50 + [0.9] * 10)
    y = np.array([0] * 50 + [1] * 10, float)
    conf = MondrianConformal(alpha_hazardous=0.01).fit(p, y)        # ceil(11*0.99)=11 > 10
    assert conf.q[1] == np.inf
    assert ((conf.predict_sets(np.array([0.0, 0.5, 1.0])) & SET_HAZ) > 0).all()


def test_bayes_threshold():
    assert bayes_threshold(100.0, 1.0) == pytest.approx(1 / 101)


# ---------------- end-to-end on real fixture data ----------------
@pytest.fixture(scope="module")
def real(tmp_path_factory, ):
    from tests.conftest import FixtureGetter
    b = build_dataset(PipelineConfig(cache_dir=tmp_path_factory.mktemp("raw")), get_json=FixtureGetter())
    dyn = extract_dynamical_features(b.orbit_df, DynamicsConfig(years=3.0, horizons=(1.0, 3.0), n_jobs=4))
    unc = physical_prior_features(b.X)
    unc["unc_log10_sig_q"] = np.log10(b.orbit_df["sigma_a"])        # stand-in uncertainty column
    unc["has_unc_clones"] = 0.0
    frames = {fs: assemble_features(fs, b.X, b.feature_groups, b.audit_df, dyn, unc)
              for fs in ("naive", "static_orbit", "static", "static+dyn", "static+dyn+unc")}
    return b, frames


def test_feature_sets_gate(real):
    b, frames = real
    assert {"moid", "H"} <= set(frames["naive"].F.columns)
    for fs in ("static_orbit", "static", "static+dyn", "static+dyn+unc"):
        assert not {"moid", "H", "moid_ld", "albedo"} & set(frames[fs].F.columns)
    assert any(c.startswith("dyn_") for c in frames["static+dyn"].branch2)
    bad = b.X.copy()
    bad["proxy"] = b.audit_df["moid"] * 2
    with pytest.raises(LeakageError):
        assemble_features("static", bad, b.feature_groups, b.audit_df)


def test_splits_disjoint_stratified_temporal(real):
    b, _ = real
    s = make_splits(b.y, b.meta_df, "stratified", seed=0)
    allidx = np.concatenate(list(s.values()))
    assert len(allidx) == len(b.y) == len(np.unique(allidx))
    rates = {k: b.y.iloc[v].mean() for k, v in s.items()}
    assert max(rates.values()) - min(rates.values()) < 0.03
    assert make_splits(b.y, b.meta_df, "stratified", seed=0)["test"].tolist() == s["test"].tolist()
    t = make_splits(b.y, b.meta_df, "temporal", seed=0)
    fo = pd.to_datetime(b.meta_df["first_obs"], errors="coerce")
    assert fo.iloc[t["train"]].max() <= fo.iloc[t["test"]].min()


def test_end_to_end_ablations(real):
    b, frames = real
    splits = make_splits(b.y, b.meta_df, "stratified", seed=0)
    cfg = ExperimentConfig(mlp=MLPConfig(max_epochs=15, patience=4), lgbm=LGBMConfig(n_estimators=400))
    preds, arte = run_ablations(frames, b.y, splits, cfg)
    assert preds.model.nunique() == 8
    assert np.isfinite(preds.p).all() and preds.p.between(0, 1).all()
    from sklearn.metrics import average_precision_score
    te = preds[preds.split == "test"]
    ap = te.groupby("model").apply(lambda d: average_precision_score(d.y, d.p), include_groups=False)
    assert ap["naive_moid_h"] > 0.95                      # the circularity trap: near-perfect
    assert ap["naive_moid_h"] > ap["static_leakfree"]
    for name in arte:
        assert arte[name]["n_cal"][1] > 0


@pytest.mark.parametrize("lfn", [10.0, 20.0, 50.0])
def test_focal_boosting_is_stable(lfn):
    """Regression: the exact focal Hessian gave +20-logit first trees; leaf caps then stalled
    training entirely (one zero-output tree).  The surrogate Hessian must train normally."""
    X, y = _synthetic(n=4000, prev=0.03)
    m = LGBMModel(LGBMConfig(objective="asym_focal", focal=AsymFocalConfig(2.0, lfn, 1.0)))
    m.fit(X[:3000], y[:3000], X[3000:], y[3000:])
    d = m.booster.trees_to_dataframe()
    first = d[d.left_child.isna() & (d.tree_index == 0)].value.abs().max()
    assert 0 < first < 1.0
    assert m.booster.best_iteration > 20


def test_exact_hessian_blows_up_surrogate_does_not():
    z = np.full(1000, np.log(0.03 / 0.97))
    y = np.r_[np.ones(30), np.zeros(970)]
    cfg = AsymFocalConfig(2.0, 10.0, 1.0)
    step = lambda kind: -np.sum(asym_focal_grad_hess(z, y, cfg, kind)[0]) / np.sum(asym_focal_grad_hess(z, y, cfg, kind)[1])
    assert step("exact") > 10 * step("surrogate")


def test_recall_at_fpr():
    from src.model import recall_at_fpr
    y = np.array([0] * 100 + [1] * 10)
    s = np.r_[np.linspace(0, 1, 100), np.full(10, 2.0)]
    assert recall_at_fpr(y, s, 0.01) == 1.0
    assert recall_at_fpr(y, -s, 0.01) == 0.0
