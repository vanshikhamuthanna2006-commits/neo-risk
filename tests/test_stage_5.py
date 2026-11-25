"""Stage 5 verification: metric definitions, bootstrap properties, paired deltas, conformal
trade-off monotonicity, LaTeX well-formedness and figure generation."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (EvalConfig, conformal_tradeoff, evaluate_predictions, expected_cost_per_1k,
                          latex_ablation_table, latex_conformal_table, plot_conformal_tradeoff, plot_cost_curve,
                          plot_pr_curves, plot_reliability, point_metrics, recall_at_fpr,
                          stratified_bootstrap_indices, trivial_cost_per_1k)
from src.model import _sigmoid


def test_expected_cost_hand_example():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.5, 0.001, 0.5, 0.001, 0.001])          # threshold 1/101
    # one FN (cost 100) + one FP (cost 1) over 5 objects
    assert expected_cost_per_1k(y, p, 100, 1) == pytest.approx(1000 * 101 / 5)
    assert trivial_cost_per_1k(y, 100, 1) == pytest.approx(1000 * min(100 * 0.4, 0.6))


def test_recall_at_fpr_and_perfect_scores():
    y = np.r_[np.zeros(200), np.ones(20)]
    s = np.r_[np.linspace(0, 1, 200), np.full(20, 5.0)]
    assert recall_at_fpr(y, s, 0.01) == 1.0
    m = point_metrics(y, s, _sigmoid(s), EvalConfig())
    assert m["pr_auc"] == pytest.approx(1.0) and m["roc_auc"] == pytest.approx(1.0)


def test_stratified_bootstrap_preserves_class_counts():
    y = np.r_[np.zeros(90), np.ones(10)]
    idx = stratified_bootstrap_indices(y, 50, np.random.default_rng(0))
    assert idx.shape == (50, 100)
    assert (y[idx].sum(axis=1) == 10).all()


def _fake_preds(n_seeds=2, n=3000, prev=0.06):
    rows = []
    for s in range(n_seeds):
        rng = np.random.default_rng(s)
        y = (rng.random(n) < prev).astype(float)
        sig = y * 2.5 + rng.normal(size=n)
        split = rng.choice(["train", "val", "cal", "test"], n, p=[0.6, 0.1, 0.15, 0.15])
        for model, noise in (("static_leakfree", 1.0), ("proposed_ensemble", 0.6), ("clone_of_ref", 1.0)):
            z = sig + rng.normal(scale=noise, size=n) - 3 if model != "clone_of_ref" else None
            rows.append(pd.DataFrame({"seed": s, "model": model, "pdes": [f"o{i}" for i in range(n)], "split": split,
                                      "y": y, "logit": z if z is not None else np.nan, "p": np.nan, "conf_set": 0}))
    df = pd.concat(rows, ignore_index=True)
    ref = df[df.model == "static_leakfree"]
    df.loc[df.model == "clone_of_ref", "logit"] = ref.logit.to_numpy()
    df["p"] = _sigmoid(df.logit.to_numpy())
    return df


def test_evaluate_paired_deltas_and_ci():
    preds = _fake_preds(n=20000)
    meta = pd.DataFrame({"short_arc": [i % 2 for i in range(20000)]}, index=[f"o{i}" for i in range(20000)])
    out = evaluate_predictions(preds, meta, EvalConfig(n_boot=100))
    s = out["summary_all"]
    # identical model -> zero delta, degenerate CI
    assert s.loc["clone_of_ref", "d_pr_auc"] == pytest.approx(0.0)
    assert s.loc["clone_of_ref", "d_pr_auc_lo"] == pytest.approx(0.0) and s.loc["clone_of_ref", "d_pr_auc_hi"] == pytest.approx(0.0)
    # truly better model -> positive delta with CI excluding 0
    assert s.loc["proposed_ensemble", "d_pr_auc_lo"] > 0
    assert s.loc["proposed_ensemble", "d_pr_auc_p_not_better"] < 0.05
    for m in ("pr_auc", "roc_auc", "brier"):
        assert (s[f"{m}_lo"] <= s[m] + 1e-9).all() and (s[m] <= s[f"{m}_hi"] + 1e-9).all()
    assert "summary_short_arc" in out and out["counts_short_arc"].n.min() > 0
    assert set(out["conformal"].alpha_haz) == {0.01, 0.05, 0.10}


def test_conformal_tradeoff_monotone_and_covers_on_average():
    """Conformal coverage is guaranteed on average over calibration draws; a single draw with
    ~180 test positives has SD ~0.033, so we check the mean over 20 independent datasets."""
    covs = []
    for s in range(20):
        preds = _fake_preds(n_seeds=s + 1, n=20000)
        d = preds[(preds.model == "proposed_ensemble") & (preds.seed == s)]
        t = conformal_tradeoff(d, EvalConfig())
        assert (np.diff(t.follow_up_rate) <= 1e-12).all()          # larger alpha -> fewer follow-ups
        covs.append(t.haz_coverage.to_numpy())
    mean_cov = np.mean(covs, axis=0)
    assert (mean_cov >= 1 - np.array(EvalConfig().alphas_haz) - 0.01).all(), mean_cov


def test_latex_tables_well_formed():
    preds = _fake_preds()
    meta = pd.DataFrame({"short_arc": 0}, index=[f"o{i}" for i in range(3000)])
    out = evaluate_predictions(preds, meta, EvalConfig(n_boot=50))
    tex = latex_ablation_table(out["summary_all"], "Test caption", "tab:test", counts=out["counts_all"], trivial_cost=12.3)
    assert tex.count("{") == tex.count("}")
    for tok in (r"\toprule", r"\midrule", r"\bottomrule", r"\begin{tabular}", r"\end{table*}", r"\label{tab:test}"):
        assert tok in tex
    body = [l for l in tex.splitlines() if l.endswith(r"\\")]
    ncols = {l.count("&") for l in body}
    assert len(ncols) == 1                                    # every row has the same column count
    assert not re.search(r"(?<!\\)_(?![{A-Za-z]*\})", tex.replace("C_{FN}", "").replace("C_{FP}", "")) or "_" not in tex.replace("C_{FN}", "").replace("C_{FP}", "")
    ctex = latex_conformal_table(out["conformal"], ("proposed_ensemble",), "c", "tab:c")
    assert ctex.count("{") == ctex.count("}") and r"\bottomrule" in ctex


def test_figures_written(tmp_path):
    preds = _fake_preds()
    preds.loc[preds.model == "clone_of_ref", "model"] = "static_dyn_unc"
    meta = pd.DataFrame({"short_arc": 0}, index=[f"o{i}" for i in range(3000)])
    out = evaluate_predictions(preds, meta, EvalConfig(n_boot=20))
    plot_pr_curves(preds, tmp_path / "pr")
    plot_cost_curve(preds, tmp_path / "cost")
    plot_conformal_tradeoff(out["conformal"], tmp_path / "conf")
    plot_reliability(preds, tmp_path / "rel")
    for n in ("pr", "cost", "conf", "rel"):
        for ext in (".pdf", ".png"):
            f = tmp_path / f"{n}{ext}"
            assert f.exists() and f.stat().st_size > 2000


def test_conformal_pvalues_match_mondrian_sets():
    """The web export forms sets from p-values (class in set iff p > alpha); this must agree
    exactly with MondrianConformal's order-statistic quantile rule for every alpha."""
    from src.model import MondrianConformal, conformal_pvalues
    rng = np.random.default_rng(7)
    y_cal = (rng.random(800) < 0.08).astype(int)
    p_cal = _sigmoid(2.5 * y_cal + rng.normal(size=800) - 3)
    p_new = _sigmoid(rng.normal(size=3000) * 2 - 2)
    pvh = conformal_pvalues(1 - p_cal[y_cal == 1], 1 - p_new)
    pvb = conformal_pvalues(p_cal[y_cal == 0], p_new)
    for ab, ah in ((0.10, 0.01), (0.10, 0.05), (0.05, 0.10), (0.20, 0.30)):
        sets = MondrianConformal(ab, ah).fit(p_cal, y_cal).predict_sets(p_new)
        assert ((sets & 2) > 0).tolist() == (pvh > ah).tolist()
        assert ((sets & 1) > 0).tolist() == (pvb > ab).tolist()
