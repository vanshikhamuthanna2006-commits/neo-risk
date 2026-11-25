"""Stage 5: publication evaluation - metrics, paired bootstrap CIs over seeds, short-arc strata,
conformal trade-offs, figures and LaTeX tables.

Uncertainty is reported at two levels:
* between-seed SD (each seed re-draws the split and re-seeds the models);
* a stratified bootstrap of each seed's test set (class counts fixed), *paired* across models
  (same resampled indices), averaged replicate-wise across seeds -> 95% percentile CI of the
  seed-averaged metric and of differences vs the reference model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .model import SET_BOTH, SET_EMPTY, MondrianConformal, bayes_threshold

MODEL_LABELS = {
    "naive_moid_h": "Naive (MOID + $H$)",
    "static_orbit_only": "Static orbit only",
    "static_leakfree": "Static leakage-free",
    "static_dyn": "+ N-body dynamics",
    "static_dyn_unc": "+ dynamics + uncertainty",
    "static_dyn_unc_focal": "+ asym.\\ focal loss",
    "dual_resmlp_focal": "Dual-branch ResMLP",
    "proposed_ensemble": "Proposed ensemble",
}
METRICS = ("pr_auc", "roc_auc", "recall_at_fpr", "cost_per_1k", "brier")
HIGHER_IS_BETTER = {"pr_auc": True, "roc_auc": True, "recall_at_fpr": True, "cost_per_1k": False, "brier": False}
METRIC_LABELS = {"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "recall_at_fpr": "Recall@1\\%FPR",
                 "cost_per_1k": "Cost/1k", "brier": "Brier"}


@dataclass
class EvalConfig:
    c_fn: float = 100.0
    c_fp: float = 1.0
    max_fpr: float = 0.01
    n_boot: int = 500
    seed: int = 0
    reference: str = "static_leakfree"
    alpha_benign: float = 0.10
    alphas_haz: tuple[float, ...] = (0.01, 0.05, 0.10)


# ======================================================================================
# Point metrics
# ======================================================================================
def recall_at_fpr(y: np.ndarray, s: np.ndarray, max_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def expected_cost_per_1k(y: np.ndarray, p: np.ndarray, c_fn: float, c_fp: float) -> float:
    """Mean misclassification cost per 1000 objects when flagging p > C_FP/(C_FP+C_FN)."""
    flag = p > bayes_threshold(c_fn, c_fp)
    return 1000.0 * float(np.mean(c_fn * ((~flag) & (y == 1)) + c_fp * (flag & (y == 0))))


def trivial_cost_per_1k(y: np.ndarray, c_fn: float, c_fp: float) -> float:
    """Cost of the better trivial policy (flag none / flag all)."""
    return 1000.0 * min(c_fn * y.mean(), c_fp * (1 - y.mean()))


def point_metrics(y: np.ndarray, logit: np.ndarray, p: np.ndarray, cfg: EvalConfig) -> dict[str, float]:
    if y.min() == y.max():
        return {m: np.nan for m in METRICS}
    return {"pr_auc": average_precision_score(y, logit), "roc_auc": roc_auc_score(y, logit),
            "recall_at_fpr": recall_at_fpr(y, logit, cfg.max_fpr),
            "cost_per_1k": expected_cost_per_1k(y, p, cfg.c_fn, cfg.c_fp),
            "brier": float(np.mean((p - y) ** 2))}


# ======================================================================================
# Bootstrap
# ======================================================================================
def stratified_bootstrap_indices(y: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n) indices resampling positives and negatives separately (counts preserved)."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return np.hstack([rng.choice(pos, (n_boot, len(pos))), rng.choice(neg, (n_boot, len(neg)))])


def bootstrap_seed(test: pd.DataFrame, cfg: EvalConfig, seed: int) -> pd.DataFrame:
    """Replicate metrics for every model in one seed's test predictions (paired indices).

    ``test`` columns: model, pdes, y, logit, p.  Returns rows (model, b, metric...).
    """
    wide_y = test.drop_duplicates("pdes").set_index("pdes")["y"]
    y = wide_y.to_numpy()
    rng = np.random.default_rng([cfg.seed, seed])
    idx = stratified_bootstrap_indices(y, cfg.n_boot, rng)
    rows = []
    for model, d in test.groupby("model", sort=False):
        d = d.set_index("pdes").loc[wide_y.index]
        lg, pp = d["logit"].to_numpy(), d["p"].to_numpy()
        for b in range(cfg.n_boot):
            ii = idx[b]
            rows.append({"model": model, "b": b, **point_metrics(y[ii], lg[ii], pp[ii], cfg)})
    return pd.DataFrame(rows)


def summarize(point: pd.DataFrame, boot: pd.DataFrame, cfg: EvalConfig) -> pd.DataFrame:
    """Seed-mean, seed-SD and bootstrap CI for each metric; paired delta vs the reference.

    point: rows (seed, model, metrics); boot: rows (seed, model, b, metrics).
    """
    mean = point.groupby("model")[list(METRICS)].mean()
    sd = point.groupby("model")[list(METRICS)].std(ddof=1)
    rep = boot.groupby(["model", "b"])[list(METRICS)].mean()          # seed-averaged replicates
    lo = rep.groupby("model").quantile(0.025)
    hi = rep.groupby("model").quantile(0.975)
    out = pd.DataFrame(index=mean.index)
    for m in METRICS:
        out[m], out[f"{m}_sd"], out[f"{m}_lo"], out[f"{m}_hi"] = mean[m], sd[m], lo[m], hi[m]
    if cfg.reference in rep.index.get_level_values(0):
        ref = rep.xs(cfg.reference, level="model")
        for model in out.index:
            dlt = rep.xs(model, level="model") - ref
            for m in METRICS:
                out.loc[model, f"d_{m}"] = mean.loc[model, m] - mean.loc[cfg.reference, m]
                out.loc[model, f"d_{m}_lo"] = dlt[m].quantile(0.025)
                out.loc[model, f"d_{m}_hi"] = dlt[m].quantile(0.975)
                worse = dlt[m] <= 0 if HIGHER_IS_BETTER[m] else dlt[m] >= 0
                out.loc[model, f"d_{m}_p_not_better"] = float(worse.mean())
    out["n_seeds"] = point.groupby("model").size()
    return out


# ======================================================================================
# Conformal trade-off
# ======================================================================================
def conformal_tradeoff(pred: pd.DataFrame, cfg: EvalConfig) -> pd.DataFrame:
    """Re-threshold conformal sets from stored calibration predictions (no retraining).

    pred: one model, one seed, with split in {cal, test}.
    """
    cal, te = pred[pred.split == "cal"], pred[pred.split == "test"]
    rows = []
    for a in cfg.alphas_haz:
        conf = MondrianConformal(cfg.alpha_benign, a).fit(cal.p.to_numpy(), cal.y.to_numpy())
        sets = conf.predict_sets(te.p.to_numpy())
        y = te.y.to_numpy()
        cov = MondrianConformal.covers(sets, y)
        rows.append({"alpha_haz": a, "haz_coverage": cov[y == 1].mean(), "benign_coverage": cov[y == 0].mean(),
                     "follow_up_rate": float(np.isin(sets, [SET_BOTH, SET_EMPTY]).mean()),
                     "haz_only_rate": float((sets == 2).mean()), "n_cal_pos": int((cal.y == 1).sum()),
                     "q_haz_finite": bool(np.isfinite(conf.q[1]))})
    return pd.DataFrame(rows)


# ======================================================================================
# Evaluation driver
# ======================================================================================
def evaluate_predictions(preds: pd.DataFrame, meta: pd.DataFrame, cfg: EvalConfig | None = None,
                         short_arc_col: str = "short_arc") -> dict[str, pd.DataFrame]:
    """preds: long frame with columns seed, model, pdes, split, y, logit, p, conf_set."""
    from joblib import Parallel, delayed

    cfg = cfg or EvalConfig()
    test = preds[preds.split == "test"].copy()
    test["short_arc"] = meta[short_arc_col].reindex(test.pdes).to_numpy()
    out: dict[str, pd.DataFrame] = {}
    for stratum, sub in (("all", test), ("short_arc", test[test.short_arc == 1])):
        if sub.empty or sub.y.nunique() < 2:            # e.g. no short-arc objects / one class only
            continue
        point = pd.DataFrame([{"seed": s, "model": m, **point_metrics(d.y.to_numpy(), d.logit.to_numpy(),
                                                                       d.p.to_numpy(), cfg)}
                              for (s, m), d in sub.groupby(["seed", "model"], sort=False)])
        boots = Parallel(n_jobs=-1)(delayed(bootstrap_seed)(d, cfg, s) for s, d in sub.groupby("seed"))
        boot = pd.concat([b.assign(seed=s) for s, b in zip(sorted(sub.seed.unique()), boots)])
        out[f"summary_{stratum}"] = summarize(point, boot, cfg)
        out[f"point_{stratum}"] = point
        n = sub.drop_duplicates(["seed", "pdes"]).groupby("seed").agg(n=("y", "size"), pos=("y", "sum"))
        out[f"counts_{stratum}"] = n
    out["conformal"] = pd.concat([conformal_tradeoff(d, cfg).assign(seed=s, model=m)
                                  for (s, m), d in preds.groupby(["seed", "model"], sort=False)])
    out["trivial_cost_per_1k"] = pd.DataFrame(
        [{"seed": s, "cost": trivial_cost_per_1k(d.drop_duplicates("pdes").y.to_numpy(), cfg.c_fn, cfg.c_fp)}
         for s, d in test.groupby("seed")])
    return out


# ======================================================================================
# LaTeX
# ======================================================================================
def _fmt(v: float, nd: int) -> str:
    return "--" if not np.isfinite(v) else f"{v:.{nd}f}"


def latex_ablation_table(summary: pd.DataFrame, caption: str, label: str, counts: pd.DataFrame | None = None,
                         order: tuple[str, ...] = tuple(MODEL_LABELS), reference: str = "static_leakfree",
                         trivial_cost: float | None = None) -> str:
    nd = {"pr_auc": 3, "roc_auc": 3, "recall_at_fpr": 3, "cost_per_1k": 1, "brier": 4}
    rows = [m for m in order if m in summary.index]
    best = {}
    for m in METRICS:
        cand = summary.loc[[r for r in rows if r != "naive_moid_h"], m]
        best[m] = cand.idxmax() if HIGHER_IS_BETTER[m] else cand.idxmin()
    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
             r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\begin{tabular}{l" + "c" * (len(METRICS) + 1) + "}", r"\toprule",
             "Model & " + " & ".join(METRIC_LABELS[m] for m in METRICS) + r" & $\Delta$PR-AUC \\", r"\midrule"]
    for r in rows:
        cells = []
        for m in METRICS:
            s = f"{_fmt(summary.loc[r, m], nd[m])} \\scriptsize{{[{_fmt(summary.loc[r, m + '_lo'], nd[m])}, {_fmt(summary.loc[r, m + '_hi'], nd[m])}]}}"
            if r == best[m]:
                s = r"\textbf{" + _fmt(summary.loc[r, m], nd[m]) + "}" + s[len(_fmt(summary.loc[r, m], nd[m])):]
            cells.append(s)
        if r == reference or "d_pr_auc" not in summary:
            d = "(ref.)" if r == reference else "--"
        else:
            d = (f"{summary.loc[r, 'd_pr_auc']:+.3f} \\scriptsize{{[{summary.loc[r, 'd_pr_auc_lo']:+.3f}, "
                 f"{summary.loc[r, 'd_pr_auc_hi']:+.3f}]}}")
        lines.append(f"{MODEL_LABELS.get(r, r)} & " + " & ".join(cells) + f" & {d} \\\\")
        if r == "naive_moid_h":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    notes = [r"Mean over seeds; brackets: 95\% paired stratified-bootstrap CI of the seed-averaged metric.",
             r"Bold: best leakage-free model. The naive row uses label-defining inputs and is shown only to expose circularity."]
    if counts is not None:
        notes.append(f"Test set per seed: $n={int(counts.n.mean())}$, positives $={counts.pos.mean():.0f}$.")
    if trivial_cost is not None:
        notes.append(f"Cost/1k uses $C_{{FN}}{{:}}C_{{FP}}=100{{:}}1$ at the Bayes threshold; best trivial policy: {trivial_cost:.1f}.")
    lines += [r"\par\vspace{2pt}\footnotesize " + " ".join(notes), r"\end{table*}"]
    return "\n".join(lines) + "\n"


def latex_conformal_table(conf: pd.DataFrame, models: tuple[str, ...], caption: str, label: str) -> str:
    agg = conf[conf.model.isin(models)].groupby(["model", "alpha_haz"]).agg(
        haz=("haz_coverage", "mean"), ben=("benign_coverage", "mean"), fu=("follow_up_rate", "mean"),
        npos=("n_cal_pos", "mean")).reset_index()
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\begin{tabular}{llccc}", r"\toprule",
             r"Model & $\alpha_H$ & Haz.\ coverage & Benign coverage & Follow-up rate \\", r"\midrule"]
    for m in models:
        for _, r in agg[agg.model == m].iterrows():
            lines.append(f"{MODEL_LABELS.get(m, m)} & {r.alpha_haz:.2f} & {r.haz:.3f} & {r.ben:.3f} & {r.fu:.3f} \\\\")
        lines.append(r"\midrule" if m != models[-1] else r"\bottomrule")
    lines += [r"\end{tabular}", r"\par\vspace{2pt}\footnotesize Mondrian split conformal; $\alpha_B=0.10$. "
              f"Mean over seeds; calibration positives per seed $\\approx {agg.npos.mean():.0f}$.", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ======================================================================================
# Figures (validated categorical palette, light surface; legend + direct labels + dashes)
# ======================================================================================
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # dataviz reference slots 1-4 (validated)
DASHES = ["-", "--", "-.", ":"]
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
FIG_MODELS = ("naive_moid_h", "static_leakfree", "static_dyn_unc", "proposed_ensemble")


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color="#e8e7e3", linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor=SURFACE)


def plot_pr_curves(preds: pd.DataFrame, path: Path, seed: int = 0, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    fig, ax = plt.subplots(figsize=(4.6, 3.6), facecolor=SURFACE)
    _style(ax)
    te = preds[(preds.split == "test") & (preds.seed == seed)]
    for k, m in enumerate(FIG_MODELS):
        d = te[te.model == m]
        if d.empty:
            continue
        pr, rc, _ = precision_recall_curve(d.y, d.logit)
        ap = average_precision_score(d.y, d.logit)
        ax.plot(rc, pr, color=PALETTE[k], ls=DASHES[k], lw=2, label=f"{MODEL_LABELS[m].replace('$', '')} ({ap:.3f})")
    base = te.drop_duplicates("pdes").y.mean()
    ax.axhline(base, color=MUTED, lw=1, ls=":")
    ax.text(0.99, base + 0.015, f"prevalence {base:.3f}", color=INK2, fontsize=7, ha="right")
    ax.set_xlabel("Recall", color=INK, fontsize=9)
    ax.set_ylabel("Precision", color=INK, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(title, color=INK, fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower left", bbox_to_anchor=(0.0, base + 0.04), labelcolor=INK)
    _save(fig, path)
    plt.close(fig)


def _repel(ys: list[float], min_gap: float) -> list[float]:
    """Spread label y-positions (data units) so neighbours are at least ``min_gap`` apart."""
    order = np.argsort(ys)
    out = np.array(ys, float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out.tolist()


def plot_cost_curve(preds: pd.DataFrame, path: Path, ratios=np.logspace(0, 3, 25), title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 3.6), facecolor=SURFACE)
    _style(ax)
    te = preds[preds.split == "test"]
    ends: list[tuple[float, str]] = []
    for k, m in enumerate(FIG_MODELS[1:], start=1):              # naive excluded: not deployable
        d = te[te.model == m]
        if d.empty:
            continue
        c = [np.mean([expected_cost_per_1k(g.y.to_numpy(), g.p.to_numpy(), r, 1.0) / trivial_cost_per_1k(g.y.to_numpy(), r, 1.0)
                      for _, g in d.groupby("seed")]) for r in ratios]
        ax.plot(ratios, c, color=PALETTE[k], ls=DASHES[k], lw=2, label=MODEL_LABELS[m].replace("$", ""))
        ends.append((c[-1], MODEL_LABELS[m].replace("$", "")))
    lo, hi = ax.get_ylim()
    ys = _repel([e[0] for e in ends], 0.045 * (hi - lo))
    for (y0, lab), yl in zip(ends, ys):
        ax.annotate(lab, (ratios[-1], y0), xytext=(ratios[-1] * 1.12, yl), fontsize=6, color=INK2, va="center",
                    annotation_clip=False)
    ax.axhline(1.0, color=MUTED, lw=1, ls=":")
    ax.set_xscale("log")                                          # scale first, then limits
    ax.set_xlim(ratios[0] * 0.9, ratios[-1] * 1.05)
    ax.set_xlabel(r"Cost ratio $C_{FN}/C_{FP}$", color=INK, fontsize=9)
    ax.set_ylabel("Expected cost / best trivial policy", color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False, labelcolor=INK)
    _save(fig, path)
    plt.close(fig)


def plot_conformal_tradeoff(conf: pd.DataFrame, path: Path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 3.6), facecolor=SURFACE)
    _style(ax)
    agg = conf.groupby(["model", "alpha_haz"])[["haz_coverage", "follow_up_rate"]].mean().reset_index()
    for k, m in enumerate(FIG_MODELS[1:], start=1):
        d = agg[agg.model == m].sort_values("alpha_haz")
        if d.empty:
            continue
        ax.plot(d.follow_up_rate, d.haz_coverage, color=PALETTE[k], ls=DASHES[k], lw=2, marker="o", ms=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=MODEL_LABELS[m].replace("$", ""))
        for _, r in d.iterrows():
            if k == 3:                                            # label alphas once
                ax.annotate(f"$\\alpha_H$={r.alpha_haz:.2f}", (r.follow_up_rate, r.haz_coverage), fontsize=6,
                            color=INK2, xytext=(4, -9), textcoords="offset points")
    ax.set_xlabel("Follow-up rate (uncertain sets)", color=INK, fontsize=9)
    ax.set_ylabel("Hazardous coverage", color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower right", labelcolor=INK)
    _save(fig, path)
    plt.close(fig)


def plot_reliability(preds: pd.DataFrame, path: Path, n_bins: int = 10, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.0, 3.6), facecolor=SURFACE)
    _style(ax)
    te = preds[preds.split == "test"]
    edges = np.quantile(te[te.model == FIG_MODELS[1]].p, np.linspace(0, 1, n_bins + 1))
    for k, m in enumerate(FIG_MODELS[1:], start=1):
        d = te[te.model == m]
        if d.empty:
            continue
        b = np.clip(np.searchsorted(edges, d.p, side="right") - 1, 0, n_bins - 1)
        g = d.assign(b=b).groupby("b").agg(p=("p", "mean"), y=("y", "mean"))
        ax.plot(g.p, g.y, color=PALETTE[k], ls=DASHES[k], lw=2, marker="o", ms=4, markeredgecolor=SURFACE,
                label=MODEL_LABELS[m].replace("$", ""))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel("Predicted probability (calibrated)", color=INK, fontsize=9)
    ax.set_ylabel("Observed frequency", color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False, labelcolor=INK)
    _save(fig, path)
    plt.close(fig)
