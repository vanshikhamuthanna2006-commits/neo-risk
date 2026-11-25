"""Stage 4: cost-sensitive, uncertainty-aware classifiers, baselines and conformal prediction.

Components
----------
* Feature assembly with an enforced leakage gate (only the naive baseline may bypass it).
* Splits: stratified (PHA x short-arc) or temporal (by first observation); train/val/cal/test.
* Asymmetric focal loss with false-negative cost:
      L = -lambda_FN (1-p)^gamma y log p  -  lambda_FP p^gamma (1-y) log(1-p)
  gamma=0, lambda=1 reduces exactly to binary log-loss.  Used as a LightGBM custom objective
  (analytic gradient, numerical Hessian clipped > 0) and as a torch loss.
* Dual-branch residual MLP (branch 1: static + priors; branch 2: dynamics + uncertainty).
* Platt calibration on the validation split (focal losses distort probabilities).
* Mondrian (label-conditional) split conformal prediction:
      P(y in C(X) | Y = y) >= 1 - alpha_y   for each class separately.
  Marginal conformal would let hazardous coverage collapse under 6% prevalence.
No synthetic oversampling (SMOTE) is used anywhere.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_pipeline import assert_no_leakage

log = logging.getLogger(__name__)

# ======================================================================================
# Feature assembly & splits
# ======================================================================================
FEATURE_SETS = ("naive", "static_orbit", "static", "static+dyn", "static+dyn+unc")


@dataclass
class FeatureFrame:
    F: pd.DataFrame
    branch1: list[str]      # static orbit / obs metadata / physical priors
    branch2: list[str]      # dynamics / uncertainty
    name: str


def assemble_features(feature_set: str, X: pd.DataFrame, groups: dict[str, list[str]],
                      audit: pd.DataFrame, dyn: pd.DataFrame | None = None,
                      unc: pd.DataFrame | None = None) -> FeatureFrame:
    if feature_set not in FEATURE_SETS:
        raise ValueError(feature_set)
    if feature_set == "static_orbit":
        F = X[groups["static_orbit"]].copy()
    else:
        F = X.copy()
    b1 = list(F.columns)
    b2: list[str] = []
    if feature_set == "naive":
        F = F.join(audit[["moid", "H"]])                 # the circularity trap, deliberately
        b1 = list(F.columns)
        return FeatureFrame(F, b1, b2, feature_set)
    if "dyn" in feature_set:
        if dyn is None:
            raise ValueError("dynamics features required")
        dcols = [c for c in dyn.columns if c.startswith(("dyn_", "has_dyn"))]
        F = F.join(dyn[dcols])
        b2 += dcols
    if "unc" in feature_set:
        if unc is None:
            raise ValueError("uncertainty features required")
        ucols = [c for c in unc.columns if c.startswith(("unc_", "has_unc"))]
        pcols = [c for c in unc.columns if c.startswith(("phys_", "has_phys"))]
        F = F.join(unc[ucols + pcols])
        b2 += ucols
        b1 += pcols
    assert_no_leakage(F, audit)                          # hard gate for every non-naive set
    return FeatureFrame(F, b1, b2, feature_set)


def make_splits(y: pd.Series, meta: pd.DataFrame, scheme: str = "stratified", seed: int = 0,
                fracs: tuple[float, float, float, float] = (0.60, 0.10, 0.15, 0.15)) -> dict[str, np.ndarray]:
    """Return positional index arrays for train / val / cal / test."""
    if abs(sum(fracs) - 1) > 1e-9:
        raise ValueError("fractions must sum to 1")
    n = len(y)
    rng = np.random.default_rng(seed)
    names = ("train", "val", "cal", "test")
    out: dict[str, list[int]] = {k: [] for k in names}
    if scheme == "stratified":
        strata = y.to_numpy().astype(int) * 2 + meta["short_arc"].reindex(y.index).fillna(0).to_numpy().astype(int)
        for s in np.unique(strata):
            idx = rng.permutation(np.flatnonzero(strata == s))
            cuts = np.round(np.cumsum(fracs)[:-1] * len(idx)).astype(int)
            for k, part in zip(names, np.split(idx, cuts)):
                out[k] += part.tolist()
    elif scheme == "temporal":
        first = pd.to_datetime(meta["first_obs"].reindex(y.index), errors="coerce")
        order = np.argsort(first.fillna(first.min()).to_numpy(), kind="stable")
        n_old = int(round((fracs[0] + fracs[1]) * n))
        old, new = order[:n_old], order[n_old:]
        old = rng.permutation(old)
        n_tr = int(round(fracs[0] / (fracs[0] + fracs[1]) * len(old)))
        new = rng.permutation(new)
        n_cal = int(round(fracs[2] / (fracs[2] + fracs[3]) * len(new)))
        out = {"train": old[:n_tr].tolist(), "val": old[n_tr:].tolist(),
               "cal": new[:n_cal].tolist(), "test": new[n_cal:].tolist()}
    else:
        raise ValueError(scheme)
    return {k: np.sort(np.asarray(v, int)) for k, v in out.items()}


# ======================================================================================
# Asymmetric focal loss
# ======================================================================================
@dataclass
class AsymFocalConfig:
    gamma: float = 2.0
    lambda_fn: float = 10.0       # weight on positives (missed hazardous objects are expensive)
    lambda_fp: float = 1.0


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def asym_focal_loss(z: np.ndarray, y: np.ndarray, cfg: AsymFocalConfig) -> np.ndarray:
    p = np.clip(_sigmoid(z), 1e-15, 1 - 1e-15)
    return -(cfg.lambda_fn * y * (1 - p) ** cfg.gamma * np.log(p)
             + cfg.lambda_fp * (1 - y) * p ** cfg.gamma * np.log(1 - p))


def asym_focal_grad(z: np.ndarray, y: np.ndarray, cfg: AsymFocalConfig) -> np.ndarray:
    """dL/dz (analytic)."""
    p = np.clip(_sigmoid(z), 1e-15, 1 - 1e-15)
    g, w1, w0 = cfg.gamma, cfg.lambda_fn, cfg.lambda_fp
    g1 = w1 * (1 - p) ** g * (g * p * np.log(p) - (1 - p))
    g0 = w0 * p ** g * (p - g * (1 - p) * np.log(1 - p))
    return y * g1 + (1 - y) * g0


def asym_focal_grad_hess(z: np.ndarray, y: np.ndarray, cfg: AsymFocalConfig, hessian: str = "surrogate",
                         eps: float = 1e-4, h_min: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Exact gradient plus a curvature term for second-order boosting.

    hessian="exact": numerical d2L/dz2 (clipped > 0).  It scales as p^(gamma+1) for easy negatives,
      so in mixed leaves the Newton step -G/H tends to ~1/p regardless of how many negatives the
      leaf holds (observed: +20-logit first-tree leaves; leaf caps then saturate every split).
    hessian="surrogate" (default): weighted-logistic curvature lambda_y * p (1 - p), a positive-
      definite quasi-Newton preconditioner.  The gradient is exact, so stationary points are those
      of the focal objective; with gamma=0 and lambda=1 it equals the exact log-loss Hessian.
    """
    g = asym_focal_grad(z, y, cfg)
    if hessian == "exact":
        h = (asym_focal_grad(z + eps, y, cfg) - asym_focal_grad(z - eps, y, cfg)) / (2 * eps)
    elif hessian == "surrogate":
        p = _sigmoid(z)
        h = np.where(y == 1, cfg.lambda_fn, cfg.lambda_fp) * p * (1 - p)
    else:
        raise ValueError(hessian)
    return g, np.maximum(h, h_min)


# ======================================================================================
# Models
# ======================================================================================
def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


@dataclass
class LGBMConfig:
    objective: str = "binary"                 # "binary" | "asym_focal"
    focal: AsymFocalConfig = field(default_factory=AsymFocalConfig)
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 3000
    early_stopping: int = 100
    min_child_samples: int = 20
    feature_fraction: float = 0.9
    # (An apparent "bagging + custom objective" failure was misdiagnosed: the real cause was the
    # leaf cap saturating split gains; with the surrogate Hessian bagging trains normally.)
    bagging_fraction: float = 0.8
    seed: int = 0
    num_threads: int = 4
    # Leaf-output cap (0 = off).  NOT used to stabilise the focal objective: LightGBM applies it
    # inside the split gain, so when every child's Newton step exceeds the cap all leaves saturate
    # and no split has positive gain (training stalls).  Stability comes from the surrogate
    # Hessian in ``asym_focal_grad_hess`` instead.
    max_delta_step: float = 0.0


class LGBMModel:
    def __init__(self, cfg: LGBMConfig | None = None):
        self.cfg = cfg or LGBMConfig()
        self.booster = None
        self.init = 0.0
        self.columns: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray) -> "LGBMModel":
        import lightgbm as lgb
        c = self.cfg
        self.columns = list(X.columns)
        self.init = _logit(float(np.clip(y.mean(), 1e-4, 1 - 1e-4)))
        params: dict[str, Any] = {
            "num_leaves": c.num_leaves, "learning_rate": c.learning_rate,
            "min_child_samples": c.min_child_samples, "feature_fraction": c.feature_fraction,
            "bagging_fraction": c.bagging_fraction, "bagging_freq": 1 if c.bagging_fraction < 1 else 0,
            "seed": c.seed,
            "deterministic": True, "force_row_wise": True, "num_threads": c.num_threads, "verbose": -1,
        }
        dtr = lgb.Dataset(X, y, init_score=np.full(len(y), self.init), free_raw_data=False)
        dva = lgb.Dataset(X_val, y_val, init_score=np.full(len(y_val), self.init), reference=dtr)
        cb = [lgb.early_stopping(c.early_stopping, verbose=False)]
        if c.objective == "binary":
            params.update(objective="binary", metric="binary_logloss")
            self.booster = lgb.train(params, dtr, c.n_estimators, valid_sets=[dva], callbacks=cb)
        elif c.objective == "asym_focal":
            focal = c.focal
            params.update(objective=lambda preds, data: asym_focal_grad_hess(preds, data.get_label(), focal),
                          metric="None", max_delta_step=c.max_delta_step)
            self.booster = lgb.train(params, dtr, c.n_estimators, valid_sets=[dva], callbacks=cb,
                                     feval=lambda preds, data: ("asym_focal",
                                                                float(np.mean(asym_focal_loss(preds, data.get_label(), focal))),
                                                                False))
        else:
            raise ValueError(c.objective)
        return self

    def predict_logit(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(X[self.columns], raw_score=True,
                                    num_iteration=self.booster.best_iteration) + self.init

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return _sigmoid(self.predict_logit(X))


# ---------------------------------------------------------------- dual-branch residual MLP
@dataclass
class MLPConfig:
    width: int = 128
    emb: int = 64
    n_blocks: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 80
    patience: int = 10
    seed: int = 0
    focal: AsymFocalConfig = field(default_factory=AsymFocalConfig)
    clip: float = 10.0


class RobustScaler:
    """Median / IQR scaling fitted on training data; NaN -> 0 after scaling (masks carry missingness)."""

    def fit(self, X: pd.DataFrame) -> "RobustScaler":
        self.columns = list(X.columns)
        self.med = X.median().fillna(0.0).to_numpy()
        iqr = (X.quantile(0.75) - X.quantile(0.25)).to_numpy()
        sd = X.std().fillna(1.0).to_numpy()
        self.scale = np.where(np.isfinite(iqr) & (iqr > 0), iqr, np.where(sd > 0, sd, 1.0))
        return self

    def transform(self, X: pd.DataFrame, clip: float) -> np.ndarray:
        Z = (X[self.columns].to_numpy(float) - self.med) / self.scale
        Z = np.clip(Z, -clip, clip)
        return np.nan_to_num(Z, nan=0.0).astype(np.float32)


def _build_mlp(d1: int, d2: int, cfg: MLPConfig):
    import torch
    from torch import nn

    class ResBlock(nn.Module):
        def __init__(self, w: int, p: float):
            super().__init__()
            self.net = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, w), nn.GELU(), nn.Dropout(p), nn.Linear(w, w))

        def forward(self, x):
            return x + self.net(x)

    def branch(d: int) -> nn.Module:
        return nn.Sequential(nn.Linear(d, cfg.width), *[ResBlock(cfg.width, cfg.dropout) for _ in range(cfg.n_blocks)],
                             nn.LayerNorm(cfg.width), nn.Linear(cfg.width, cfg.emb))

    class DualBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.b1 = branch(d1)
            self.b2 = branch(d2) if d2 > 0 else None
            fused = cfg.emb * (2 if d2 > 0 else 1)
            self.head = nn.Sequential(nn.GELU(), ResBlock(fused, cfg.dropout), nn.LayerNorm(fused), nn.Linear(fused, 1))

        def forward(self, x1, x2):
            h = self.b1(x1)
            if self.b2 is not None:
                h = torch.cat([h, self.b2(x2)], dim=1)
            return self.head(h).squeeze(1)

    return DualBranch()


def asym_focal_loss_torch(logits, y, cfg: AsymFocalConfig):
    import torch
    logp = torch.nn.functional.logsigmoid(logits)
    log1mp = torch.nn.functional.logsigmoid(-logits)
    p = torch.exp(logp)
    return -(cfg.lambda_fn * y * (1 - p) ** cfg.gamma * logp + cfg.lambda_fp * (1 - y) * p ** cfg.gamma * log1mp)


class DualBranchMLP:
    def __init__(self, branch1: list[str], branch2: list[str], cfg: MLPConfig | None = None):
        self.cfg = cfg or MLPConfig()
        self.b1, self.b2 = list(branch1), list(branch2)

    def _xs(self, X: pd.DataFrame):
        import torch
        x1 = torch.from_numpy(self.s1.transform(X[self.b1], self.cfg.clip))
        x2 = torch.from_numpy(self.s2.transform(X[self.b2], self.cfg.clip)) if self.b2 else torch.zeros((len(X), 0))
        return x1, x2

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray) -> "DualBranchMLP":
        import torch
        c = self.cfg
        torch.manual_seed(c.seed)
        torch.use_deterministic_algorithms(True)
        self.s1 = RobustScaler().fit(X[self.b1])
        self.s2 = RobustScaler().fit(X[self.b2]) if self.b2 else None
        x1, x2 = self._xs(X)
        v1, v2 = self._xs(X_val)
        yt, yv = torch.tensor(y, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
        self.net = _build_mlp(x1.shape[1], x2.shape[1], c)
        with torch.no_grad():                                    # start at the class prior
            self.net.head[-1].bias.fill_(_logit(float(np.clip(y.mean(), 1e-4, 1 - 1e-4))))
        opt = torch.optim.AdamW(self.net.parameters(), lr=c.lr, weight_decay=c.weight_decay)
        g = torch.Generator().manual_seed(c.seed)
        best, best_state, bad = np.inf, None, 0
        self.history: list[float] = []
        for epoch in range(c.max_epochs):
            self.net.train()
            perm = torch.randperm(len(yt), generator=g)
            for i in range(0, len(perm), c.batch_size):
                b = perm[i:i + c.batch_size]
                loss = asym_focal_loss_torch(self.net(x1[b], x2[b]), yt[b], c.focal).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = float(asym_focal_loss_torch(self.net(v1, v2), yv, c.focal).mean())
            self.history.append(vl)
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= c.patience:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def predict_logit(self, X: pd.DataFrame) -> np.ndarray:
        import torch
        x1, x2 = self._xs(X)
        with torch.no_grad():
            return self.net(x1, x2).numpy().astype(float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return _sigmoid(self.predict_logit(X))


# ---------------------------------------------------------------- calibration & ensemble
class PlattCalibrator:
    """p = sigmoid(a * logit + b), fitted by (unweighted) log-loss on the validation split."""

    def fit(self, logit: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit.reshape(-1, 1), y)
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def __call__(self, logit: np.ndarray) -> np.ndarray:
        return _sigmoid(self.a * logit + self.b)


class LogitEnsemble:
    def __init__(self, models: list[Any], weights: list[float] | None = None):
        self.models = models
        self.w = np.asarray(weights or [1.0] * len(models), float)
        self.w = self.w / self.w.sum()

    def predict_logit(self, X: pd.DataFrame) -> np.ndarray:
        return sum(w * m.predict_logit(X) for w, m in zip(self.w, self.models))


# ======================================================================================
# Conformal prediction (Mondrian / label-conditional split conformal)
# ======================================================================================
SET_EMPTY, SET_BENIGN, SET_HAZ, SET_BOTH = 0, 1, 2, 3
SET_LABELS = {SET_BENIGN: "Benign", SET_HAZ: "Hazardous", SET_BOTH: "Uncertain/Follow-up",
              SET_EMPTY: "Uncertain/Follow-up (atypical)"}


class MondrianConformal:
    """Per-class split conformal on calibrated probabilities.

    Nonconformity s(x, y) = 1 - p(y | x).  For each class y with n_y calibration points,
    q_y is the ceil((n_y + 1)(1 - alpha_y)) / n_y empirical quantile ("higher" interpolation);
    if that rank exceeds n_y the class is always included (q_y = +inf).
    Guarantee (exchangeability): P(y in C(X) | Y = y) >= 1 - alpha_y.
    """

    def __init__(self, alpha_benign: float = 0.10, alpha_hazardous: float = 0.01):
        self.alpha = {0: alpha_benign, 1: alpha_hazardous}

    def fit(self, p_cal: np.ndarray, y_cal: np.ndarray) -> "MondrianConformal":
        self.q: dict[int, float] = {}
        for cls in (0, 1):
            s = 1.0 - (p_cal[y_cal == cls] if cls == 1 else 1.0 - p_cal[y_cal == 0])
            n = len(s)
            k = int(np.ceil((n + 1) * (1 - self.alpha[cls])))
            self.q[cls] = np.inf if k > n or n == 0 else float(np.sort(s)[k - 1])
        self.n_cal = {cls: int((y_cal == cls).sum()) for cls in (0, 1)}
        return self

    def predict_sets(self, p: np.ndarray) -> np.ndarray:
        inc_h = (1.0 - p) <= self.q[1]
        inc_b = p <= self.q[0]                    # s(x, benign) = 1 - (1 - p) = p
        return inc_b * SET_BENIGN + inc_h * SET_HAZ

    @staticmethod
    def covers(sets: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.where(y == 1, (sets & SET_HAZ) > 0, (sets & SET_BENIGN) > 0)


def conformal_pvalues(cal_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Split-conformal p-values p = (1 + #{cal >= s}) / (n + 1).  Including a class iff
    p > alpha reproduces ``MondrianConformal`` (q = ceil((n+1)(1-alpha))-th smallest score)."""
    c = np.sort(np.asarray(cal_scores, float))
    ge = len(c) - np.searchsorted(c, np.asarray(scores, float), side="left")
    return (1 + ge) / (len(c) + 1)


def recall_at_fpr(y: np.ndarray, score: np.ndarray, max_fpr: float = 0.01) -> float:
    """Highest recall achievable with false-positive rate <= max_fpr (threshold-free)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, score)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def bayes_threshold(c_fn: float, c_fp: float) -> float:
    """Cost-optimal threshold for calibrated probabilities: flag if p > c_fp / (c_fp + c_fn)."""
    return c_fp / (c_fp + c_fn)


# ======================================================================================
# Experiment runner (ablation ladder)
# ======================================================================================
ABLATIONS: tuple[tuple[str, str, str], ...] = (
    ("naive_moid_h", "naive", "lgbm_logloss"),
    ("static_orbit_only", "static_orbit", "lgbm_logloss"),
    ("static_leakfree", "static", "lgbm_logloss"),
    ("static_dyn", "static+dyn", "lgbm_logloss"),
    ("static_dyn_unc", "static+dyn+unc", "lgbm_logloss"),
    ("static_dyn_unc_focal", "static+dyn+unc", "lgbm_focal"),
    ("dual_resmlp_focal", "static+dyn+unc", "mlp_focal"),
    ("proposed_ensemble", "static+dyn+unc", "ensemble"),
)


@dataclass
class ExperimentConfig:
    focal: AsymFocalConfig = field(default_factory=AsymFocalConfig)
    lgbm: LGBMConfig = field(default_factory=LGBMConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    alpha_benign: float = 0.10
    alpha_hazardous: float = 0.01
    c_fn: float = 100.0
    c_fp: float = 1.0
    # (gamma, lambda_FN) candidates, chosen on the VALIDATION split by recall@1%FPR (ties: PR-AUC).
    # (0, 1) is exactly log-loss, so selection can reject the asymmetric loss if it does not help.
    focal_grid: tuple[tuple[float, float], ...] = ((0.0, 1.0), (1.0, 5.0), (2.0, 10.0), (1.0, 20.0))


def run_ablations(frames: dict[str, FeatureFrame], y: pd.Series, splits: dict[str, np.ndarray],
                  cfg: ExperimentConfig | None = None,
                  only: tuple[str, ...] | None = None) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Train every ablation row; return long-format predictions and per-row artefacts.

    Predictions frame: one row per (model, object) with split, y, raw logit, calibrated p,
    conformal set code.
    """
    cfg = cfg or ExperimentConfig()
    yv = y.to_numpy().astype(float)
    tr, va, ca, te = (splits[k] for k in ("train", "val", "cal", "test"))
    split_name = np.empty(len(y), dtype=object)
    for k, idx in splits.items():
        split_name[idx] = k
    preds, arte, trained = [], {}, {}
    for name, fset, kind in ABLATIONS:
        if only and name not in only:
            continue
        ff = frames[fset]
        F = ff.F.loc[y.index]
        Ftr, Fva = F.iloc[tr], F.iloc[va]
        chosen, grid_scores = None, None
        if kind == "lgbm_logloss":
            m = LGBMModel(LGBMConfig(**{**cfg.lgbm.__dict__, "objective": "binary"})).fit(Ftr, yv[tr], Fva, yv[va])
        elif kind in ("lgbm_focal", "mlp_focal"):
            from sklearn.metrics import average_precision_score
            best, grid_scores = None, {}
            for gamma, lfn in cfg.focal_grid:
                fc = AsymFocalConfig(gamma, lfn, cfg.focal.lambda_fp)
                if kind == "lgbm_focal":
                    cand = LGBMModel(LGBMConfig(**{**cfg.lgbm.__dict__, "objective": "asym_focal", "focal": fc}))
                else:
                    cand = DualBranchMLP(ff.branch1, ff.branch2, MLPConfig(**{**cfg.mlp.__dict__, "focal": fc}))
                cand.fit(Ftr, yv[tr], Fva, yv[va])
                zv = cand.predict_logit(Fva)
                score = (recall_at_fpr(yv[va], zv, 0.01), average_precision_score(yv[va], zv))
                grid_scores[f"g{gamma}_l{lfn}"] = score
                if best is None or score > best[0]:
                    best = (score, cand, (gamma, lfn))
            _, m, chosen = best
        elif kind == "ensemble":
            m = LogitEnsemble([trained["static_dyn_unc_focal"], trained["dual_resmlp_focal"]])
        else:
            raise ValueError(kind)
        trained[name] = m
        z = m.predict_logit(F)
        cal = PlattCalibrator().fit(z[va], yv[va])
        p = cal(z)
        conf = MondrianConformal(cfg.alpha_benign, cfg.alpha_hazardous).fit(p[ca], yv[ca])
        sets = conf.predict_sets(p)
        preds.append(pd.DataFrame({"model": name, "pdes": y.index, "split": split_name, "y": yv,
                                   "logit": z, "p": p, "conf_set": sets}))
        arte[name] = {"feature_set": fset, "kind": kind, "n_features": F.shape[1], "platt": (cal.a, cal.b),
                      "conformal_q": conf.q, "n_cal": conf.n_cal,
                      "best_iter": getattr(getattr(m, "booster", None), "best_iteration", None),
                      "epochs": len(getattr(m, "history", [])) or None,
                      "focal_chosen": chosen, "focal_grid_val": grid_scores}
        log.info("trained %s (%s, %d features)", name, kind, F.shape[1])
    return pd.concat(preds, ignore_index=True), arte
