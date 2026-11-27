"""Generate every number and table used in the paper from the result files.

python scripts/paper_numbers.py   ->  paper/generated/numbers.tex, paper/generated/tab_*.tex

Nothing in paper/main.tex is typed by hand: it uses the macros defined here, so the text
always matches results/ (re-run after any re-evaluation, then recompile).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

root = Path(__file__).resolve().parents[1]
res, proc, gen = root / "results", root / "data" / "processed" / "pha_default", root / "paper" / "generated"
gen.mkdir(parents=True, exist_ok=True)
M: dict[str, str] = {}


def tab(mode, split, stratum="all"):
    return pd.read_csv(res / "tables" / f"summary_{mode}_{split}_{stratum}.csv", index_col=0)


def f3(x):  # signed 3-decimal
    return f"{x:+.3f}".replace("-", "$-$")


def ci(t, m):
    r = t.loc[m]; return f"{f3(r.d_pr_auc)} [{f3(r.d_pr_auc_lo)}, {f3(r.d_pr_auc_hi)}]"


# ------------------------------------------------------------------ data
prov = json.load(open(proc / "provenance.json"))
cb = prov["class_balance"]; n = cb["0"] + cb["1"]
audit = pd.read_parquet(proc / "audit_df.parquet"); meta = pd.read_parquet(proc / "meta_df.parquet")
X = pd.read_parquet(proc / "X.parquet")
M.update(NObj=f"{n:,}", NPHA=f"{cb['1']:,}", PrevPHA=f"{100 * cb['1'] / n:.1f}",
         NSentry=f"{int(audit.in_sentry.sum()):,}", PrevSentry=f"{100 * audit.in_sentry.mean():.1f}",
         NShortArc=f"{int(meta.short_arc.sum()):,}", PctShortArc=f"{100 * meta.short_arc.mean():.0f}",
         NStaticFeat=str(X.shape[1]), SnapshotDate=prov["snapshot_utc"][:10],
         NPhaDisagree=str(prov.get("pha_flag_vs_recomputed_disagree", "")))
xi = json.load(open(root / "docs" / "data" / "explain" / "index.json"))
M["NFullFeat"] = str(len(xi["modes"]["sentry"]["names"]))
M["NDiam"] = f"{int(X['has_diameter'].sum()):,}"

# ------------------------------------------------------------------ leakage
p1 = json.load(open(res / "stage1_leakage_probe.json")); s3 = json.load(open(res / "stage3_report.json"))
M.update(ProbeHAll=f"{p1['ALL']['r2_H']:.2f}", ProbeMoidAll=f"{p1['ALL']['r2_moid']:.2f}",
         ProbeHObs=f"{p1['obs_meta']['r2_H']:.2f}", ProbeMoidOrbit=f"{p1['static_orbit']['r2_moid']:.2f}",
         ProbeHUnc=f"{s3['probe_X_plus_unc_subset']['r2_H']:.2f}", ProbeMoidUnc=f"{s3['probe_X_plus_unc_subset']['r2_moid']:.2f}",
         MaxRankCorr=f"{s3['max_rank_corr_vs_audit']:.3f}")
with open(gen / "tab_probe.tex", "w") as fh:
    rows = [("Orbital elements only", "static_orbit"), ("Observation metadata only", "obs_meta"), ("Physical (masked) only", "physical"),
            ("Orbit + observation", "orbit+obs"), ("All leak-free inputs", "ALL")]
    fh.write("\\begin{tabular}{lcc}\n\\toprule\nInput group & $R^2(H)$ & $R^2(\\mathrm{MOID})$\\\\\n\\midrule\n")
    for lab, k in rows:
        fh.write(f"{lab} & {p1[k]['r2_H']:.2f} & {p1[k]['r2_moid']:.2f}\\\\\n")
    fh.write(f"All + uncertainty features & {s3['probe_X_plus_unc_subset']['r2_H']:.2f} & {s3['probe_X_plus_unc_subset']['r2_moid']:.2f}\\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ dynamics validation
s2 = json.load(open(res / "stage2_report_50yr.json"))
M.update(NIAS=str(s2["n_ias15"]), NLost=str(s2["n_lost"]), EnergyErr=f"{s2['max_energy_err_bulk']:.1e}".replace("e-", "\\times10^{-") + "}",
         FracDminGeMoid=f"{100 * s2['frac_dmin_ge_moid']:.0f}", SpearmanDminMoid=f"{s2['spearman_dmin_vs_moid']:.2f}",
         NDminLtFive=f"{s2['n_dmin_lt_0.05']:,}", ProbeHDyn=f"{s2['probe_dyn_only']['r2_H']:.2f}",
         ProbeMoidDyn=f"{s2['probe_dyn_only']['r2_moid']:.2f}", ProbeHXDyn=f"{s2['probe_X_plus_dyn']['r2_H']:.2f}",
         ProbeMoidXDyn=f"{s2['probe_X_plus_dyn']['r2_moid']:.2f}")
M.update(UncCoverage=f"{100 * s3['has_unc_clones_in_selected']:.1f}", FracUnbound=f"{100 * s3['frac_unbound_mean']:.2f}")

# ------------------------------------------------------------------ headline results
S = {f"{m}_{s}_{st}": tab(m, s, st) for m in ("pha", "sentry") for s in ("stratified", "temporal") for st in ("all", "short_arc")}
for key, t in S.items():
    k = "".join(w.capitalize() for w in key.replace("short_arc", "short").split("_"))
    M[f"PR{k}Static"] = f"{t.loc['static_leakfree', 'pr_auc']:.3f}"
    M[f"PR{k}Full"] = f"{t.loc['static_dyn_unc', 'pr_auc']:.3f}"
    M[f"PR{k}Naive"] = f"{t.loc['naive_moid_h', 'pr_auc']:.3f}"
    M[f"D{k}Full"] = ci(t, "static_dyn_unc")
    M[f"D{k}Dyn"] = ci(t, "static_dyn")
    M[f"D{k}Focal"] = ci(t, "static_dyn_unc_focal")
    M[f"D{k}Ens"] = ci(t, "proposed_ensemble")
    M[f"Rec{k}Static"] = f"{100 * t.loc['static_leakfree', 'recall_at_fpr']:.1f}"
    M[f"Rec{k}Full"] = f"{100 * t.loc['static_dyn_unc', 'recall_at_fpr']:.1f}"
    M[f"Cost{k}Static"] = f"{t.loc['static_leakfree', 'cost_per_1k']:.0f}"
    M[f"Cost{k}Full"] = f"{t.loc['static_dyn_unc', 'cost_per_1k']:.0f}"
    M[f"Brier{k}Static"] = f"{t.loc['static_leakfree', 'brier']:.4f}"
    M[f"Brier{k}Full"] = f"{t.loc['static_dyn_unc', 'brier']:.4f}"

ORDER = [("static_orbit_only", "Orbital elements only"), ("static_dyn", "+ N-body dynamics"),
         ("static_dyn_unc", "+ dynamics + uncertainty"), ("static_dyn_unc_focal", "\\quad with asymmetric focal loss"),
         ("dual_resmlp_focal", "Dual-branch MLP (focal)"), ("proposed_ensemble", "Ensemble (LightGBM + MLP)")]


def sig(t, m):
    r = t.loc[m]; return r.d_pr_auc_lo > 0 or r.d_pr_auc_hi < 0


with open(gen / "tab_ablation.tex", "w") as fh:
    cols = [("sentry", "stratified"), ("sentry", "temporal"), ("pha", "stratified"), ("pha", "temporal")]
    fh.write("\\begin{tabular}{l" + "c" * 4 + "}\n\\toprule\n")
    fh.write(" & \\multicolumn{2}{c}{Mode B: Sentry list} & \\multicolumn{2}{c}{Mode A: PHA}\\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n")
    fh.write("Model & Stratified & Temporal & Stratified & Temporal\\\\\n\\midrule\n")
    fh.write("\\multicolumn{5}{l}{\\emph{PR-AUC of the reference and of the circular model}}\\\\\n")
    for m, lab in (("static_leakfree", "Leak-free static (reference)"), ("naive_moid_h", "Naive, uses MOID and $H$")):
        fh.write(lab + " & " + " & ".join(f"{S[f'{a}_{b}_all'].loc[m, 'pr_auc']:.3f}" for a, b in cols) + "\\\\\n")
    fh.write("\\midrule\n\\multicolumn{5}{l}{\\emph{$\\Delta$PR-AUC versus the reference [95\\% CI]}}\\\\\n")
    for m, lab in ORDER:
        cells = []
        for a, b in cols:
            t = S[f"{a}_{b}_all"]; r = t.loc[m]
            c = f"{f3(r.d_pr_auc)}\\,{{\\scriptsize[{f3(r.d_pr_auc_lo)}, {f3(r.d_pr_auc_hi)}]}}"
            cells.append(f"\\textbf{{{c}}}" if sig(t, m) and r.d_pr_auc > 0 else c)
        fh.write(lab + " & " + " & ".join(cells) + "\\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

with open(gen / "tab_shortarc.tex", "w") as fh:
    fh.write("\\begin{tabular}{lcccc}\n\\toprule\nSetting (short arcs, $<$30\\,d) & $n_{\\mathrm{test}}$ & Reference PR-AUC & Full PR-AUC & $\\Delta$PR-AUC [95\\% CI]\\\\\n\\midrule\n")
    summ = json.load(open(res / "stage5_summary.json"))
    for a, b, lab in (("sentry", "stratified", "Sentry, stratified"), ("sentry", "temporal", "Sentry, temporal"),
                      ("pha", "stratified", "PHA, stratified"), ("pha", "temporal", "PHA, temporal")):
        t = S[f"{a}_{b}_short_arc"]; cnt = summ[f"{a}_{b}"].get("short_arc_counts", {})
        fh.write(f"{lab} & {cnt.get('n', float('nan')):,.0f} ({cnt.get('pos', float('nan')):.0f}) & {t.loc['static_leakfree', 'pr_auc']:.3f} & {t.loc['static_dyn_unc', 'pr_auc']:.3f} & {ci(t, 'static_dyn_unc')}\\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ conformal (mean over seeds)
with open(gen / "tab_conformal.tex", "w") as fh:
    fh.write("\\begin{tabular}{llcccc}\n\\toprule\nTarget, split & Model & $\\alpha_H$ & Hazard coverage & Benign coverage & Follow-up rate\\\\\n\\midrule\n")
    for a, b, lab in (("sentry", "temporal", "Sentry, temporal"), ("pha", "temporal", "PHA, temporal")):
        c = pd.read_csv(res / "tables" / f"conformal_{a}_{b}.csv")
        for m, ml in (("static_leakfree", "Leak-free static"), ("static_dyn_unc", "+ dyn. + unc.")):
            for al in (0.05, 0.10):
                r = c[(c.model == m) & (np.isclose(c.alpha_haz, al))].mean(numeric_only=True)
                fh.write(f"{lab if (m == 'static_leakfree' and al == 0.05) else ''} & {ml if al == 0.05 else ''} & {al:.2f} & {r.haz_coverage:.3f} & {r.benign_coverage:.3f} & {r.follow_up_rate:.3f}\\\\\n")
        if a == "sentry":
            fh.write("\\midrule\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
    c = pd.read_csv(res / "tables" / "conformal_sentry_temporal.csv")
    for m, k in (("static_leakfree", "Static"), ("static_dyn_unc", "Full")):
        M[f"FollowSenTemp{k}"] = f"{100 * c[(c.model == m) & np.isclose(c.alpha_haz, 0.10)].follow_up_rate.mean():.1f}"
    c = pd.read_csv(res / "tables" / "conformal_pha_temporal.csv")
    M["NCalPosPhaTemp"] = f"{c.n_cal_pos.mean():.0f}"

# ------------------------------------------------------------------ out-of-fold release, prior sensitivity, risk context
sm = json.load(open(root / "docs" / "data" / "summary.json"))
for mm in ("pha", "sentry"):
    for mdl in ("naive", "static", "full"):
        M[f"OOF{mm.capitalize()}{mdl.capitalize()}"] = f"{sm[mm]['metrics'][mdl]['pr_auc']:.3f}"
pi = json.load(open(res / "prior_update_impact.json"))["modes"]
M.update(PriorDeltaPha=f3(pi["pha"]["pr_auc_delta"]), PriorDeltaSentry=f3(pi["sentry"]["pr_auc_delta"]))
sc = sm["sentry_context"]
M.update(TorinoMax=f"{sc['torino_max']:.0f}", NTorinoRated=f"{sc['n_torino_rated']:,}", PalermoMax=f"{sc['palermo_cum_max']:.2f}".replace("-", "$-$"),
         PalermoMaxName=sc["palermo_cum_max_name"])
o = json.load(open(root / "docs" / "data" / "objects.json"))
k = o["pdes"].index("101955")
M["BennuPSentry"] = f"{100 * o['p_sentry_full'][k]:.2f}"
imp = sm["sentry"]["importance"]["full"]
top = list(imp.items())[:3]
dynunc = sum(v for f_, v in imp.items() if f_.startswith(("unc_", "dyn_")))
M["TopFeatSentry"] = top[0][0].replace("_", "\\_"); M["TopFeatSentryShare"] = f"{100 * top[0][1]:.1f}"
M["DynUncShareTopFifteen"] = f"{100 * dynunc:.0f}"
fl = json.load(open(root / "docs" / "data" / "flybys.json"))["flybys"]
M["FlybyMaxErrKm"] = f"{max(abs(f_['horizons_min_km'] - f_['dist_km']) for f_ in fl):.0f}"

# ------------------------------------------------------------------ paired tests between variants
pw = json.load(open(res / "pairwise_tests.json"))
PW = [("static_dyn_unc__vs__static_dyn", "Uncertainty features beyond dynamics"),
      ("static_dyn_unc_focal__vs__static_dyn_unc", "Asymmetric focal loss vs log-loss"),
      ("proposed_ensemble__vs__static_dyn_unc", "Ensemble vs LightGBM"),
      ("dual_resmlp_focal__vs__static_dyn_unc", "Dual-branch MLP vs LightGBM")]
SET4 = [("sentry", "stratified"), ("sentry", "temporal"), ("pha", "stratified"), ("pha", "temporal")]
with open(gen / "tab_pairwise.tex", "w") as fh:
    fh.write("\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{Mode B: Sentry list} & \\multicolumn{2}{c}{Mode A: PHA}\\\\\n"
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nComparison (A $-$ B) & Stratified & Temporal & Stratified & Temporal\\\\\n\\midrule\n")
    for key, lab in PW:
        cells = []
        for a, b in SET4:
            r = pw[f"{a}_{b}_all"][key]
            c = f"{f3(r['pr_auc'])}\\,{{\\scriptsize[{f3(r['pr_auc_lo'])}, {f3(r['pr_auc_hi'])}]}}"
            sigp = r["pr_auc_lo"] > 0 or r["pr_auc_hi"] < 0
            cells.append(f"\\textbf{{{c}}}" if sigp else c)
        fh.write(lab + " & " + " & ".join(cells) + "\\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
nsig = {key: sum((pw[f"{a}_{b}_{st}"][key]["pr_auc_lo"] > 0) or (pw[f"{a}_{b}_{st}"][key]["pr_auc_hi"] < 0)
                 for a, b in SET4 for st in ("all", "short_arc")) for key, _ in PW}
M.update(NSigUncOverDyn=str(nsig[PW[0][0]]), NSigFocal=str(nsig[PW[1][0]]), NSigEns=str(nsig[PW[2][0]]), NSigMLP=str(nsig[PW[3][0]]))
r = pw["pha_temporal_all"][PW[0][0]]; M["UncOverDynPhaTemporal"] = f"{f3(r['pr_auc'])} [{f3(r['pr_auc_lo'])}, {f3(r['pr_auc_hi'])}]"
r = pw["sentry_temporal_all"][PW[0][0]]; M["UncOverDynSentryTemporal"] = f"{f3(r['pr_auc'])} [{f3(r['pr_auc_lo'])}, {f3(r['pr_auc_hi'])}]"

# ------------------------------------------------------------------ feature inventory
fg = prov["feature_groups"]
names = xi["modes"]["sentry"]["names"]
groups = [("Orbital elements and derived", fg["static_orbit"], "$a, e, i, q, Q$, angles as sin/cos, $T_J$"),
          ("Observation metadata", fg["obs_meta"], "data arc, $n_{\\mathrm{obs}}$, radar, condition code, RMS, formal $\\sigma$, staleness"),
          ("Physical (measured, masked)", fg["physical"], "diameter, rotation period, taxonomy"),
          ("Missingness masks", fg["masks"], "one indicator per field that can be missing"),
          ("N-body dynamics", [n_ for n_ in names if n_.startswith(("dyn_", "has_dyn"))], "$\\Delta a,\\Delta e,\\Delta i$ at 10/25/50 yr, $d_{\\min}$, encounters, divergence"),
          ("Covariance clones", [n_ for n_ in names if n_.startswith(("unc_", "has_unc"))], "clone $d_{\\min}$ spread and quantiles, $P(d_{\\min}<0.05,0.01$ au$)$, $\\sigma$ of elements"),
          ("Physical priors", [n_ for n_ in names if n_.startswith(("phys_", "has_phys"))], "albedo/density priors, $v_\\infty$, impact speed, mass, energy")]
with open(gen / "tab_features.tex", "w") as fh:
    fh.write("\\begin{tabular}{>{\\raggedright\\arraybackslash}p{0.30\\columnwidth}c>{\\raggedright\\arraybackslash}p{0.47\\columnwidth}}\n\\toprule\nGroup & $n$ & Examples\\\\\n\\midrule\n")
    for lab, cols_, ex in groups:
        fh.write(f"{lab} & {len(cols_)} & {ex}\\\\\n")
    fh.write(f"\\midrule\nTotal (full model) & {len(names)} & \\\\\n\\bottomrule\n\\end{{tabular}}\n")

# ------------------------------------------------------------------ hyperparameters (from the code's own defaults)
import sys
sys.path.insert(0, str(root))
from src.model import ExperimentConfig, LGBMConfig, MLPConfig
from src.dynamics import DynamicsConfig
from src.uncertainty import UncertaintyConfig
lg, ml, ex, dc, uc = LGBMConfig(), MLPConfig(), ExperimentConfig(), DynamicsConfig(), UncertaintyConfig()
chosen = {f"{a}_{b}": json.load(open(res / f"artefacts_{a}_{b}.json"))["static_dyn_unc_focal"]["focal_chosen"] for a, b in SET4}
n_plain = sum(1 for v in chosen.values() if v and v[0] == 0 and v[1] == 1)
M["NFocalPlain"] = str(n_plain)
grid = ", ".join(f"({g:g}, {l:g})" for g, l in ex.focal_grid)
with open(gen / "tab_hparams.tex", "w") as fh:
    rows = [("LightGBM", f"{lg.num_leaves} leaves, learning rate {lg.learning_rate}, up to {lg.n_estimators:,} trees, early stopping {lg.early_stopping}, "
                         f"min.\\ child {lg.min_child_samples}, feature fraction {lg.feature_fraction}, bagging {lg.bagging_fraction}"),
            ("Focal grid $(\\gamma,\\lambda_{\\mathrm{FN}})$", f"{grid}; chosen on validation by recall at 1\\% FPR"),
            ("Dual-branch MLP", f"width {ml.width}, embedding {ml.emb}, {ml.n_blocks} residual blocks, dropout {ml.dropout}, AdamW lr {ml.lr}, "
                                f"weight decay {ml.weight_decay}, batch {ml.batch_size}, $\\le${ml.max_epochs} epochs, patience {ml.patience}"),
            ("N-body", f"{dc.years:g} yr, step {dc.dt_days:g} d, MERCURIUS; IAS15 if perihelion passage $<{dc.peri_resolution_steps:g}$ steps; encounter threshold {dc.enc_threshold_au} au"),
            ("Clones", f"{uc.n_clones} antithetic draws per object"),
            ("Decision layer", f"$C_{{\\mathrm{{FN}}}}:C_{{\\mathrm{{FP}}}}={ex.c_fn:g}:{ex.c_fp:g}$; conformal $\\alpha_B={ex.alpha_benign}$, $\\alpha_H\\in\\{{0.01,0.05,0.10\\}}$"),
            ("Evaluation", "60/10/15/15 split, 5 seeds, 500 paired stratified bootstrap replicates per seed")]
    fh.write("\\begin{tabular}{>{\\raggedright\\arraybackslash}p{0.24\\columnwidth}>{\\raggedright\\arraybackslash}p{0.64\\columnwidth}}\n\\toprule\nComponent & Setting\\\\\n\\midrule\n")
    for a, b in rows:
        fh.write(f"{a} & {b}\\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ full metrics, reference vs full model
with open(gen / "tab_metrics.tex", "w") as fh:
    fh.write("\\begin{tabular}{llccccc}\n\\toprule\nSetting & Model & PR-AUC & ROC-AUC & Recall@1\\%FPR & Brier & Cost/1k\\\\\n\\midrule\n")
    for a, b in SET4:
        t = S[f"{a}_{b}_all"]
        for j, (m, lab) in enumerate((("static_leakfree", "Leak-free static"), ("static_dyn_unc", "+ dyn.\\ + unc."))):
            r = t.loc[m]
            fh.write(f"{(('Sentry' if a == 'sentry' else 'PHA') + ', ' + b) if j == 0 else ''} & {lab} & {r.pr_auc:.3f} & {r.roc_auc:.3f} & {100 * r.recall_at_fpr:.1f}\\% & {r.brier:.4f} & {r.cost_per_1k:.0f}\\\\\n")
        if (a, b) != SET4[-1]:
            fh.write("\\midrule\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
    triv = {f"{a}_{b}": json.load(open(res / "stage5_summary.json"))[f"{a}_{b}"]["trivial_cost_per_1k"] for a, b in SET4}
    M["TrivCostSentryTemporal"] = f"{triv['sentry_temporal']:.0f}"

# ------------------------------------------------------------------ runtimes
s5 = json.load(open(res / "stage5_summary.json"))
M.update(WallStageTwo=f"{s2['wall_s'] / 60:.1f}", WallStageThree=f"{s3['wall_s'] / 60:.0f}",
         WallEvalMin=f"{min(s5[f'{a}_{b}']['wall_s'] for a, b in SET4) / 60:.0f}",
         WallEvalMax=f"{max(s5[f'{a}_{b}']['wall_s'] for a, b in SET4) / 60:.0f}")

with open(gen / "numbers.tex", "w") as fh:
    fh.write("% Generated by scripts/paper_numbers.py from results/; do not edit by hand.\n")
    for key, v in sorted(M.items()):
        fh.write(f"\\newcommand{{\\{key}}}{{{v}}}\n")
bad = [k for k in M if not k.isalpha()]
assert not bad, f"macro names must be letters only: {bad}"
print(f"wrote {len(M)} macros and 9 tables to {gen}")
