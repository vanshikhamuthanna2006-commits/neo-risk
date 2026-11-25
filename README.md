# neo-risk: Uncertainty-aware, physics-informed NEO impact-risk classifier

Stage 1 (ingestion + anti-leakage gate) is implemented in `src/data_pipeline.py`.

```bash
pip install -r requirements.txt
python -m pytest                      # offline, deterministic (real JPL fixtures)
python -m pytest -m network           # live JPL smoke test
python scripts/build_dataset.py       # full snapshot -> data/processed/, results/stage1_leakage_probe.json
```
Data snapshots change daily; cite `provenance.json` (`snapshot_utc`, payload SHA-256) in the paper.

## Stage 2: N-body dynamics (`src/dynamics.py`)
```bash
python scripts/run_dynamics.py 50     # full catalogue, ~5 min on 4 cores -> data/processed/pha_default/dyn_df_50yr.parquet
python scripts/run_dynamics.py 100    # 100-year variant (adds 100-yr drift features)
```
Requires the Stage 1 build (`scripts/build_dataset.py`) first. Planet ICs: Standish (JPL) approximate
elements; MERCURIUS (dt = 1 d) for the bulk, per-object IAS15 when perihelion passage is under-resolved.

## Stage 3: uncertainty & physical priors (`src/uncertainty.py`)
```bash
python scripts/fetch_covariances.py   # resumable; seeded random order (any prefix = uniform sample); ~2.5 obj/s
python scripts/run_uncertainty.py     # clone features on the fetched prefix + priors -> data/processed/pha_default/unc_df.parquet
```
K=32 antithetic clones from the full cometary covariance at its own epoch, re-centred on the
Stage-2 nominal at t0. Physical priors use taxonomy only (never H); mass/KE only for measured diameters.

## Stage 4: models (`src/model.py`)
```bash
python scripts/run_models.py pha      # Mode A ablation ladder (stratified + temporal splits)
python scripts/run_models.py sentry   # Mode B (Sentry risk-list membership)
```
Outputs `results/predictions_<target>_<scheme>.parquet` (per-object logits, calibrated p, conformal set).

## Stage 5: evaluation (`src/evaluate.py`)
```bash
python scripts/run_evaluation.py 5 500   # 5 seeds x {pha, sentry} x {stratified, temporal}; 500 bootstrap reps
```
Outputs: `results/ablation_table.tex` (main paper table: Mode A, stratified), `results/tables/*.tex|csv`
(all targets/splits, short-arc strata, conformal trade-offs), `results/figures/*.pdf|png`, `results/stage5_summary.json`.
CIs: paired stratified bootstrap of each seed's test set, averaged replicate-wise across seeds.

## Full reproduction order
`build_dataset.py` -> `run_dynamics.py 50` -> `fetch_covariances.py` -> `run_uncertainty.py` -> `run_evaluation.py`
 -> `export_web.py` -> `export_orbits.py` -> `export_clouds.py` -> `export_closeapp.py` (interactive site data)

## Interactive site (`docs/`, GitHub Pages)
A static site with no server, in two pages.

**`docs/index.html`: NEO Atlas (3-D, WebGL via three.js 0.160.0).**
- **Live motion:** all 43,160 asteroids move in real time, or at up to a month per second. Their two-body Kepler motion is solved on the GPU from JPL osculating elements.
- **Planets:** positions come from Standish's approximate elements.
- **Colour:** each asteroid is coloured by the full model's out-of-fold risk score.
- **Interaction:** click or search for any asteroid to see its orbit and risk card.
- **Close encounters:** six real flybys from 2027 to 2075 (Apophis, 2001 WN5, 1999 AN10, 2024 YR4, 2012 UE34, 2005 YU55). Each replays JPL Horizons geocentric trajectories with a to-scale Earth, the Moon and the geostationary belt.
- **Data:** `docs/data/orbits.bin` and `docs/data/flybys.json`, from `python scripts/export_orbits.py`. This needs `ssd.jpl.nasa.gov` (Horizons) and `ssd-api.jpl.nasa.gov`.
- **Also in the atlas:**
  - "why this score" drivers (`explain/`, in 1,000-object chunks loaded on demand);
  - uncertainty clouds of covariance clones for 605 high-risk and featured objects (`clouds.bin`, from `export_clouds.py`);
  - JPL close approaches for the next 24 months (`closeapp.json`, from `export_closeapp.py`; re-run it to refresh);
  - discovery-history mode, an Earth-centred view, a 1950–2100 timeline, shareable links, keyboard shortcuts (`?`) and a guided tour.
- **Verification:** run `loop/verify.sh`. It runs pytest, the data checks (`tests/web/check_data.py`), the browser checks (`tests/web/e2e.js`, which include accuracy against JPL Horizons) and the simulated usability check (`tests/web/usability.js --assert`: 13 device profiles, slow-CPU limits and five scripted personas).

**`docs/explorer.html`: data explorer.** It reads `docs/data/objects.json` and `docs/data/summary.json`.

- **Explore objects:** search any of the 43,160 NEOs and see its out-of-fold risk from the leak-free baseline and the full model, its conformal set, its dynamics and clone features, and the audit-only MOID, H, PHA and Sentry values.
- **Decision trade-off:** a cost-ratio slider (Bayes threshold) and α sliders that recompute the conformal sets exactly from exported conformal p-values.
- **Results:** out-of-fold PR curves, Δ PR-AUC with 95% CIs from the official 5-seed test splits, and feature importance.
- **Leakage audit:** the naive (MOID + H) model against the leak-free models, plus the probe R² values.

Regenerate the data with `python scripts/export_web.py`. This runs 5-fold CV with train/val/cal inside each fold, so every object is scored by a model that never saw it.

To preview locally, run `python -m http.server -d docs 8000` and open http://localhost:8000. Opening the file directly does not work, because browsers block `fetch` from `file://`.

To publish, push the repo to GitHub and go to Settings → Pages → Source: *Deploy from a branch*, branch `main`, folder `/docs`.
