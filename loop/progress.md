# Progress log (newest last, 2–3 lines per iteration)

- **i0 (setup):** branch loop/site-v2; verifier moved into tests/web/ (e2e.js, horizons_truth.json, check_data.py); baseline verify run recorded in loop/last_verify.txt.
- **i1 (R1) VERIFIED:** S/C priors now Thomas 2011 + Carry 2012 (C albedo 0.06->0.13 was the main-belt value); pinned by test. Retrain impact PR-AUC -0.001/+0.002 (results/prior_update_impact.json); full 5-seed re-evaluation running. Lesson: never run the verifier alongside model jobs; verify.sh now pauses them.
- **i2 (R2) VERIFIED (87/87):** Research-gaps tab, 11 gaps (5 addressed, 4 partial, 2 open), 34 numbers tagged and checked against summary.json. First run crashed on test step ordering (tab left open), fixed without loosening. Next: faster e2e (smaller viewport, ready-signal waits), then W1.
- **i3 (W1) VERIFIED (91/91):** TreeSHAP drivers per object from the fold model that scored it; export asserts exact sum to logit; check_data re-derives every published p from logit + fold Platt (max err < 1e-4). Shown in atlas card (follows colour mode) and explorer. e2e now 960x600 + ready signal: ~2 min instead of ~7.
