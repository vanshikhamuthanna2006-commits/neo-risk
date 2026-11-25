# Progress log (newest last, 2–3 lines per iteration)

- **i0 (setup):** branch loop/site-v2; verifier moved into tests/web/ (e2e.js, horizons_truth.json, check_data.py); baseline verify run recorded in loop/last_verify.txt.
- **i1 (R1) VERIFIED:** S/C priors now Thomas 2011 + Carry 2012 (C albedo 0.06->0.13 was the main-belt value); pinned by test. Retrain impact PR-AUC -0.001/+0.002 (results/prior_update_impact.json); full 5-seed re-evaluation running. Lesson: never run the verifier alongside model jobs; verify.sh now pauses them.
