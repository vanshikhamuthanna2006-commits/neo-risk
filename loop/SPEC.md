# Loop spec: NEO Atlas v2

```
GOAL:          Every item in loop/TODO.md is done, and loop/verify.sh exits 0 on two
               consecutive clean runs.
TRIGGER:       The previous iteration ends (run inside one supervised Claude Code session).
UNIT OF WORK:  The top unchecked item in loop/TODO.md, and only that item.
CONTEXT IN:    loop/SPEC.md, loop/TODO.md, loop/progress.md, the last verify output
               (loop/last_verify.txt), and the files the item names.
VERIFIER:      loop/verify.sh. It runs:
                 1. pytest: the model and pipeline unit tests;
                 2. tests/web/check_data.py: integrity of the site's data files;
                 3. tests/web/e2e.js: a headless browser that checks positions against
                    JPL Horizons, checks every interaction on both pages and requires zero
                    page errors.
               Each item adds its own checks to tests/web/ BEFORE implementing it. An
               existing check may be tightened but never loosened or deleted.
STATE OUT:     One commit per verified item on branch loop/site-v2, plus a line in
               loop/progress.md.
STOP WHEN:     Success: TODO.md is all checked and verify passes twice.
               Safety: 12 iterations; the same failing check 3 times in a row; or the
               session's context budget runs low (write progress.md and stop).
ESCALATE WHEN: An item needs a secret, a paid service, a destructive action, or a
               judgement of visual taste. Visual taste is reviewed by the user from
               screenshots in loop/shots/.
GUARDRAILS:    Work only on branch loop/site-v2. Merge into main (the live site) only
               after the final double verify. No force-push. No changes to model
               results unless an item is explicitly about them and documents the impact.
```

## Non-machine-checkable quality bar (for the human review at the end)
- Professional: every number on screen is sourced, labelled with units, and caveated where approximate.
- Visual: one coherent system; no overlapping labels; legible on a phone; dark theme throughout the atlas.
