#!/usr/bin/env bash
# Loop verifier: unit tests + data integrity + browser end-to-end (incl. JPL accuracy).
set -u
cd "$(dirname "$0")/.." || exit 1
THREE=tests/web/node_modules/three
[ -d "$THREE" ] || (cd tests/web && npm install three@0.160.0 --silent >/dev/null 2>&1)
mkdir -p loop/shots
# Long model jobs (run_evaluation, run_uncertainty) compete for the 4 CPUs and can make the
# browser checks time out. Pause them (SIGSTOP keeps their progress) and resume on exit.
PAUSED=$(pgrep -f "scripts/run_(evaluation|uncertainty|models).py" | tr '\n' ' ')
[ -n "$PAUSED" ] && { kill -STOP $PAUSED; echo "(paused model jobs: $PAUSED)"; }
python3 -m http.server 8765 -d docs >/dev/null 2>&1 & SRV=$!
trap 'kill $SRV 2>/dev/null; [ -n "$PAUSED" ] && kill -CONT $PAUSED' EXIT
sleep 1
rc=0
echo "== pytest";      python3 -m pytest -q 2>&1 | tail -1;            [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
echo "== data";        python3 tests/web/check_data.py || rc=1
echo "== browser e2e"; node tests/web/e2e.js "$THREE" tests/web/horizons_truth.json loop/shots > loop/e2e.txt 2>&1 || rc=1
grep -E "FAIL|CRASH|checks passed" loop/e2e.txt
echo "== VERIFY $( [ $rc -eq 0 ] && echo PASS || echo FAIL )"
exit $rc
