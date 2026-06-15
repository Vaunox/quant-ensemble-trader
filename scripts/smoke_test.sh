#!/usr/bin/env bash
# Smoke tests for algo_v2 — fast (<1 min), no training, no network calls.
# Run after every restructure step to prove nothing broke.
#
# Usage:
#   scripts/smoke_test.sh
#   PYTHON=~/smoke_env/bin/python scripts/smoke_test.sh   # explicit interpreter
#
# Covers:
#   S1  compileall            — every .py parses
#   S2-S5  run_smoke.py       — imports, config contract, env step, logging/notify
#   S6  CLI --help            — argparse entry points load
#   S7  bash -n               — every shell script parses
set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
fails=0

echo "== S1: compileall =="
if "$PYTHON" -m compileall -q .; then echo "  PASS"; else echo "  FAIL"; fails=$((fails + 1)); fi

echo "== S2-S5: python smoke (run_smoke.py) =="
"$PYTHON" tests/smoke/run_smoke.py || fails=$((fails + 1))

echo "== S6: CLI --help =="
# Mix of not-yet-moved flat scripts and packaged `python -m` entry points.
for cmd in "-m algo_v2.evaluation.validate" "-m algo_v2.training.tune" "-m algo_v2.training.train_ensemble" "-m algo_v2.live.execute"; do
    if "$PYTHON" $cmd --help >/dev/null 2>&1; then
        echo "  PASS $cmd --help"
    else
        echo "  FAIL $cmd --help"; fails=$((fails + 1))
    fi
done

echo "== S7: bash -n on shell scripts =="
for s in scripts/*.sh *.sh; do
    [ -f "$s" ] || continue
    if bash -n "$s"; then echo "  PASS bash -n $s"; else echo "  FAIL bash -n $s"; fails=$((fails + 1)); fi
done

echo
if [ "$fails" -eq 0 ]; then
    echo "ALL SMOKE PASS"
else
    echo "SMOKE FAILURES: $fails"
    exit 1
fi
