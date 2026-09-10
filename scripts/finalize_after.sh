#!/usr/bin/env bash
# After the queues have finished, bring every run's metric blocks up to the current evaluation code
# (scripts/recompute_metrics.py) and build the consolidated comparison table.
#
#   bash scripts/finalize_after.sh logs/queue2.out 4        # wait for 4 END lines, then finalize outputs_v2
set -uo pipefail
cd "$(dirname "$0")/.."
QUEUE_LOG="${1:-logs/queue2.out}"; N_DONE="${2:-4}"; ROOT="${3:-outputs_v2}"
PY=".venv/Scripts/python.exe"; [ -x "$PY" ] || PY=".venv/bin/python"; [ -x "$PY" ] || PY="python"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8

echo "[$(date '+%F %T')] waiting for $N_DONE END lines in $QUEUE_LOG"
until [ "$(grep -c '\] END' "$QUEUE_LOG" 2>/dev/null || echo 0)" -ge "$N_DONE" ]; do sleep 120; done

echo "[$(date '+%F %T')] recomputing metrics for finished runs under $ROOT"
runs=()
for r in "$ROOT"/*/*/results.json; do [ -f "$r" ] && runs+=("$(dirname "$r")"); done
[ "${#runs[@]}" -gt 0 ] && "$PY" scripts/recompute_metrics.py "${runs[@]}"
"$PY" scripts/aggregate_results.py "$ROOT" --out results/summary_all.md --compare hybrid
for r in "${runs[@]}"; do "$PY" scripts/analyze_run.py "$r" >/dev/null 2>&1 || echo "analysis failed for $r"; done
echo "[$(date '+%F %T')] done: results/summary_all.md"
