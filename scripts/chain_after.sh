#!/usr/bin/env bash
# Wait until a running queue (scripts/run_experiments.sh writing logs/<queue>.out) has finished
# N runs, then start another queue. Lets a second batch of experiments be scheduled without
# touching the one that is running.
#
#   bash scripts/chain_after.sh logs/queue.out 4 configs/baselines/blca_survpath.yaml configs/baselines/blca_abmil.yaml ...
set -uo pipefail
cd "$(dirname "$0")/.."
QUEUE_LOG="$1"; N_DONE="$2"; shift 2

echo "[$(date '+%F %T')] waiting for $N_DONE END lines in $QUEUE_LOG before running: $*"
until [ "$(grep -c '\] END' "$QUEUE_LOG" 2>/dev/null || echo 0)" -ge "$N_DONE" ]; do
  sleep 120
done
echo "[$(date '+%F %T')] previous queue finished; starting chained queue"
bash scripts/run_experiments.sh "$@"
