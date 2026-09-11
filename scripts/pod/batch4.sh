#!/usr/bin/env bash
# Pod batch 4 (2026-09-11) - fourth pod on the shared volume; writes to /workspace/outputs_e20.
#   20-epoch fixed budget for PathQ-Former (fast schedule), PathQ-Former + auxiliary unimodal heads, and the
#   official SurvPath, seed 0 on all five cohorts, then seeds 1-2. The per-epoch histories also give the
#   10-epoch numbers post hoc (scripts/epoch_curves.py).
# Removes its pod at the end. Launch detached:
#   setsid bash scripts/pod/batch4.sh > /workspace/logs/pod_batch4.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # the repo this script lives in
export CACHE_COHORTS="BLCA STAD HNSC COADREAD BRCA"
export OUT=/workspace/outputs_e20 PYTHONUTF8=1
CFGS="configs/protocol_fixed/pathq_fast_e20.yaml configs/protocol_fixed/pathq_fast_e20_aux.yaml configs/protocol_fixed/survpath_e20.yaml"
SEEDS_TO_RUN="${1:-0 1 2}"   # e.g. `batch4.sh 0` on one pod and `batch4.sh "1 2"` on another
echo "[batch4] $(date '+%F %T') 20-epoch budget, five cohorts, seeds: $SEEDS_TO_RUN"
COHORTS="BLCA STAD HNSC COADREAD BRCA" SEEDS="$SEEDS_TO_RUN" CONFIGS="$CFGS" bash scripts/pod/run_queue_pod.sh
echo "[batch4] $(date '+%F %T') done"
