#!/usr/bin/env bash
# Batch 5A (2026-09-12, after the OOM night). Resumes: finished folds/runs are skipped.
#   1. fixed-budget seed 0 remainder (COADREAD, BRCA; earlier cohorts skip) -> /workspace/outputs_v2
#   2. 20-epoch budget seeds 1-2, five cohorts (batch 4 share) -> /workspace/outputs_e20
# Removes its pod at the end. Launch detached:
#   setsid bash scripts/pod/batch5_a.sh > /workspace/logs/pod_batch5_a.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CACHE_COHORTS="BLCA STAD HNSC COADREAD BRCA" PYTHONUTF8=1   # cache is capped at 45 % of the container limit
echo "[batch5a] $(date '+%F %T') fixed-budget seed 0 remainder"
KEEP_POD=1 OUT=/workspace/outputs_v2 COHORTS="BLCA STAD HNSC COADREAD BRCA" SEEDS="0" \
  CONFIGS="configs/protocol_fixed/pathq_fast_e10.yaml configs/protocol_fixed/survpath_e10.yaml" bash scripts/pod/run_queue_pod.sh
echo "[batch5a] $(date '+%F %T') 20-epoch budget, seeds 1-2"
OUT=/workspace/outputs_e20 COHORTS="BLCA STAD HNSC COADREAD BRCA" SEEDS="1 2" \
  CONFIGS="configs/protocol_fixed/pathq_fast_e20.yaml configs/protocol_fixed/pathq_fast_e20_aux.yaml configs/protocol_fixed/survpath_e20.yaml" bash scripts/pod/run_queue_pod.sh
echo "[batch5a] $(date '+%F %T') done"
