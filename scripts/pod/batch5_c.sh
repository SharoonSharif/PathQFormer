#!/usr/bin/env bash
# Batch 5C (2026-09-12, after the OOM night). Resumes: finished folds/runs are skipped.
#   1. SurvPath e10 seeds 1-2 remainder, five cohorts -> /workspace/outputs_ablate
#   2. 20-epoch budget seed 0, five cohorts (batch 4 share) -> /workspace/outputs_e20
# Removes its pod at the end. Launch detached:
#   setsid bash scripts/pod/batch5_c.sh > /workspace/logs/pod_batch5_c.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CACHE_COHORTS="BLCA STAD HNSC COADREAD BRCA" PYTHONUTF8=1
echo "[batch5c] $(date '+%F %T') SurvPath e10 seeds 1-2 remainder"
KEEP_POD=1 OUT=/workspace/outputs_ablate COHORTS="BLCA STAD HNSC COADREAD BRCA" SEEDS="1 2" \
  CONFIGS="configs/protocol_fixed/survpath_e10.yaml" bash scripts/pod/run_queue_pod.sh
echo "[batch5c] $(date '+%F %T') 20-epoch budget, seed 0 (remainder) then seeds 1-2 for COADREAD"
KEEP_POD=1 OUT=/workspace/outputs_e20 COHORTS="BLCA STAD HNSC COADREAD BRCA" SEEDS="0" \
  CONFIGS="configs/protocol_fixed/pathq_fast_e20.yaml configs/protocol_fixed/pathq_fast_e20_aux.yaml configs/protocol_fixed/survpath_e20.yaml" bash scripts/pod/run_queue_pod.sh
OUT=/workspace/outputs_e20 COHORTS="COADREAD" SEEDS="1 2" \
  CONFIGS="configs/protocol_fixed/pathq_fast_e20.yaml configs/protocol_fixed/pathq_fast_e20_aux.yaml configs/protocol_fixed/survpath_e20.yaml" bash scripts/pod/run_queue_pod.sh
echo "[batch5c] $(date '+%F %T') done"
