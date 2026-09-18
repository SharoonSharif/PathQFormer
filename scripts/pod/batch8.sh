#!/usr/bin/env bash
# Pod batch 8 (2026-09-18): finish what the balance cut-off interrupted, then remove the pod.
#   1. OS-endpoint remainder (BRCA, seed 0; everything finished is skipped)
#   2. batch 7: single-modality baseline seeds 1-2 (20 ep) + clean efficiency numbers; removes the pod
#   setsid bash scripts/pod/batch8.sh > /workspace/logs/pod_batch8.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CACHE_COHORTS="BLCA STAD HNSC COADREAD BRCA" PYTHONUTF8=1
echo "[batch8] $(date '+%F %T') 1. OS endpoint remainder"
KEEP_POD=1 OUT=/workspace/outputs_os COHORTS="BLCA STAD HNSC COADREAD BRCA" SEEDS="0" EXTRA="endpoint=os" \
  CONFIGS="configs/protocol_fixed/pathq_fast_e20_aux.yaml configs/protocol_fixed/survpath_e20.yaml" bash scripts/pod/run_queue_pod.sh
echo "[batch8] $(date '+%F %T') 2. batch 7 (baseline seeds, efficiency) - removes the pod when done"
bash scripts/pod/batch7.sh
echo "[batch8] $(date '+%F %T') done"
