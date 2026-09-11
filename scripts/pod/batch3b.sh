#!/usr/bin/env bash
# Pod batch 3b (2026-09-11) - third pod on the shared volume; writes to /workspace/outputs_ablate.
#   Phase B: official SurvPath seeds 1-2 on all five cohorts under the fixed-budget protocol.
# Removes its pod at the end. Launch detached:
#   setsid bash scripts/pod/batch3b.sh > /workspace/logs/pod_batch3b.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # the repo this script lives in
export CACHE_COHORTS="BLCA STAD HNSC COADREAD BRCA"   # 125 GB RAM per pod: cache every cohort as float16
export OUT=/workspace/outputs_ablate PYTHONUTF8=1
echo "[batch3b] $(date '+%F %T') phase B: SurvPath seeds 1-2, five cohorts (fixed budget, RAM cache)"
COHORTS="BLCA STAD HNSC COADREAD BRCA" SEEDS="1 2" CONFIGS="configs/protocol_fixed/survpath_e10.yaml" bash scripts/pod/run_queue_pod.sh
echo "[batch3b] $(date '+%F %T') done"
