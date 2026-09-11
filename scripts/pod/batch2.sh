#!/usr/bin/env bash
# Pod batch 2 (2026-09-11).
#   Phase A: COADREAD under the v2 val-loss protocol for all five methods (completes Table 1 v1;
#            the first attempt crashed in the bootstrap CI - fixed in fd1d7ba).
#   Phase B: fixed-budget protocol (selection_metric: last, 10 epochs, final checkpoint) for
#            PathQ-Former (lr 1e-4, accumulation 2) and the official SurvPath on all five cohorts.
# The pod removes itself after phase B. Launch detached:
#   setsid bash scripts/pod/batch2.sh > /workspace/logs/pod_batch2.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # the repo this script lives in
export CACHE_COHORTS="BLCA STAD HNSC COADREAD BRCA"   # 125 GB RAM per pod: cache every cohort as float16
echo "[batch2] $(date '+%F %T') phase A: COADREAD, val-loss protocol, 5 methods"
KEEP_POD=1 COHORTS="COADREAD" bash scripts/pod/run_queue_pod.sh
echo "[batch2] $(date '+%F %T') phase B: fixed-budget protocol, 5 cohorts x {PathQ-Former fast, SurvPath}"
COHORTS="BLCA STAD HNSC COADREAD BRCA" \
CONFIGS="configs/protocol_fixed/pathq_fast_e10.yaml configs/protocol_fixed/survpath_e10.yaml" \
bash scripts/pod/run_queue_pod.sh
echo "[batch2] $(date '+%F %T') done"
