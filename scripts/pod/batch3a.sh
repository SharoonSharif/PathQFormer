#!/usr/bin/env bash
# Pod batch 3a (2026-09-11) - second pod on the shared volume; writes to /workspace/outputs_ablate.
#   Phase A: BLCA fusion ablations under the fixed-budget protocol (modality dropout, K, fusion depth,
#            single-modality models) + the late-fusion baseline.
#   Phase C: PathQ-Former (fast schedule) seeds 1-2 on all five cohorts (fixed budget).
# Removes its pod at the end. Launch detached:
#   setsid bash scripts/pod/batch3a.sh > /workspace/logs/pod_batch3a.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # the repo this script lives in
export CACHE_COHORTS="BLCA STAD HNSC COADREAD BRCA"   # 125 GB RAM per pod: cache every cohort as float16
export OUT=/workspace/outputs_ablate PYTHONUTF8=1
ABL="configs/ablation_fixed/pathq_fast_e10_md0.yaml configs/ablation_fixed/pathq_fast_e10_md30.yaml configs/ablation_fixed/pathq_fast_e10_md50.yaml \
configs/ablation_fixed/pathq_fast_e10_k16.yaml configs/ablation_fixed/pathq_fast_e10_k64.yaml \
configs/ablation_fixed/pathq_fast_e10_fusion1.yaml configs/ablation_fixed/pathq_fast_e10_fusion3.yaml \
configs/ablation_fixed/pathq_fast_e10_wsi_only.yaml configs/ablation_fixed/pathq_fast_e10_genomic_only.yaml \
configs/protocol_fixed/pathq_fast_e10.yaml"

echo "[batch3a] $(date '+%F %T') phase A: BLCA ablations (fixed budget, RAM cache)"
KEEP_POD=1 COHORTS="BLCA" CONFIGS="$ABL" bash scripts/pod/run_queue_pod.sh
python3 scripts/late_fusion.py "$OUT/pathq_fast_e10_wsi_only/blca" "$OUT/pathq_fast_e10_genomic_only/blca" \
  --out "$OUT/late_fusion_fast_e10/blca" || echo "[batch3a] late fusion failed"

echo "[batch3a] $(date '+%F %T') phase C: PathQ-Former seeds 1-2, five cohorts (fixed budget)"
COHORTS="BLCA STAD HNSC COADREAD BRCA" SEEDS="1 2" CONFIGS="configs/protocol_fixed/pathq_fast_e10.yaml" bash scripts/pod/run_queue_pod.sh
echo "[batch3a] $(date '+%F %T') done"
