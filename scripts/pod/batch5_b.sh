#!/usr/bin/env bash
# Batch 5B (2026-09-12, after the OOM night). Resumes: finished folds/runs are skipped.
#   1. BLCA fusion ablations remainder + late-fusion baseline -> /workspace/outputs_ablate
#   2. PathQ-Former fast e10 seeds 1-2, five cohorts -> /workspace/outputs_ablate
# Removes its pod at the end. Launch detached:
#   setsid bash scripts/pod/batch5_b.sh > /workspace/logs/pod_batch5_b.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CACHE_COHORTS="BLCA STAD HNSC COADREAD BRCA" OUT=/workspace/outputs_ablate PYTHONUTF8=1
ABL="configs/ablation_fixed/pathq_fast_e10_md0.yaml configs/ablation_fixed/pathq_fast_e10_md30.yaml configs/ablation_fixed/pathq_fast_e10_md50.yaml \
configs/ablation_fixed/pathq_fast_e10_k16.yaml configs/ablation_fixed/pathq_fast_e10_k64.yaml \
configs/ablation_fixed/pathq_fast_e10_fusion1.yaml configs/ablation_fixed/pathq_fast_e10_fusion3.yaml \
configs/ablation_fixed/pathq_fast_e10_wsi_only.yaml configs/ablation_fixed/pathq_fast_e10_genomic_only.yaml \
configs/protocol_fixed/pathq_fast_e10.yaml"
echo "[batch5b] $(date '+%F %T') BLCA ablations remainder"
KEEP_POD=1 COHORTS="BLCA" CONFIGS="$ABL" bash scripts/pod/run_queue_pod.sh
python3 scripts/late_fusion.py "$OUT/pathq_fast_e10_wsi_only/blca" "$OUT/pathq_fast_e10_genomic_only/blca" \
  --out "$OUT/late_fusion_fast_e10/blca" || echo "[batch5b] late fusion failed"
echo "[batch5b] $(date '+%F %T') PathQ-Former e10 seeds 1-2, five cohorts"
COHORTS="BLCA STAD HNSC COADREAD BRCA" SEEDS="1 2" CONFIGS="configs/protocol_fixed/pathq_fast_e10.yaml" bash scripts/pod/run_queue_pod.sh
echo "[batch5b] $(date '+%F %T') done"
