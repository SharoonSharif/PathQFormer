#!/usr/bin/env bash
# Pod batch 3 (2026-09-11) - runs on a second pod sharing the volume with batch 2; writes to /workspace/outputs_ablate.
#   Phase A: BLCA fusion ablations under the fixed-budget protocol (modality dropout, K, fusion depth,
#            single-modality models for the late-fusion baseline), then the late-fusion baseline itself.
#   Phase B: SurvPath seeds 1-2 on all five cohorts (fixed budget).
#   Phase C: PathQ-Former (fast schedule) seeds 1-2 on all five cohorts (fixed budget) - last, so it can be
#            re-pointed at a better configuration if phase A finds one.
# The pod removes itself after phase C. Launch detached:
#   setsid bash scripts/pod/batch3.sh > /workspace/logs/pod_batch3.out 2>&1 < /dev/null &
set -uo pipefail
cd /workspace/PathQFormer
export OUT=/workspace/outputs_ablate PYTHONUTF8=1
ABL="configs/ablation_fixed/pathq_fast_e10_md0.yaml configs/ablation_fixed/pathq_fast_e10_md30.yaml configs/ablation_fixed/pathq_fast_e10_md50.yaml \
configs/ablation_fixed/pathq_fast_e10_k16.yaml configs/ablation_fixed/pathq_fast_e10_k64.yaml \
configs/ablation_fixed/pathq_fast_e10_fusion1.yaml configs/ablation_fixed/pathq_fast_e10_fusion3.yaml \
configs/ablation_fixed/pathq_fast_e10_wsi_only.yaml configs/ablation_fixed/pathq_fast_e10_genomic_only.yaml \
configs/protocol_fixed/pathq_fast_e10.yaml"

echo "[batch3] $(date '+%F %T') phase A: BLCA ablations (fixed budget)"
KEEP_POD=1 COHORTS="BLCA" CONFIGS="$ABL" bash scripts/pod/run_queue_pod.sh
python3 scripts/late_fusion.py "$OUT/pathq_fast_e10_wsi_only/blca" "$OUT/pathq_fast_e10_genomic_only/blca" \
  --out "$OUT/late_fusion_fast_e10/blca" || echo "[batch3] late fusion failed"

echo "[batch3] $(date '+%F %T') phase B: SurvPath seeds 1-2, five cohorts (fixed budget)"
KEEP_POD=1 COHORTS="BLCA STAD HNSC COADREAD BRCA" SEEDS="1 2" CONFIGS="configs/protocol_fixed/survpath_e10.yaml" bash scripts/pod/run_queue_pod.sh

echo "[batch3] $(date '+%F %T') phase C: PathQ-Former seeds 1-2, five cohorts (fixed budget)"
COHORTS="BLCA STAD HNSC COADREAD BRCA" SEEDS="1 2" CONFIGS="configs/protocol_fixed/pathq_fast_e10.yaml" bash scripts/pod/run_queue_pod.sh
echo "[batch3] $(date '+%F %T') done"
