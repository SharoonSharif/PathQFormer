#!/usr/bin/env bash
# Pod batch 6 (2026-09-14): the experiments that complete the paper, cheapest and most valuable first.
#   1. efficiency benchmark (minutes)
#   2. single-modality baselines under the fixed 20-epoch budget, five cohorts (ABMIL, SNN, MLP; ~2 h)
#   3. mean-imputation missing-modality evaluation of the finished SurvPath and PathQ-Former 20-epoch runs (eval only)
#   4. single-modality PathQ-Former at 20 epochs on five cohorts + late-fusion baseline per cohort (~8 h)
#   5. second endpoint: overall survival (OS) for PathQ-Former+aux and SurvPath, five cohorts, seed 0 (~10 h)
# Everything resumes. Removes its pod at the end. Launch detached:
#   setsid bash scripts/pod/batch6.sh > /workspace/logs/pod_batch6.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CACHE_COHORTS="BLCA STAD HNSC COADREAD BRCA" PYTHONUTF8=1
E20=/workspace/outputs_e20
COH="BLCA STAD HNSC COADREAD BRCA"

echo "[batch6] $(date '+%F %T') 1. efficiency"
python3 scripts/efficiency.py --config configs/protocol_fixed/pathq_fast_e20_aux.yaml --baseline configs/protocol_fixed/survpath_e20.yaml \
  --n 40 --out /workspace/outputs_e20/efficiency.md 2>&1 | tail -n 8 || echo "[batch6] efficiency failed"

echo "[batch6] $(date '+%F %T') 2. single-modality baselines, 20 epochs"
KEEP_POD=1 OUT=$E20 COHORTS="$COH" SEEDS="0" \
  CONFIGS="configs/protocol_fixed/abmil_e20.yaml configs/protocol_fixed/snn_e20.yaml configs/protocol_fixed/mlp_omics_e20.yaml" bash scripts/pod/run_queue_pod.sh

echo "[batch6] $(date '+%F %T') 3. mean-imputation missing-modality evaluation"
for c in blca stad hnsc coadread brca; do
  python3 scripts/eval_missing_impute.py "$E20/survpath_e20/$c" "$E20/pathq_fast_e20_aux/$c" "$E20/pathq_fast_e20/$c" 2>&1 | grep -E "^==|fold" || true
done

echo "[batch6] $(date '+%F %T') 4. single-modality PathQ-Former 20 ep + late fusion"
KEEP_POD=1 OUT=$E20 COHORTS="$COH" SEEDS="0" \
  CONFIGS="configs/protocol_fixed/pathq_e20_wsi_only.yaml configs/protocol_fixed/pathq_e20_genomic_only.yaml" bash scripts/pod/run_queue_pod.sh
for c in blca stad hnsc coadread brca; do
  python3 scripts/late_fusion.py "$E20/pathq_e20_wsi_only/$c" "$E20/pathq_e20_genomic_only/$c" --out "$E20/late_fusion_e20/$c" 2>&1 | grep -E "^fold|nan" || echo "[batch6] late fusion $c failed"
done

echo "[batch6] $(date '+%F %T') 5. overall-survival endpoint"
OUT=/workspace/outputs_os COHORTS="$COH" SEEDS="0" EXTRA="endpoint=os" \
  CONFIGS="configs/protocol_fixed/pathq_fast_e20_aux.yaml configs/protocol_fixed/survpath_e20.yaml" bash scripts/pod/run_queue_pod.sh
echo "[batch6] $(date '+%F %T') done"
