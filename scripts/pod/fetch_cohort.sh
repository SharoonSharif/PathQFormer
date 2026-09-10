#!/usr/bin/env bash
# Stream a TCGA cohort's UNI2-h embedding archive(s) from HuggingFace straight into the network volume
# (curl | tar), so no transient disk is needed for the 17-65 GB .tar.gz files.
#
#   bash scripts/pod/fetch_cohort.sh STAD            # -> /workspace/embeddings/STAD/**.h5
#
# Needs HF_TOKEN in the environment (source /workspace/.env). Idempotent: a cohort with a
# .complete marker is skipped. COADREAD = COAD + READ archives if both exist.
set -uo pipefail
COHORT="${1:?cohort, e.g. BLCA BRCA STAD HNSC COADREAD}"
ROOT="${EMBEDDINGS_ROOT:-/workspace/embeddings}"
REPO="https://huggingface.co/datasets/MahmoodLab/UNI2-h-features/resolve/main/TCGA"
: "${HF_TOKEN:?HF_TOKEN not set (source /workspace/.env)}"

case "$COHORT" in
  BLCA)     ARCHIVES=(TCGA-BLCA.tar.gz) ;;
  BRCA)     ARCHIVES=(TCGA-BRCA_IDC.tar.gz TCGA-BRCA_OTHERS.tar.gz) ;;
  STAD)     ARCHIVES=(TCGA-STAD.tar.gz) ;;
  HNSC)     ARCHIVES=(TCGA-HNSC.tar.gz) ;;
  COADREAD) ARCHIVES=(TCGA-COAD.tar.gz TCGA-READ.tar.gz) ;;
  *)        ARCHIVES=("TCGA-${COHORT}.tar.gz") ;;
esac

dest="$ROOT/$COHORT"
if [ -f "$dest/.complete" ]; then
  echo "[fetch] $COHORT already complete ($(find "$dest" -name '*.h5' | wc -l) slides)"; exit 0
fi
mkdir -p "$dest"
status=0
for a in "${ARCHIVES[@]}"; do
  echo "[fetch] $(date '+%T') $COHORT <- $a"
  if ! curl -sfL --retry 5 --retry-delay 10 -H "Authorization: Bearer $HF_TOKEN" "$REPO/$a" | tar -xz -C "$dest"; then
    echo "[fetch] FAILED: $a (missing archive or interrupted stream)"; status=1
  fi
done
n=$(find "$dest" -name '*.h5' | wc -l)
echo "[fetch] $(date '+%T') $COHORT: $n slides, $(du -sh "$dest" | cut -f1)"
if [ "$status" -eq 0 ] && [ "$n" -gt 0 ]; then touch "$dest/.complete"; fi
exit $status
