#!/usr/bin/env bash
# One-time setup on a Runpod PyTorch pod (template runpod-torch-v280, network volume at /workspace).
# Expects: /workspace/PathQFormer (unzipped source) and /workspace/.env (HF_TOKEN, RUNPOD_API_KEY_PASTED).
#
#   bash /workspace/PathQFormer/scripts/pod/setup_pod.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # the repo this script lives in
export PIP_BREAK_SYSTEM_PACKAGES=1 PYTHONUTF8=1

echo "[setup] python: $(python3 --version) | torch: $(python3 -c 'import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")')"
pip install -q -r requirements.txt einops 2>&1 | tail -n 2

if [ ! -d data/survpath_repo/splits ]; then
  echo "[setup] cloning SurvPath (splits, metadata, RNA, pathway compositions)"
  rm -rf data/survpath_repo
  git clone -q --depth 1 https://github.com/mahmoodlab/SurvPath data/survpath_repo
fi

# runpodctl on the pod lets the queue remove its own pod when finished (cost guard)
if ! command -v runpodctl >/dev/null 2>&1; then
  curl -sSL https://cli.runpod.net | bash >/dev/null 2>&1 || echo "[setup] runpodctl install failed (self-termination disabled)"
fi

python3 -m pytest -q tests 2>&1 | tail -n 3
mkdir -p /workspace/embeddings /workspace/outputs_v2 /workspace/logs
echo "[setup] done"
