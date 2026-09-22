# GPU image for training and evaluation. Mount the UNI2-h features at /app/data/embeddings/uni2h.
#
#   docker build -t pathqformer .
#   docker run --gpus all -v /path/to/uni2h:/app/data/embeddings/uni2h -v $PWD/outputs_e20:/app/outputs_e20 pathqformer \
#       python -m src.training.train --config configs/protocol_fixed/pathq_fast_e20_aux.yaml --cancer_type blca
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONUTF8=1 PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt einops

COPY . .
# SurvPath supplies the 5-fold splits, clinical metadata, RNA matrices and pathway compositions.
RUN git clone --depth 1 https://github.com/mahmoodlab/SurvPath data/survpath_repo

# Random BLCA features so `python -m pytest -m "not slow"` and `--smoke` work inside the image.
RUN python scripts/make_dummy_embeddings.py --cancer blca

CMD ["bash"]
