# Polyp Segmentation (Kvasir-SEG) — Reproducible MLOps Baseline

> NOT FOR CLINICAL USE. Research/education project only.

[![CI](https://github.com/CommonGibbon/Polyp-Segmentation/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/CommonGibbon/Polyp-Segmentation/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12+-blue)
![Status](https://img.shields.io/badge/status-alpha-yellow)

Segment colorectal polyps in endoscopy images (binary mask) with a modern, reproducible MLOps stack: DVC for data/versioning, Great Expectations for data checks, Hydra configs, MLflow tracking/registry-ready, and containerized serving planned. Baseline model: U-Net (ResNet34 encoder), 256×256 input.

- Scope: single-class polyp segmentation on Kvasir-SEG or equivalent.
- Success criteria (initial): Dice ≥ 0.80 (val), one-command reproducibility (data→train→eval→package), <150 ms CPU ONNX @256×256, green CI, governance docs.
- Status: alpha; training skeleton + data split + MLflow logging scaffolding being finalized.

## Quickstart

Prereqs:
- Python 3.12+, Poetry, Git, DVC (with a configured remote optional), CUDA GPU optional (training supports GPU).

Setup:
```bash
# clone and install
git clone https://github.com/CommonGibbon/Polyp-Segmentation.git
cd REPO
poetry install --with dev
pre-commit install

Data:
bash

# Kvasir-SEG already tracked via DVC; pull data (requires access to remote or existing cache)
dvc pull

# create deterministic 70/15/15 splits (committed to repo)
poetry run python -m src.data.make_splits --root data/kvasir

Train (GPU if available):
bash

# 1-epoch smoke run; logs metrics/artifacts to MLflow
poetry run python -m src.train train.epochs=1 train.batch_size=8 train.num_workers=2

MLflow UI:
bash

# view runs, metrics, overlays, artifacts
poetry run mlflow ui --backend-store-uri ./mlruns
# open http://127.0.0.1:5000
```

Expected artifacts:

    overlays: artifacts/overlay_e0_*.png (qualitative)
    best checkpoint: artifacts/best.pt
    MLflow run: params/metrics + artifacts

Repository structure

configs/
  defaults.yaml
  data/kvasir.yaml
  model/unet_resnet34.yaml
  train/default.yaml
  eval/default.yaml
src/
  data/make_splits.py
  train.py
data/
  raw/Kvasir-SEG/
    images/  # DVC
    masks/   # DVC
  splits/  # generated CSVs (committed)
docs/
  assets/   # overlay screenshots (to be added)
  model_card.md (stub, planned)
  data_card.md  (stub, planned)

Tech stack

    Modeling: PyTorch, segmentation-models-pytorch (U-Net ResNet34), Albumentations
    Config: Hydra (deterministic seeds, reproducible runs)
    Metrics: torchmetrics (Dice/F1, IoU)
    Tracking: MLflow (params, metrics, artifacts; tags include git SHA)
    Data & validation: DVC + Great Expectations
    Serving (planned): ONNX export + FastAPI API + Streamlit demo
    CI/CD (planned): ruff, pytest, smoke train, Docker build

Results (alpha)

    Baseline training skeleton operational; qualitative overlays logged to MLflow.
    Target: Dice ≥ 0.80 on validation at 256×256 after full training/eval pass.

Governance

    License: Apache-2.0 (code). See LICENSE.
    Dataset: Kvasir-SEG — cite and follow dataset license. Include source and license in Data Card.
    Model/Usage: research only; NOT FOR CLINICAL USE.

Roadmap

    Eval script + summary report (plots, per-image distributions)
    ONNX export + onnxruntime validation (CPU latency target)
    FastAPI /predict + Streamlit demo
    CI workflow (ruff, pytest -q, smoke train, Docker)
    Model Card + Data Card; sample overlay screenshots in docs/assets

References

    Kvasir-SEG dataset: https://paperswithcode.com/dataset/kvasir-seg (add canonical source in Data Card)
    Spec document: docs/spec.pdf (add your spec file here and link it)
