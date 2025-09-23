Project overview and goals

    Problem: Segment colorectal polyps in endoscopy images; output is a per‑pixel binary mask.
    Objective: Ship a reproducible, production‑grade ML system with proper MLOps: data/version control, experiment tracking, registry, CI/CD, containerized serving, and basic monitoring.
    Success criteria:
        Dice ≥ 0.80 on held‑out validation set.
        One‑command reproducibility (data → train → eval → package).
        CPU inference latency <150 ms at 256×256 via onnxruntime.
        Green CI: lint, tests, smoke train/eval, Docker build.
        Governance docs: Data Card, Model Card, “not for clinical use” disclaimer.

Scope and non‑goals

    In scope: Single‑class polyp segmentation on Kvasir‑SEG or equivalent; U‑Net baseline; modern MLOps (DVC, MLflow tracking/registry, CI/CD, Dockerized API, monitoring).
    Out of scope: Clinical validation/regulatory workflows, PHI/PII, multi‑class tasks, distributed training, GPU inference optimization.

Datasets and data management

    Dataset: Kvasir‑SEG (~1k images + masks); include license/source in a Data Card.
    Splits: 70/15/15 with fixed seed; commit split indices to repo. Prefer patient/video‑level grouping if metadata allows.
    Versioning: DVC for raw and processed data; optional remote (S3/MinIO/Drive).
    Validation: Great Expectations checks (image/mask pairing, shapes, dtypes, stats, class balance). Fail pipeline on critical rules.

Modeling and training

    Model: U‑Net (ResNet34 encoder) from segmentation_models_pytorch; binary sigmoid output.
    Loss/metrics: Dice + BCE (weighted). Track Dice/IoU via torchmetrics.
    Augmentations: Albumentations (flip/rotate/scale, brightness/contrast, elastic/affine, normalization). Persist config.
    Config: Hydra for data/model/train/eval; deterministic seeds.
    Tracking: MLflow for params/metrics/artifacts; tag runs with git SHA, DVC data rev, and model version/commit.

Evaluation and reporting

    Metrics: Dice (primary), IoU (secondary), PR metrics at 0.5 threshold, per‑image distributions.
    Qualitative: Save overlays (input + mask outline/fill) for random, best, and worst cases.
    Reports: Auto‑generate an eval summary with plots + montage; log as MLflow artifact.

Packaging and serving

    Export: ONNX for inference (static 256×256); validate with onnxruntime; keep PyTorch checkpoint for training.
    API: FastAPI /predict accepts image; returns mask (RLE or PNG) + latency_ms + model_version; consistent preprocessing shared with training.
    Demo UI: Streamlit drag‑and‑drop with overlay and threshold control; display model/version.
    Containers: Dockerfile for API; optional docker‑compose for API + Streamlit.

MLflow tracking and registry (local dev)

    Local server (SQLite backend + local artifacts):

bash

mlflow server \
  --backend-store-uri sqlite:///mlruns.db \
  --default-artifact-root file:$(pwd)/mlruns \
  --host 127.0.0.1 --port 5000

    Client env:

bash

export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
export MLFLOW_EXPERIMENT_NAME="polyp-seg"

    Logging guidelines:
        Log params: model/backbone, img_size, aug config, optimizer, seed.
        Log metrics: train/val Dice, IoU, loss curves; log per‑image metrics as artifacts.
        Log artifacts: overlays, confusion/PR curves, eval report, ONNX, conda/requirements, code snapshot/hash.
        Log model with mlflow.onnx flavor and mlflow.pyfunc wrapper; include model signature and example input.
    Registry workflow:
        Registered model name: polyp-seg-unet.
        On merge to main: auto‑register the best run’s model as a new version and transition to Staging.
        Manual approval to promote Staging → Production; keep at most one Production version; archive prior Production on promotion.

CI/CD and quality gates

    Pull Requests:
        Run pre‑commit (ruff/black/isort), pytest unit tests, and a 16‑sample smoke train/eval (time‑boxed).
        Start ephemeral local MLflow server in‑job (SQLite + local artifacts); log runs; do not write to Registry.
        Parity check: PyTorch vs. ONNX predictions on 5 samples within tolerance (e.g., mean abs diff < 1e‑4 post‑sigmoid).
    main:
        Start ephemeral MLflow server; full eval on validation set; log artifacts.
        Register best model to Registry → Staging; attach signature and example input; verify model load.
        Build Docker image; run container; smoke test /predict; export Prometheus metrics snapshot.
        Quality gates (fail CI if not met): Dice ≥ 0.80, parity check passes, signature present, API latency <150 ms on CPU for 256×256.
    Releases (tag):
        Optional: auto‑promote Staging → Production if gates met and a manual approval step is satisfied.
        Push Docker image to registry (GHCR/Docker Hub) and publish eval report as a release asset.

Monitoring and observability

    Online: Prometheus metrics from API (request_count, request_latency_ms histogram, error_count; labels: model_version).
    Logs: Structured JSON (request_id, timing, model_version, input shape).
    Offline: Evidently report comparing demo inputs vs. validation distribution and basic quality trends; store HTML in MLflow artifacts.
    Shadow mode: Optional flag to save inputs/outputs for offline analysis (no private data; demo only).

Security, privacy, ethics

    Only public, non‑identifiable images; respect dataset license.
    No clinical claims; “not for clinical use” in README/UI.
    Supply chain: pin dependencies; run pip‑audit in CI; pre‑commit gates.

System architecture

    Data: DVC‑tracked datasets + Great Expectations.
    Training: PyTorch + SMP; configs via Hydra; MLflow tracking.
    Registry/Artifacts: MLflow local server (SQLite + file artifacts).
    Serving: FastAPI + onnxruntime; Streamlit demo; Dockerized.
    Monitoring: Prometheus client + Evidently offline.
    CI/CD: GitHub Actions running lint/tests/smoke train/eval/parity, registry staging, image build, and API smoke test.

Repository layout

    docs/: PROJECT_SPEC.md, Data Card, Model Card, system diagram.
    configs/: Hydra configs (data.yaml, model.yaml, train.yaml, eval.yaml, serve.yaml).
    src/: data/ (download, preprocess, dataset, augmentations), models/ (unet, losses, metrics, export_onnx), train.py, eval.py, infer.py, validate_data.py.
    pipelines/: dvc.yaml, params.yaml.
    serving/: api.py, app.py (Streamlit), Dockerfile, docker-compose.yml.
    tests/: unit + integration + parity tests.
    mlruns/ (local artifact root), mlruns.db (SQLite) – gitignored.
    .github/workflows/: ci.yml.
    Makefile, pre-commit-config.yaml, requirements.txt or conda.yaml, .env.sample.

Quality and testing

    Unit: transforms, loss/metric shapes, dataset I/O, preprocessing parity trainer/serving.
    Integration: 16‑sample smoke train/eval; asserts on minimal Dice and runtime.
    Export: ONNX validity + parity; enforce MLflow model signature and example input presence.
    Data validation: Great Expectations executed in pipeline; block on critical failures.

Risks and mitigations

    Small dataset overfitting: strong augmentations, early stopping, and per‑image reporting.
    Train/serve skew: single preprocessing module imported by both paths; CI parity test.
    Repro drift: version pinning, DVC‑tracked data, MLflow environment capture, seeds.
