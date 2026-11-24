# ForgeVision AI: Step-by-Step Implementation Guide

This guide turns [ForgeVision_AI.md](../ForgeVision_AI.md) into concrete steps. It assumes **no local NVIDIA GPU**:

- Everything that doesn't need a GPU runs on your Windows machine, on CPU.
- Everything that needs a GPU runs in **Google Colab (free T4)**, using code pulled from your GitHub repo.

Each phase ends with a **"Done when"** checklist. Don't start the next phase until that list is true.

---

## 0. How the work splits between local and Colab

| Work | Where | Why |
|---|---|---|
| Repo, data code, evaluation code, unit tests | Local (CPU) | No GPU needed. Fast edit/test loop. |
| Fitting PatchCore/PaDiM, training the autoencoder, running evaluation | Colab T4 | Feature extraction on CPU is very slow. |
| FastAPI backend, Postgres, React dashboard | Local (Docker Desktop) | Inference uses **ONNX Runtime on CPU**, so the whole product runs without a GPU. |
| ONNX export and parity checks | Colab (export), local (CPU parity) | |
| TensorRT engines, FP16/INT8, GPU benchmarks | Colab T4 | `pip install tensorrt` works there |
| CUDA kernel, C++ TensorRT runtime, Nsight profiling | Colab T4 | `nvcc` is available. Nsight Systems installs with `apt` in each session. |
| DeepStream | **Optional**: a rented Linux GPU for a few hours | Needs Docker with the NVIDIA runtime, which Colab can't provide |
| Streaming simulator, Prometheus/Grafana | Local (Docker) | |

**Rules of the Colab workflow:**

1. All code lives in the GitHub repo. A notebook only does `git clone/pull`, `pip install -e .`, downloads data and runs `python -m ...`. **No model logic in notebooks.** Reviewers need to be able to reproduce the work from the repo.
2. Datasets go on the Colab VM disk (`/content/data`), downloaded fresh each session (see [data/README.md](../data/README.md)).
3. **Outputs go to Google Drive** (`/content/drive/MyDrive/forgevision/`): checkpoints, ONNX files, MLflow runs, benchmark JSON and profiles. The VM is wiped when the session ends (a ~12 h hard cap, plus an idle timeout).
4. Write checkpoints often. Free Colab GPU quota changes over time and isn't guaranteed. **Kaggle Notebooks** are a backup: a 12 h session cap and a weekly GPU quota (often quoted as ~30 h, but check your account).
5. In a GPU notebook, **record the environment for every run**: `nvidia-smi`, `nvcc --version`, `torch.__version__`, the TensorRT version and the git commit hash. Benchmark numbers mean nothing without these.

---

## Phase 0: Setup (about 2–4 days)

### 0.1 Install locally (Windows)

- **Git** and a **GitHub** account
- **Python:** check which version Colab currently runs (`!python --version`) and use the **same minor version** locally. Use [uv](https://docs.astral.sh/uv/) to manage Python and dependencies.
- **Node.js LTS** (for the React dashboard)
- **Docker Desktop** with the **WSL2** backend (for Postgres, Prometheus, Grafana and MediaMTX)
- **VS Code** with the Python, Ruff, ESLint, Docker and C/C++ extensions

### 0.2 Accounts

- Google account (Colab and Drive)
- **NVIDIA Developer** account (free). You need it for the TensorRT tar package (C++ headers and `trtexec`), Nsight and the DeepStream documentation.

### 0.3 Create the repo

```bash
cd forge-vision-ai
git init -b main
mkdir -p configs training/forgevision_train inference/python inference/cpp inference/cuda \
  streaming backend frontend benchmarks tests infra docs notebooks .github/workflows
```

The `.gitignore` must include at least:

```gitignore
data/raw/
*.ckpt
*.pt
*.onnx
*.engine
*.plan
mlruns/
artifacts/
*.nsys-rep
.env
node_modules/
__pycache__/
.venv/
```

Use **one Python package** with a `pyproject.toml`, so `pip install -e .` works the same locally and in Colab. Split dependencies into extras:

- `[train]`: torch, torchvision, timm, scikit-learn, mlflow, anomalib (for reference comparisons only)
- `[serve]`: fastapi, uvicorn, pydantic-settings, sqlalchemy, alembic, psycopg, onnxruntime, pillow, prometheus-client, python-jose or pyjwt, passlib
- `[dev]`: pytest, ruff, mypy, pre-commit, httpx

Pin versions in a lock file (`uv lock`). Colab already ships torch and CUDA, so **don't reinstall torch in Colab**. Install your package with `pip install -e ".[train]" --no-deps` and then add any missing packages, or you'll break Colab's CUDA build of torch.

### 0.4 Quality gates from day one

- `ruff` (lint and format), `mypy --strict` on `backend/` and `inference/python/`, and `pytest`
- `pre-commit` running ruff
- A GitHub Actions workflow (`.github/workflows/ci.yml`) with ruff, mypy and pytest for Python, plus lint and build for the frontend. Mark GPU tests `@pytest.mark.gpu` and skip them in CI.

### 0.5 Colab bootstrap notebook (`notebooks/00_colab_bootstrap.ipynb`)

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone https://github.com/<you>/forge-vision-ai.git /content/repo || (cd /content/repo && git pull)
%cd /content/repo
!pip install -q -e ".[train]"
!bash scripts/download_mvtec.sh /content/data/mvtec_ad bottle metal_nut
!nvidia-smi; python -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"
```

For a private repo, store a GitHub token in **Colab Secrets** (the key icon) and read it with `google.colab.userdata`. Never paste the token into a cell.

**Done when:** `pytest` passes locally and CI is green on GitHub. In Colab, the bootstrap notebook clones the repo, downloads `bottle` and prints the T4 GPU name.

---

## Phase 1: Data and baseline (weeks 1–2)

### 1.1 Download scripts

Put the commands from [data/README.md](../data/README.md) into `scripts/download_mvtec.sh`, which takes a category list as arguments. Record the `sha256sum` of each archive in `configs/datasets.lock.json`. The download script checks against it, so you'll notice if the upstream files change.

### 1.2 Splits: decide them once and freeze them

MVTec AD has no validation split, and `train/` contains only good images. You can't set a defect threshold without seeing defects, so use this protocol and document it in the model card:

| Split | Source | Used for |
|---|---|---|
| `train` | 90% of `train/good` | Fitting the memory bank or statistics |
| `val_normal` | 10% of `train/good` | Score normalization and sanity checks |
| `val` | **A stratified 30% of the official `test/`** (good and each defect type), fixed seed | Choosing thresholds and the review band |
| `test` | The remaining 70% of `test/` | **Final reported metrics.** Touched once per model version. |

Also report **image AUROC and pixel AUROC on the full official test set**, clearly labeled, so your numbers can be compared with published papers. AUROC needs no threshold, so this doesn't leak.

Write the split to `configs/splits/mvtec_<category>_seed42.json` as lists of relative paths, and commit it. Add a test that fails if any file appears in two splits.

### 1.3 Dataset code (`training/forgevision_train/data.py`)

- A `Dataset` that returns `image`, `mask` (all zeros for good images), `label` and `path`
- **One preprocessing spec** shared by training and inference: resize, center crop, ImageNet mean and std, RGB order. Store it in the model manifest (see 2.5). The C++ and CUDA code will reproduce it exactly later.
- Load masks with **nearest-neighbour** resize. Bilinear resizing creates fractional mask values.

### 1.4 Evaluation code (`training/forgevision_train/evaluate.py`)

Build this **before** any model and test it on synthetic score arrays.

- Image level: AUROC, AUPR, and at a chosen threshold: precision, recall, F1, FPR, FNR, and the confusion matrix
- Pixel level: pixel AUROC, plus **AUPRO** (area under the per-region overlap curve, up to 30% FPR). It's the localization metric MVTec reviewers expect.
- Threshold selection on `val`: either maximize F1, or (more realistic for a factory) set a **target FNR of at most X%** and accept the FPR that results. Report both.
- **Review band:** two thresholds, `t_pass < t_reject`. A score below `t_pass` is a PASS, a score above `t_reject` is a REJECT, and anything in between is REVIEW. Choose them on `val`, e.g. so that the automatic decisions keep precision ≥ 99%.
- Save per-image scores to CSV so failure analysis doesn't need re-inference.

### 1.5 Experiment tracking

Use MLflow with a **file store on Drive**: `mlflow.set_tracking_uri("file:/content/drive/MyDrive/forgevision/mlruns")`. Log the config, seed, git commit, GPU name, timings, metrics and artifact hashes. Locally, open the UI with `mlflow ui --backend-store-uri <synced folder>`. Google Drive for desktop can sync the folder to your machine.

### 1.6 Baseline: convolutional autoencoder

- A small conv autoencoder at 256×256, trained only on `train`, with MSE + SSIM loss
- The anomaly map is the per-pixel reconstruction error, smoothed. The image score is the max (or top-k mean) of the map.
- It's **meant to be weak**. Its job is to show that you measured the improvement PatchCore gives over a naive approach.
- Fix every seed and use `torch.use_deterministic_algorithms(True, warn_only=True)`.

**Done when:** splits are frozen and tested. `python -m forgevision_train.evaluate --model autoencoder --category bottle` in Colab gives the same metrics on two runs, and MLflow shows the run.

---

## Phase 2: Anomaly model and localization (weeks 3–4)

### 2.1 PatchCore: implement it yourself (about 200 lines), check it against Anomalib

How it works (describe it this way in your write-up; it's fitting, not training):

1. A frozen ImageNet backbone (`wide_resnet50_2`) extracts features from **layer2 and layer3**.
2. Average-pool the features with a 3×3 window (local neighbourhood). Upsample layer3 to layer2's resolution, concatenate, and project to a lower dimension. For a 224 input this gives a 28×28 grid of patch features.
3. Collect all patch features from `train`, then apply **greedy coreset subsampling** (1–10%) to build the **memory bank**.
4. At inference, each patch's score is the distance to its nearest neighbour in the memory bank. Upsample the score grid to the image size and apply a Gaussian blur (σ≈4). The image score is the maximum patch score.

**Implementation details:**

- Size the memory bank before you build it. For example, ~200 images × 784 patches × 10% ≈ 16k vectors × 1536 dims × 4 bytes ≈ **96 MB**. Coreset selection on all ~160k vectors runs on the GPU in Colab. Use a random projection inside the greedy step to keep it fast.
- Compute nearest-neighbour distances with **explicit matrix operations**: `‖a‖² − 2a·b + ‖b‖²`, then `clamp(min=0).sqrt()`. Don't use `torch.cdist`, because it doesn't export cleanly to ONNX. Anomalib made the same change for the same reason.
- Run Anomalib's `Patchcore` on the same split as a **reference check**. Your image AUROC should be within about 1 point of it. If it's further off, compare the preprocessing first.
- Anomalib API (v2): `from anomalib.data import MVTecAD`, `from anomalib.models import Patchcore`, `from anomalib.engine import Engine` → `engine.fit(...)`, `engine.test(...)`, `engine.export(...)`. Anomalib's automatic MVTec download URLs have broken before, so point `root=` at the data you downloaded yourself.

### 2.2 PaDiM (second model)

- A per-patch multivariate Gaussian. The score is the Mahalanobis distance.
- **Check the memory size.** A 28×28 grid × a d×d inverse covariance matrix grows as d². With d=550 that's about 950 MB, so use **resnet18 with d=100**: 784 × 100 × 100 × 4 B ≈ 31 MB.
- PaDiM is **the easiest model to export to ONNX and TensorRT** (just matrix operations, no nearest-neighbour search). Plan to use PaDiM for the first TensorRT milestone if PatchCore causes trouble.

### 2.3 Run the protocol

For `bottle`, then `metal_nut`, then the other MVTec categories, report for each:

- Image AUROC, pixel AUROC, AUPRO
- At the chosen threshold: precision, recall, F1, FPR, FNR, and the percentage of images sent to review
- Fit time, inference time per image (with the GPU named), memory bank size

Then run the same code on **VisA** (`pcb1`–`pcb4`, `capsules`) without changing hyperparameters. That tests whether the method generalizes, which matters more than one more decimal point on MVTec.

### 2.4 Failure analysis (`docs/reports/failure_analysis.md`)

- The 10 worst false negatives and 10 worst false positives, each with the image, heatmap and mask
- Group them by cause, e.g. tiny defect, defect near the border, misalignment, lighting
- This section is what makes the project look like an engineer's work. Most portfolio projects skip it.

### 2.5 Model manifest and packaging

Every model you export produces a folder like this:

```text
artifacts/models/patchcore-bottle-v1/
  model.onnx
  manifest.json
  metrics.json
```

```json
{
  "name": "patchcore-bottle", "version": "1.0.0",
  "framework": "onnx", "opset": 17,
  "input": {"shape": [1, 3, 224, 224], "dtype": "float32", "layout": "NCHW", "color": "RGB"},
  "preprocess": {"resize": 256, "center_crop": 224, "interpolation": "bilinear",
                 "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
  "outputs": {"anomaly_map": [1, 1, 224, 224], "score": [1]},
  "thresholds": {"pass_below": 0.0, "reject_above": 0.0},
  "supported_precision": ["fp32"],
  "artifact_sha256": "…", "git_commit": "…", "dataset_split": "configs/splits/mvtec_bottle_seed42.json"
}
```

Leave the thresholds as placeholders until they're chosen on `val`. Then write the real values.

**Done when:** PatchCore and PaDiM results exist for at least 2 categories. Your PatchCore is within about 1 AUROC point of Anomalib. The failure analysis is written. The ONNX model plus manifest is in Drive.

---

## Phase 3: Backend API and database (weeks 5–6, local, no GPU)

### 3.1 Docker Compose (`infra/docker-compose.yml`)

Services: `postgres`, `api`, and (later) `frontend`, `prometheus`, `grafana`, `mediamtx`. Keep all secrets in a `.env` file that's never committed. Commit a `.env.example` instead.

### 3.2 Inference runtime abstraction (`inference/python/`)

```python
class InferenceRuntime(Protocol):
    manifest: ModelManifest
    def warmup(self) -> None: ...
    def infer(self, image_rgb_uint8: np.ndarray) -> InferenceResult: ...  # score, anomaly_map, timings
```

Implementations:

- `OnnxRuntimeCPU`: the default locally
- `TorchRuntime`: Colab and reference
- `TensorRTRuntime`: Phase 5

Preprocessing lives in **one function** driven by the manifest. Its output is the "golden" reference that every other implementation (C++, CUDA) must match.

### 3.3 Database (SQLAlchemy 2.0 + Alembic)

Tables from the spec: `production_lines`, `products`, `model_versions`, `inspections`, `defect_regions`, `reviews`, `audit_events`.

- `inspections.status` is an enum: `PASS`, `REJECT`, `REVIEW`, `ERROR`. **An exception during inference always produces `ERROR`, never `PASS`.** Write a test that injects a runtime failure and checks this.
- Index `(line_id, created_at)`, `status` and `model_version_id`.
- `audit_events` is append-only: actor, action, entity, entity_id, before/after JSON, timestamp. Write to it in the same transaction as the change it records.
- `defect_regions`: threshold the anomaly map, find connected components, and store bounding boxes and areas.
- Images and heatmaps go to `artifacts/` under **server-generated UUID paths**. Never use the uploaded filename, which prevents path traversal. The database stores only the relative key.

### 3.4 Endpoints

Implement the table in spec §8. Important details:

- **Upload validation:** maximum file size (e.g. 10 MB), check the magic bytes and not only the `Content-Type`, decode with Pillow, set a maximum pixel count (`Image.MAX_IMAGE_PIXELS`) against decompression bombs, convert to RGB, reject anything that fails to decode.
- **Authentication:** OAuth2 password flow with JWT. Roles: `operator`, `quality_engineer`, `ml_engineer`, `admin`. Activating a model needs `ml_engineer` or `admin`, and the change is audited.
- **Model activation:** load the new runtime, warm it up and run a smoke test, *then* swap it in atomically. If the smoke test fails, keep the old model. Rollback is just activating the previous version.
- `/readyz` returns 503 until a model is loaded and warmed up. `/healthz` is liveness only.
- `/metrics`: Prometheus counters and histograms for inference latency, requests by status, errors and queue depth.
- Run inference outside the event loop (`run_in_threadpool`) so ONNX Runtime doesn't block the async server.

### 3.5 Tests

- Unit tests: validation, decision logic (pass/review/reject bands), path safety
- Integration tests: use the real Postgres in Docker (or testcontainers), not SQLite, because enum and JSON behaviour differ.
- API tests with `httpx.AsyncClient`: permissions (an operator can't activate a model), audit rows written, invalid uploads rejected (oversized file, wrong type, truncated image)

**Done when:** `docker compose up` gives a working API. Uploading an MVTec image returns a decision and a heatmap from ONNX Runtime on CPU. Results persist, invalid uploads return 4xx, and every state change has an audit row.

---

## Phase 4: Operator dashboard (weeks 7–8, local)

### 4.1 Stack

Vite + React + TypeScript, TanStack Query (server state), React Router, a component library (shadcn/ui or MUI) and Recharts. Generate a typed API client from FastAPI's OpenAPI schema (`openapi-typescript`), so the frontend and backend can't drift apart.

### 4.2 Screens (spec §9)

1. **Inspect:** upload or choose an image. Show the image with a heatmap overlay (opacity slider), score against the thresholds, decision, model version and latency.
2. **Review queue:** list `REVIEW` and `ERROR` items. Accept or reject, add a note, view the history (from `audit_events`).
3. **Analytics:** defect rate per category over time, review outcomes (how often operators override the model, i.e. the false-positive and false-negative rates in practice).
4. **Models:** versions, metrics, the active model, activate or roll back (role-gated).
5. **Operations:** throughput, p95 latency, error rate, queue depth. Show GPU utilization as N/A on the CPU deployment.

For live updates, use **SSE** (`/v1/events`) and push new inspections to the Inspect and Review screens.

### 4.3 Export corrected labels

Add a quality-engineer action that exports reviewed images and their corrected labels as a CSV plus a zip. This is the start of the active-learning loop.

### 4.4 MVP release

- README quick start: `docker compose up`, demo login, seed script that loads the model and a few sample inspections
- A model card (`docs/model_card.md`): intended use, data, splits, metrics, failure modes, license limits
- Tag `v0.1.0` and record a 3-minute demo

**Done when:** the full flow works in the browser (inspect, review, audit history visible, model rollback). **The MVP is shippable at this point.** If time runs out, stop here and list the advanced items as future work.

---

## Phase 5: ONNX parity and TensorRT (weeks 9–10, Colab)

### 5.1 ONNX parity (CPU, can run locally)

- Run the same 50 images through PyTorch and ONNX Runtime. Assert `max_abs_diff(anomaly_map) < 1e-4` and that the decisions are identical. Commit this as a test.
- Use `onnx.checker` and run `onnxsim` (onnx-simplifier) before TensorRT.

### 5.2 TensorRT in Colab

```python
!pip install -q tensorrt  # Python builder and runtime; does NOT include trtexec
```

- **Check that your TensorRT version supports the T4** (Turing, compute capability 7.5) in the TensorRT support matrix. If the newest release has dropped Turing, pin an earlier version.
- Build engines with the Python API (`trt.Builder`, `OnnxParser`, `BuilderConfig`) and save them to Drive. Use an **optimization profile** for the batch dimension (min 1, optimal 4, max 8).
- If you want `trtexec` (useful for quick checks), download the **TensorRT tar package** from NVIDIA once, store it in Drive, and extract it in each session. The same package provides the C++ headers and libraries you need in Phase 6.
- Engines are tied to a specific GPU and TensorRT version. Rebuild them on each new machine and never commit them.

### 5.3 Precision experiments

| Run | What to check |
|---|---|
| FP32 | Must match ONNX within tolerance |
| FP16 | Is the AUROC change below 0.1 points? PatchCore distances can lose precision, so check the decisions too. |
| INT8 (optional) | Calibrate on `train` images only. Report the accuracy change honestly. Skip it if it hurts. |

### 5.4 Benchmark harness (`benchmarks/run_gpu_bench.py`)

- Use the same inputs and the same preprocessing for every backend: CPU (ONNX Runtime), PyTorch GPU, TensorRT FP32 and TensorRT FP16.
- Do **50 warm-up iterations**, then **500 timed** ones.
- Time the GPU work with **CUDA events** and the end-to-end latency with `time.perf_counter()` after `torch.cuda.synchronize()`. Report p50, p95 and p99, throughput (FPS) at batch sizes 1, 4 and 8, and peak memory.
- **Colab-specific:** the T4 is shared and may throttle. Run each configuration 3 times and report the median with its spread. Log `nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu --format=csv` before and after.
- Write the results to JSON, then generate the benchmark table in `docs/reports/benchmarks.md` with a script. Never type numbers in by hand.

**Done when:** a parity test exists for each backend, the FP32/FP16 table includes accuracy changes, and the benchmark report states the exact GPU, driver, CUDA and TensorRT versions.

---

## Phase 6: C++ runtime and custom CUDA kernel (weeks 11–12, Colab)

Write the code locally in VS Code, push it, then `git pull` and build in Colab:

```bash
!nvcc --version && cmake --version
!cmake -S inference/cpp -B build -DCMAKE_BUILD_TYPE=Release -DTensorRT_ROOT=/content/TensorRT-x.y && cmake --build build -j
!./build/tests/forgevision_tests
```

### 6.1 Fused preprocessing kernel (`inference/cuda/`)

One kernel replaces five CPU steps: **uint8 HWC BGR → bilinear resize → BGR to RGB → normalize (mean/std) → float32 (or half) NCHW**.

- One thread per output pixel. Read the 4 neighbouring source pixels, interpolate, normalize all 3 channels and write to the 3 planes.
- **The bilinear sampling must match your Python reference exactly.** OpenCV and PIL treat pixel centres (`align_corners`) differently. Pick one convention, document it, and test against it.
- Baselines to compare against: (a) OpenCV on CPU and NumPy, (b) the same steps as separate PyTorch GPU operations. Report the speed-up of the kernel alone **and** of the full pipeline.

### 6.2 Kernel tests (GoogleTest via CMake `FetchContent`)

- Compare against a C++ CPU reference: max abs diff ≤ 1e-5 for FP32, ≤ 1e-2 for FP16
- Edge cases: odd sizes (1×1, 7×13, 1999×1001), upscaling and downscaling, non-contiguous row pitch (pitched input)
- Run `compute-sanitizer --tool memcheck ./forgevision_tests` (included with the CUDA toolkit) to catch out-of-bounds memory access.

### 6.3 C++ TensorRT runtime (`inference/cpp/`)

- Read `manifest.json` (nlohmann/json), deserialize the engine, create the execution context, allocate device buffers once and reuse them.
- Pipeline: decode the image (OpenCV) → copy to **pinned** host memory → async copy to device on a CUDA stream → preprocessing kernel → `enqueueV3` → async copy of the results back
- Run this only **after** you have a baseline: measure pageable vs pinned memory and synchronous vs asynchronous copies, and show the difference in Nsight.
- Parity test: C++ output vs Python ONNX Runtime output on the same images

### 6.4 Profiling (Nsight Systems in Colab)

Install it with `apt` from NVIDIA's repository at the start of each session (it isn't kept between sessions), then:

```bash
!nsys profile -o /content/drive/MyDrive/forgevision/profiles/cpp_fp16 --trace=cuda,nvtx ./build/forgevision_bench
!nsys stats /content/drive/MyDrive/forgevision/profiles/cpp_fp16.nsys-rep
```

Open the `.nsys-rep` file in the **Nsight Systems GUI on Windows** (it doesn't need a GPU to view a report). Add NVTX ranges around preprocess, infer and postprocess so the timeline is readable. Put screenshots and conclusions in `docs/reports/profiling.md`: where the time goes, what you changed, and the before/after numbers.

**Done when:** kernel tests pass, including the sanitizer, the C++ output matches Python, and the profiling report shows at least one measured optimization with evidence.

---

## Phase 7: Streaming, monitoring, release (weeks 13–14)

### 7.1 Camera simulator (local, CPU)

- Run **MediaMTX** in Docker as an RTSP server.
- `streaming/simulator.py` turns a folder of test images into a video with ffmpeg at a set frame rate (e.g. 10 FPS) and publishes it as `rtsp://localhost:8554/line1`. Run N of these for N cameras.
- Mix good and defective images in a known order, so streaming results can be checked against ground truth.

### 7.2 Stream consumer

- One reader thread per camera (OpenCV `VideoCapture`) feeding a **bounded queue** (`maxsize=8`).
- **Drop policy: drop the oldest frame** when the queue is full, because a live inspection system prefers fresh frames. Count every dropped frame.
- Metrics per camera: received, processed, dropped, queue wait time, end-to-end latency
- Tests: a slow consumer (drops increase but memory doesn't), the stream disconnecting (reconnect with backoff, `ERROR` status recorded), and recovery.

### 7.3 DeepStream (optional)

DeepStream needs Linux, an NVIDIA GPU and Docker with the NVIDIA runtime. Colab can't provide that. Choose one:

- **Skip it** and say so in "Known limitations". Your custom pipeline already shows the same ideas.
- **Rent a Linux GPU for a few hours** (RunPod, Vast.ai or Lambda), run the DeepStream container with an `nvinfer` config that points to your engine, record the throughput and shut the instance down. Set a spending cap first.

### 7.4 Monitoring

Prometheus scrapes `/metrics`. Grafana is provisioned from files in `infra/grafana/` with one dashboard: request rate, p50/p95/p99 latency, error rate, decisions by status, review rate, and dropped frames per camera. On GPU hosts, add the DCGM exporter for GPU utilization.

### 7.5 Release checklist (spec §16)

- [ ] `README.md`: problem, architecture diagram, quick start (CPU only), demo GIF, results table, known limitations
- [ ] `docs/model_card.md`, `docs/reports/{benchmarks,failure_analysis,profiling}.md`
- [ ] Every number in the docs comes from a script in `benchmarks/` or `training/`, with the hardware stated
- [ ] CI green. GPU tests documented with the exact Colab commands to run them.
- [ ] A 3-minute demo video: inspect, then review, then roll back the model, then the TensorRT benchmark
- [ ] A technical write-up (blog post or `docs/writeup.md`): failure cases, the optimization trade-offs, and what you would do with real factory data
- [ ] Tag `v1.0.0`

---

## Common pitfalls

1. **Tuning on the test set.** That includes choosing thresholds on the same 70% you report. Freeze the splits before the first model run.
2. **Preprocessing drift** between Python, C++ and CUDA: RGB vs BGR, resize conventions, crop order. The manifest plus parity tests prevent it.
3. **Reinstalling torch in Colab** breaks its CUDA build. Install with `--no-deps` and add only what's missing.
4. **Benchmarking without warm-up or synchronization.** GPU calls are asynchronous, so unsynchronized timings measure nothing.
5. **Comparing CPU and GPU numbers without naming the hardware.** Always report the exact CPU model, GPU, clocks and versions.
6. **Claiming PatchCore is "trained".** It's fitted: a frozen backbone plus a memory bank. Describe it accurately.
7. **Treating an inference failure as PASS.** Always `ERROR` or `REVIEW`, and test it.
8. **Committing data, checkpoints or engines.** Check `.gitignore` before the first push.
9. **Losing Colab work.** Save checkpoints and reports to Drive as you go, not at the end.

## Cost

- Colab free tier is enough for Phases 1, 2, 5 and 6 if you checkpoint to Drive. Colab pay-as-you-go (compute units) is an optional backup if the free GPU is often unavailable.
- Kaggle Notebooks are a free backup GPU.
- DeepStream rental is optional: a few hours on a hourly-billed GPU, with a spending cap.
- Everything else runs on your own machine.

## References

- [Anomalib docs ("Anomalib in 15 Minutes")](https://anomalib.readthedocs.io/en/stable/markdown/get_started/anomalib.html) · [PatchCore in Anomalib](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/patchcore.html)
- [PatchCore paper: *Towards Total Recall in Industrial Anomaly Detection*](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf)
- [TensorRT pip installation](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html) (wheels don't include `trtexec`)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide) · [Nsight on Colab example](https://rastringer.github.io/gpu_cuda_book/nsight_attention.html)
- [Colab FAQ (usage limits)](https://research.google.com/colaboratory/faq.html)
- [Benchmark saturation on MVTec: *Beyond Academic Benchmarks*](https://arxiv.org/html/2503.23451v1)
