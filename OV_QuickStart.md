# KPR OpenVINO Quick Start

Keypoint Promptable Re-Identification (KPR) — from PyTorch weights to OpenVINO IR,
with inference and performance benchmarking on **CPU / NPU / GPU**.

> Reference platform: Intel Core Ultra X7 358H (PTL) — CPU + iGPU + NPU (Intel AI Boost),
> OpenVINO 2026.1.

## 1. Environment setup

```bash
git clone https://github.com/VlSomers/keypoint_promptable_reidentification ~/keypoint_promptable_reidentification
cd ~/keypoint_promptable_reidentification

python3 -m venv ~/python3-venv/kpr-reid
source ~/python3-venv/kpr-reid/bin/activate

export HF_ENDPOINT=https://hf-mirror.com
pip install -r requirements.txt
pip install -r requirements-ov.txt      # OpenVINO 2026.1 + NNCF
```

## 2. Download the demo weights

Download from Google Drive:
<https://drive.google.com/drive/folders/1t4wXc2c3qlFaqUCifAlc_OPrFwvb7peD>

Save the checkpoint to:

```
pretrained_models/kpr_occ_pt_IN_82.34_92.33_42323828.pth.tar
```

## 3. Model overview

KPR takes a person crop plus a keypoint prompt and produces part-based re-ID embeddings.

**Inputs**

| Name | Shape | Notes |
|---|---|---|
| `images` | `[B, 3, 256, 128]` | RGB crop, normalized (ImageNet mean/std) |
| `prompt_masks` | `[B, 10, 256, 128]` | Prompt heatmaps: 8 body parts + 1 background token + 1 negative-keypoints channel |

**Outputs**

| Name | Shape | Notes |
|---|---|---|
| `embeddings` | `[B, 9, 512]` | Per-part appearance features (8 parts + 1 foreground/global), 512-d each |
| `visibility_scores` | `[B, 9]` | Per-part visibility |
| `parts_masks` | `[B, 9, 64, 32]` | Per-part attention masks |

`B` is the batch size. The exported IR keeps a **dynamic batch** for CPU/GPU; the NPU path
reshapes it to a fixed batch (see §8).

**Complexity**

| Metric | Value | Notes |
|---|---|---|
| Parameters | **91.03 M** | SwinV2-base appearance backbone + KPR heads (91.02 M trainable) |
| FLOPs | **≈ 29.65 GFLOPs** (14.82 GMACs) | Per sample, `B=1`, input `3×256×128` + prompt `10×256×128` |

## 4. Smoke test (PyTorch)

```bash
python demo.py
```

> Results are saved to `assets/demo/results`.

## 5. Benchmark the PyTorch model (CPU)

```bash
python tools/benchmark_kpr_cpu.py --duration 60 --batch-size 1
```

Each run produces exactly two artifacts under `results/`:

- `kpr_cpu_benchmark_<timestamp>.csv` — single-row summary.
- `kpr_cpu_benchmark_<timestamp>.report.log` — full terminal + system-monitor log.

See [tools/benchmark_readme.md](tools/benchmark_readme.md) for all options.

## 6. Convert to OpenVINO IR (run on PTL 358H)

```bash
python tools/convert_kpr_openvino.py --config configs/kpr/imagenet/kpr_occ_posetrack_test.yaml --precision all
```

The IR models are saved to:

```
openvino_models/
├── kpr_fp16
│   ├── kpr.bin
│   └── kpr.xml
└── kpr_fp16_int8
    ├── kpr.bin
    └── kpr.xml
```

## 7. Smoke test (OpenVINO)

```bash
# CPU / GPU accept the dynamic-batch IR directly
python demo-ov.py --precision fp16       --device CPU
python demo-ov.py --precision fp16-int8  --device GPU

# NPU cannot compile a dynamic shape, so the model is reshaped to a static
# batch of 1 automatically (batch size is forced to 1 on NPU)
python demo-ov.py --precision fp16-int8  --device NPU
```

> Results are saved to `assets/demo/results`, named per precision and per device
> (e.g. `samples_grid_ov_fp16-int8_CPU.png`).

## 8. Benchmark the OpenVINO models (CPU / NPU / GPU)

`tools/benchmark_kpr_ov.py` mirrors the PyTorch benchmark harness (same system monitor,
same two artifacts) and adds `--device` / `--precision` / `--static-batch` / `--cache-dir`.

```bash
# CPU
python tools/benchmark_kpr_ov.py --device CPU --precision fp16       --duration 60
python tools/benchmark_kpr_ov.py --device CPU --precision fp16-int8  --duration 60

# GPU (iGPU)
python tools/benchmark_kpr_ov.py --device GPU --precision fp16       --duration 60
python tools/benchmark_kpr_ov.py --device GPU --precision fp16-int8  --duration 60

# NPU — static batch of 1; first compile takes a few seconds, use --cache-dir to amortize
python tools/benchmark_kpr_ov.py --device NPU --precision fp16-int8  --duration 60 \
  --cache-dir openvino_models/.ov_cache
python tools/benchmark_kpr_ov.py --device NPU --precision fp16       --duration 60 \
  --cache-dir openvino_models/.ov_cache
```

Each run produces two artifacts under `results/`:

- `kpr_ov_benchmark_<device>_<precision>_<timestamp>.csv` — single-row summary.
- `kpr_ov_benchmark_<device>_<precision>_<timestamp>.report.log` — full log.

### Performance data

Measured on the reference platform (Intel Core Ultra X7 358H, OpenVINO 2026.1),
`batch_size = 1`, 60 s timed run, `soccer_players` demo. Input `images 1×3×256×128` +
`prompt_masks 1×10×256×128`; output `embeddings 1×9×512`, `visibility_scores 1×9`,
`parts_masks 1×9×64×32`.

| Device | Precision | FPS | Latency (ms) | CPU avg/max | NPU avg/max | Peak mem (GiB) |
|---|---|--:|--:|--:|--:|--:|
| CPU | fp16 | 7.07 | 141.37 | 28.8% / 32.0% | — | 14.57 |
| CPU | fp16-int8 | 15.57 | 64.24 | 36.4% / 41.0% | — | 14.64 |
| NPU | fp16 | 58.35 | 17.14 | 47.0% / 49.0% | 73.9% / 80.0% | 14.90 |
| NPU | fp16-int8 | 44.20 | 22.62 | 35.6% / 39.0% | 81.1% / 85.0% | 14.86 |
| GPU | fp16 | **63.85** | **15.66** | 79.6% / 80.0% | — | 15.09 |
| GPU | fp16-int8 | 63.62 | 15.72 | 79.8% / 82.0% | — | 15.03 |

### Summary CSV columns

The summary CSV is a single row with these columns:

| Column | Meaning |
|---|---|
| `device` | `pytorch:CPU`, `openvino:CPU:fp16`, `openvino:NPU:fp16-int8`, `openvino:GPU:fp16`, … |
| `batch_size` | Samples per inference call |
| `model_input` | Input tensor shapes |
| `model_output` | Output tensor shapes |
| `fps` | Images per second |
| `latency_ms_avg` | Average per-inference latency (ms) |
| `cpu_pct_avg` / `cpu_pct_max` | System CPU utilization |
| `npu_pct_avg` / `npu_pct_max` | NPU utilization (from `intel_vpu` sysfs) |
| `mem_used_gib_max` | Peak used memory (GiB) |

Latency percentiles (p50/p95/min/max), average memory, swap, and per-sample monitor rows
are written to the `report.log`.

### Notes

- **NPU** requires a static shape. The benchmark auto-reshapes the IR to the effective batch
  size (default 1) before compiling; force it with `--static-batch N` or disable with
  `--static-batch 0` (dynamic — CPU/GPU only).
- **NPU first-run compile** (`Level0 pfnCreate`) takes a few seconds; `--cache-dir` caches the
  compiled blob so subsequent runs start fast.
- List available devices with:
  ```bash
  python -c "import openvino as ov; print(ov.Core().available_devices)"
  ```
