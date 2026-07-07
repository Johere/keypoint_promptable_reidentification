#!/usr/bin/env python3
"""Benchmark KPR OpenVINO IR inference (FP16 / FP16-INT8) on CPU / NPU / GPU.

Produces exactly two artifacts per run: a single-row summary CSV and a report.log.
Reuses the measurement harness (system monitor, CSV writer, timing loop) from
tools/benchmark_kpr_cpu.py.
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import torch

# Reuse the pure harness helpers (no dependency on benchmark_kpr_cpu's lazy globals).
from benchmark_kpr_cpu import (
    TeeStream,
    average,
    build_model_output_str,
    chunk_samples,
    derive_paths,
    fresh_batch,
    monitor_tmp_path,
    percentile,
    run_timed_loop,
    start_monitor,
    stop_monitor,
    summarize_monitor_csv,
    write_summary_csv,
    build_summary_row,
)
from torchreid.scripts.builder import build_config
from torchreid.tools.openvino_feature_extractor import KPROpenVINOFeatureExtractor, load_kpr_samples

DEFAULT_CONFIG = ROOT_DIR / "configs/kpr/imagenet/kpr_occ_posetrack_test.yaml"
DEFAULT_DEMO_DIR = ROOT_DIR / "assets/demo/soccer_players"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results"
DEFAULT_FP16_MODEL = ROOT_DIR / "openvino_models/kpr_fp16/kpr.xml"
DEFAULT_FP16_INT8_MODEL = ROOT_DIR / "openvino_models/kpr_fp16_int8/kpr.xml"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="KPR config yaml path.")
    parser.add_argument("--demo-dir", default=str(DEFAULT_DEMO_DIR), help="Demo dataset directory.")
    parser.add_argument("--model", default="", help="OpenVINO .xml model path. Default follows --precision.")
    parser.add_argument(
        "--precision",
        choices=["fp16", "fp16-int8"],
        default="fp16",
        help="Default model precision when --model is not specified. Default: fp16.",
    )
    parser.add_argument("--device", default="CPU", help="OpenVINO device: CPU, NPU, GPU, AUTO. Default: CPU.")
    parser.add_argument("--duration", type=float, default=60.0, help="Timed inference duration in seconds. Default: 60.")
    parser.add_argument("--warmup-iters", type=int, default=1, help="Warmup passes before timed measurement. Default: 1.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Samples per inference call. 0 means all loaded samples in one batch.",
    )
    parser.add_argument(
        "--static-batch",
        type=int,
        default=-1,
        help="Reshape the IR to this fixed batch before compiling. -1=auto (NPU->batch size, else dynamic). 0=dynamic.",
    )
    parser.add_argument("--cache-dir", default="", help="OpenVINO CACHE_DIR to amortize compile time (esp. NPU).")
    parser.add_argument("--monitor-interval", type=int, default=5, help="Seconds between system monitor rows. Default: 5.")
    parser.add_argument("--output-csv", default="", help="Summary CSV path. Default: results/kpr_ov_benchmark_<device>_<precision>_<ts>.csv.")
    parser.add_argument("--log-file", default="", help="Report log path. Default: same basename as summary CSV with .report.log suffix.")
    parser.add_argument("--no-monitor", action="store_true", help="Disable tools/monitor-system.sh for quick debugging.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append the summary row to --output-csv instead of overwriting (header written only if new).",
    )
    parser.add_argument("--limit-samples", type=int, default=0, help="Limit loaded demo samples. 0 uses all samples.")
    return parser.parse_args()


def resolve_model_path(args):
    if args.model:
        return Path(args.model)
    if args.precision == "fp16-int8":
        return DEFAULT_FP16_INT8_MODEL
    return DEFAULT_FP16_MODEL


def load_demo_samples(demo_dir, limit_samples):
    demo_dir = Path(demo_dir)
    all_samples = []
    for group_dir in sorted(demo_dir.glob("group*")):
        images_folder = group_dir / "images"
        keypoints_folder = group_dir / "keypoints"
        if images_folder.is_dir() and keypoints_folder.is_dir():
            all_samples.extend(load_kpr_samples(images_folder, keypoints_folder))
    if not all_samples:
        raise RuntimeError(f"No demo samples found under {demo_dir}")
    if limit_samples > 0:
        all_samples = all_samples[:limit_samples]
    return all_samples


def resolve_static_batch(args, effective_batch):
    device = args.device.upper()
    if args.static_batch >= 0:
        # Explicit override: 0 -> dynamic (None), N -> fixed batch N.
        return args.static_batch or None
    # Auto: NPU cannot compile dynamic shapes; CPU/GPU/AUTO stay dynamic.
    if device == "NPU":
        return effective_batch
    return None


def main():
    args = parse_args()
    DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = resolve_model_path(args)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"OpenVINO model not found: {model_path}\n"
            "Convert it first, for example:\n"
            "  python tools/convert_kpr_openvino.py --precision all"
        )

    device = args.device.upper()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    default_name = f"kpr_ov_benchmark_{device}_{args.precision}_{timestamp}.csv"
    output_csv = args.output_csv or str(DEFAULT_RESULTS_DIR / default_name)
    output_csv, default_report_log = derive_paths(output_csv)
    report_log = Path(args.log_file) if args.log_file else default_report_log
    monitor_tmp = monitor_tmp_path(report_log)

    report_log.parent.mkdir(parents=True, exist_ok=True)
    report_log_file = open(report_log, "w", encoding="utf-8")
    sys.stdout = TeeStream(sys.stdout, report_log_file)
    sys.stderr = TeeStream(sys.stderr, report_log_file)
    print(f"Report log: {report_log}")

    samples = load_demo_samples(args.demo_dir, args.limit_samples)
    sample_batches = chunk_samples(samples, args.batch_size)
    effective_batch = args.batch_size if args.batch_size > 0 else len(samples)

    static_batch = resolve_static_batch(args, effective_batch)
    if device == "NPU" and args.batch_size <= 0 and static_batch:
        # A single static shape can't cover variable group sizes; pin the batch.
        print("NPU needs a fixed batch; using --batch-size 1 with a static batch of 1.")
        args.batch_size = 1
        sample_batches = chunk_samples(samples, 1)
        effective_batch = 1
        static_batch = 1

    cfg = build_config(config_path=args.config)
    cfg.use_gpu = False

    print(f"config: {Path(args.config).resolve()}")
    print(f"model: {model_path.resolve()}  precision: {args.precision}")
    print(f"ov_device: {device}  batch_size: {effective_batch}  static_batch: {static_batch}")
    if args.cache_dir:
        print(f"cache_dir: {args.cache_dir}")
    print(f"input_samples: {len(samples)}")

    compile_start = time.perf_counter()
    extractor = KPROpenVINOFeatureExtractor(
        cfg,
        model_path=model_path,
        image_size=(cfg.data.height, cfg.data.width),
        pixel_mean=cfg.data.norm_mean,
        pixel_std=cfg.data.norm_std,
        device=device,
        batch_size=args.batch_size,
        static_batch=static_batch,
        cache_dir=args.cache_dir or None,
    )
    print(f"model compiled in {round(time.perf_counter() - compile_start, 3)} s")

    # Warmup + probe output shapes.
    for _ in range(max(args.warmup_iters, 0)):
        for batch in sample_batches:
            extractor(fresh_batch(batch))
    probe_samples, probe_emb, probe_vis, probe_parts = extractor(fresh_batch(sample_batches[0]))
    model_output = build_model_output_str(probe_emb, probe_vis, probe_parts)
    model_input = f"images:{effective_batch}x3x{cfg.data.height}x{cfg.data.width};prompt_masks:{effective_batch}x{extractor.prompt_mask_channels}x{cfg.data.height}x{cfg.data.width}"

    monitor_process = None
    if not args.no_monitor:
        monitor_process = start_monitor(args.monitor_interval, args.duration, monitor_tmp)

    try:
        latencies, timed_batches, timed_samples, actual_duration = run_timed_loop(
            extractor, sample_batches, args.duration
        )
    finally:
        stop_monitor(monitor_process)

    monitor_summary = summarize_monitor_csv(monitor_tmp if not args.no_monitor else "")

    latency_ms = [value * 1000.0 for value in latencies]
    fps = timed_samples / actual_duration if actual_duration > 0 else 0
    latency_ms_avg = average(latency_ms) if latency_ms else ""

    row = build_summary_row(
        device=f"openvino:{device}:{args.precision}",
        batch_size=effective_batch,
        model_input=model_input,
        model_output=model_output,
        fps=fps,
        latency_ms_avg=latency_ms_avg,
        monitor_summary=monitor_summary,
    )
    write_summary_csv(output_csv, row, append=args.append)

    print("---- detailed metrics (report.log only) ----")
    print(f"actual_duration_s: {round(actual_duration, 4)}")
    print(f"timed_inferences: {timed_batches}  timed_samples: {timed_samples}")
    if latency_ms:
        print(
            "latency_ms  avg/min/p50/p95/max: {}/{}/{}/{}/{}".format(
                round(average(latency_ms), 4),
                round(min(latency_ms), 4),
                round(percentile(latency_ms, 50), 4),
                round(percentile(latency_ms, 95), 4),
                round(max(latency_ms), 4),
            )
        )
    print(
        "cpu_pct avg/max: {}/{}  npu_pct avg/max: {}/{}  mem_used_gib avg/max: {}/{}  monitor_samples: {}".format(
            monitor_summary.get("cpu_pct_avg", ""),
            monitor_summary.get("cpu_pct_max", ""),
            monitor_summary.get("npu_pct_avg", ""),
            monitor_summary.get("npu_pct_max", ""),
            monitor_summary.get("mem_used_gib_avg", ""),
            monitor_summary.get("mem_used_gib_max", ""),
            monitor_summary.get("monitor_samples", 0),
        )
    )

    try:
        Path(monitor_tmp).unlink(missing_ok=True)
    except Exception:
        pass

    print(f"Benchmark summary CSV: {output_csv}")
    print(f"Report log: {report_log}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
