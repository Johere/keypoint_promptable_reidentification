#!/usr/bin/env python3
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

DEFAULT_CONFIG = ROOT_DIR / "configs/kpr/imagenet/kpr_occ_posetrack_test.yaml"
DEFAULT_DEMO_DIR = ROOT_DIR / "assets/demo/soccer_players"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results"

# Strictly ordered columns of the single-row summary CSV. Everything else goes to report.log.
SUMMARY_FIELDS = [
    "device",
    "batch_size",
    "model_input",
    "model_output",
    "fps",
    "latency_ms_avg",
    "cpu_pct_avg",
    "cpu_pct_max",
    "npu_pct_avg",
    "npu_pct_max",
    "mem_used_gib_max",
]


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark KPR PyTorch CPU inference and collect system resource metrics."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="KPR config yaml path.")
    parser.add_argument(
        "--demo-dir",
        default=str(DEFAULT_DEMO_DIR),
        help="Demo dataset directory containing group*/images and group*/keypoints.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Timed inference duration in seconds. Default: 60.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=1,
        help="Warmup passes before timed measurement. Default: 1.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Samples per inference call. 0 means all loaded samples in one batch.",
    )
    parser.add_argument(
        "--monitor-interval",
        type=int,
        default=5,
        help="Seconds between system monitor rows. Default: 5.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="torch.set_num_threads value. 0 keeps PyTorch default.",
    )
    parser.add_argument(
        "--torch-interop-threads",
        type=int,
        default=0,
        help="torch.set_num_interop_threads value. 0 keeps PyTorch default.",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Summary CSV path. Default: results/kpr_cpu_benchmark_<timestamp>.csv.",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="Report log path. Default: same basename as summary CSV with .report.log suffix.",
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Disable tools/monitor-system.sh for quick debugging.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append the summary row to --output-csv instead of overwriting (header written only if new).",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=0,
        help="Limit loaded demo samples. 0 uses all samples.",
    )
    return parser.parse_args()


def load_kpr_samples(images_folder, keypoints_folder):
    image_files = sorted(f for f in os.listdir(images_folder) if f.endswith(".jpg"))
    samples = []
    metadata = []

    for img_name in image_files:
        img_path = Path(images_folder) / img_name
        json_path = Path(keypoints_folder) / img_name.replace(".jpg", ".json")

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        if not json_path.is_file():
            raise FileNotFoundError(f"Missing keypoints json: {json_path}")

        with open(json_path, "r", encoding="utf-8") as json_file:
            keypoints_data = json.load(json_file)

        keypoints_xyc = []
        negative_kps = []
        for entry in keypoints_data:
            if entry["is_target"]:
                keypoints_xyc.append(entry["keypoints"])
            else:
                negative_kps.append(entry["keypoints"])

        if len(keypoints_xyc) != 1:
            raise ValueError(f"Expected exactly one target keypoint set in {json_path}")

        keypoints_xyc = np.asarray(keypoints_xyc[0])
        negative_kps = np.asarray(negative_kps)
        samples.append(
            {
                "image": image,
                "keypoints_xyc": keypoints_xyc,
                "negative_kps": negative_kps,
            }
        )
        metadata.append(
            {
                "image_path": str(img_path),
                "keypoints_path": str(json_path),
                "image_height": image.shape[0],
                "image_width": image.shape[1],
                "image_channels": image.shape[2] if image.ndim == 3 else 1,
                "target_keypoints": int(keypoints_xyc.shape[0]),
                "keypoint_dims": int(keypoints_xyc.shape[1]) if keypoints_xyc.ndim == 2 else 0,
                "negative_person_prompts": int(negative_kps.shape[0]) if negative_kps.ndim > 0 else 0,
            }
        )

    return samples, metadata


def load_demo_samples(demo_dir, limit_samples):
    demo_dir = Path(demo_dir)
    all_samples = []
    all_metadata = []
    for group_dir in sorted(demo_dir.glob("group*")):
        images_folder = group_dir / "images"
        keypoints_folder = group_dir / "keypoints"
        if images_folder.is_dir() and keypoints_folder.is_dir():
            samples, metadata = load_kpr_samples(images_folder, keypoints_folder)
            all_samples.extend(samples)
            all_metadata.extend(metadata)

    if not all_samples:
        raise RuntimeError(f"No demo samples found under {demo_dir}")

    if limit_samples > 0:
        all_samples = all_samples[:limit_samples]
        all_metadata = all_metadata[:limit_samples]

    return all_samples, all_metadata


def summarize_inputs(metadata):
    sizes = sorted({f"{m['image_width']}x{m['image_height']}x{m['image_channels']}" for m in metadata})
    return {
        "input_samples": len(metadata),
        "samples_with_keypoints": sum(1 for m in metadata if m["target_keypoints"] > 0),
        "target_keypoints_total": sum(m["target_keypoints"] for m in metadata),
        "negative_person_prompts_total": sum(m["negative_person_prompts"] for m in metadata),
        "input_image_sizes": ";".join(sizes),
    }


def chunk_samples(samples, batch_size):
    if batch_size <= 0 or batch_size >= len(samples):
        return [samples]
    return [samples[i : i + batch_size] for i in range(0, len(samples), batch_size)]


def fresh_batch(batch):
    return [sample.copy() for sample in batch]


def stream_process_output(process):
    # Fold the monitor's stdout into the report log by writing through sys.stdout
    # (a TeeStream -> terminal + report.log).
    for line in process.stdout:
        sys.stdout.write(line)


def start_monitor(interval, duration, monitor_csv):
    monitor_script = ROOT_DIR / "tools/monitor-system.sh"
    monitor_duration = max(int(duration) + 5, int(duration))
    Path(monitor_csv).parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        ["bash", str(monitor_script), str(interval), str(monitor_duration), str(monitor_csv)],
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    thread = threading.Thread(target=stream_process_output, args=(process,), daemon=True)
    thread.start()
    return process


def stop_monitor(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except Exception:
        process.kill()


def parse_float(value):
    if value is None or value == "":
        return None
    return float(value)


def average(values):
    return sum(values) / len(values) if values else ""


def maximum(values):
    return max(values) if values else ""


def summarize_monitor_csv(monitor_csv):
    summary = {
        "monitor_samples": 0,
        "cpu_pct_avg": "",
        "cpu_pct_max": "",
        "mem_pct_avg": "",
        "mem_pct_max": "",
        "mem_used_gib_avg": "",
        "mem_used_gib_max": "",
        "swap_pct_avg": "",
        "swap_pct_max": "",
        "npu_pct_avg": "",
        "npu_pct_max": "",
    }
    if not monitor_csv or not Path(monitor_csv).is_file():
        return summary

    columns = {
        "cpu_pct": [],
        "mem_pct": [],
        "mem_used_gib": [],
        "swap_pct": [],
        "npu_pct": [],
    }
    with open(monitor_csv, "r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            summary["monitor_samples"] += 1
            for name in columns:
                value = parse_float(row.get(name))
                if value is not None:
                    columns[name].append(value)

    for name, values in columns.items():
        summary[f"{name}_avg"] = round(average(values), 4) if values else ""
        summary[f"{name}_max"] = round(maximum(values), 4) if values else ""
    return summary


def format_pct(value):
    if value is None or value == "":
        return ""
    return f"{value}%"


def percentile(values, pct):
    if not values:
        return ""
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, int(round((pct / 100.0) * (len(sorted_values) - 1)))))
    return sorted_values[index]


def shape_string(value):
    if value is None:
        return ""
    if hasattr(value, "shape"):
        return "x".join(str(dim) for dim in value.shape)
    return ""


def build_model_input_str(images, prompt_masks):
    parts = []
    if images is not None:
        parts.append(f"images:{shape_string(images)}")
    if prompt_masks is not None:
        parts.append(f"prompt_masks:{shape_string(prompt_masks)}")
    return ";".join(parts)


def build_model_output_str(embeddings, visibility_scores, parts_masks):
    return ";".join(
        [
            f"embeddings:{shape_string(embeddings)}",
            f"visibility_scores:{shape_string(visibility_scores)}",
            f"parts_masks:{shape_string(parts_masks)}",
        ]
    )


def build_summary_row(device, batch_size, model_input, model_output, fps, latency_ms_avg, monitor_summary):
    return {
        "device": device,
        "batch_size": batch_size,
        "model_input": model_input,
        "model_output": model_output,
        "fps": round(fps, 4) if fps != "" else "",
        "latency_ms_avg": round(latency_ms_avg, 4) if latency_ms_avg != "" else "",
        "cpu_pct_avg": format_pct(monitor_summary.get("cpu_pct_avg", "")),
        "cpu_pct_max": format_pct(monitor_summary.get("cpu_pct_max", "")),
        "npu_pct_avg": format_pct(monitor_summary.get("npu_pct_avg", "")),
        "npu_pct_max": format_pct(monitor_summary.get("npu_pct_max", "")),
        "mem_used_gib_max": monitor_summary.get("mem_used_gib_max", ""),
    }


def write_summary_csv(output_csv, row, append=False):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # When appending to a shared CSV, only write the header if the file is new.
    write_header = not (append and output_csv.is_file())
    mode = "a" if append and output_csv.is_file() else "w"
    with open(output_csv, mode, newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in SUMMARY_FIELDS})


def derive_paths(output_csv):
    output_csv = Path(output_csv)
    base = output_csv.with_suffix("")
    report_log = base.parent / f"{base.name}.report.log"
    return output_csv, report_log


def monitor_tmp_path(report_log):
    # Base the temp monitor CSV on the (per-run unique) report log name so that
    # parallel/sweep runs sharing one summary CSV never collide.
    return DEFAULT_RESULTS_DIR / ".monitor_tmp" / f"{Path(report_log).stem}.monitor.csv"


def run_timed_loop(extractor, sample_batches, duration):
    latencies = []
    timed_batches = 0
    timed_samples = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration:
        for batch in sample_batches:
            if time.perf_counter() - start >= duration:
                break
            batch_start = time.perf_counter()
            extractor(fresh_batch(batch))
            latencies.append(time.perf_counter() - batch_start)
            timed_batches += 1
            timed_samples += len(batch)
    actual_duration = time.perf_counter() - start
    return latencies, timed_batches, timed_samples, actual_duration


def report_and_write(
    device,
    batch_size,
    model_input,
    model_output,
    latencies,
    timed_batches,
    timed_samples,
    actual_duration,
    monitor_summary,
    output_csv,
    append=False,
):
    latency_ms = [value * 1000.0 for value in latencies]
    fps = timed_samples / actual_duration if actual_duration > 0 else 0
    latency_ms_avg = average(latency_ms) if latency_ms else ""

    row = build_summary_row(device, batch_size, model_input, model_output, fps, latency_ms_avg, monitor_summary)
    write_summary_csv(output_csv, row, append=append)

    # Detailed metrics -> report.log only (captured via TeeStream on sys.stdout).
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
    return row


def main():
    global cv2, np, torch, build_config, KPRFeatureExtractor

    args = parse_args()
    DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_csv = args.output_csv or str(DEFAULT_RESULTS_DIR / f"kpr_cpu_benchmark_{timestamp}.csv")
    report_log = args.log_file or ""
    output_csv, default_report_log = derive_paths(output_csv)
    report_log = Path(report_log) if report_log else default_report_log
    monitor_tmp = monitor_tmp_path(report_log)

    report_log.parent.mkdir(parents=True, exist_ok=True)
    report_log_file = open(report_log, "w", encoding="utf-8")
    sys.stdout = TeeStream(sys.stdout, report_log_file)
    sys.stderr = TeeStream(sys.stderr, report_log_file)
    print(f"Report log: {report_log}")

    import cv2
    import numpy as np
    import torch

    from torchreid.scripts.builder import build_config
    from torchreid.tools.feature_extractor import KPRFeatureExtractor

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    if args.torch_interop_threads > 0:
        torch.set_num_interop_threads(args.torch_interop_threads)

    samples, metadata = load_demo_samples(args.demo_dir, args.limit_samples)
    input_summary = summarize_inputs(metadata)
    sample_batches = chunk_samples(samples, args.batch_size)
    effective_batch = args.batch_size if args.batch_size > 0 else len(samples)

    cfg = build_config(config_path=args.config)
    cfg.use_gpu = False
    extractor = KPRFeatureExtractor(
        cfg,
        image_size=(cfg.data.height, cfg.data.width),
        pixel_mean=cfg.data.norm_mean,
        pixel_std=cfg.data.norm_std,
    )
    extractor.model.cpu()
    extractor.model.eval()

    print(f"config: {Path(args.config).resolve()}")
    print(f"weights: {Path(cfg.model.load_weights).resolve() if cfg.model.load_weights else ''}")
    print(f"torch_threads: {torch.get_num_threads()}  interop: {torch.get_num_interop_threads()}")
    print(f"model_input_size: {cfg.data.height}x{cfg.data.width}  input_modalities: image,keypoints_xyc,negative_kps")
    print(f"input_summary: {input_summary}")

    with torch.inference_mode():
        for _ in range(max(args.warmup_iters, 0)):
            for batch in sample_batches:
                extractor(fresh_batch(batch))
        _, probe_embeddings, probe_visibility_scores, probe_parts_masks = extractor(fresh_batch(sample_batches[0]))

    model_output = build_model_output_str(probe_embeddings, probe_visibility_scores, probe_parts_masks)
    model_input = f"images:{effective_batch}x3x{cfg.data.height}x{cfg.data.width}"

    monitor_process = None
    if not args.no_monitor:
        monitor_process = start_monitor(args.monitor_interval, args.duration, monitor_tmp)

    try:
        with torch.inference_mode():
            latencies, timed_batches, timed_samples, actual_duration = run_timed_loop(
                extractor, sample_batches, args.duration
            )
    finally:
        stop_monitor(monitor_process)

    monitor_summary = summarize_monitor_csv(monitor_tmp if not args.no_monitor else "")

    report_and_write(
        device="pytorch:CPU",
        batch_size=effective_batch,
        model_input=model_input,
        model_output=model_output,
        latencies=latencies,
        timed_batches=timed_batches,
        timed_samples=timed_samples,
        actual_duration=actual_duration,
        monitor_summary=monitor_summary,
        output_csv=output_csv,
        append=args.append,
    )

    # Only the summary CSV and report.log are kept as artifacts.
    try:
        Path(monitor_tmp).unlink(missing_ok=True)
    except Exception:
        pass

    print(f"Benchmark summary CSV: {output_csv}")
    print(f"Report log: {report_log}")


if __name__ == "__main__":
    main()
