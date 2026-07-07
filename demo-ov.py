#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import torch

from torchreid.metrics.distance import compute_distance_matrix_using_bp_features
from torchreid.scripts.builder import build_config
from torchreid.tools.openvino_feature_extractor import KPROpenVINOFeatureExtractor, load_demo_groups
from torchreid.utils.visualization.display_kpr_samples import display_kpr_reid_samples_grid, display_distance_matrix

DEFAULT_CONFIG = ROOT_DIR / "configs/kpr/imagenet/kpr_occ_posetrack_test.yaml"
DEFAULT_DEMO_DIR = ROOT_DIR / "assets/demo/soccer_players"
DEFAULT_FP16_MODEL = ROOT_DIR / "openvino_models/kpr_fp16/kpr.xml"
DEFAULT_FP16_INT8_MODEL = ROOT_DIR / "openvino_models/kpr_fp16_int8/kpr.xml"
DEFAULT_RESULTS_DIR = ROOT_DIR / "assets/demo/results"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the KPR demo with an OpenVINO IR model.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="KPR config yaml path.")
    parser.add_argument("--model", default="", help="OpenVINO .xml model path. Default follows --precision.")
    parser.add_argument(
        "--precision",
        choices=["fp16", "fp16-int8"],
        default="fp16",
        help="Default model precision when --model is not specified. Default: fp16.",
    )
    parser.add_argument("--device", default="CPU", help="OpenVINO device, e.g. CPU, GPU, NPU, AUTO. Default: CPU.")
    parser.add_argument("--batch-size", type=int, default=1, help="OpenVINO inference batch size. 0 runs each group as one batch.")
    parser.add_argument("--demo-dir", default=str(DEFAULT_DEMO_DIR), help="Demo dataset directory.")
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR), help="Directory for generated figures.")
    parser.add_argument("--display-mode", choices=["plot", "save"], default="save", help="Visualization mode.")
    return parser.parse_args()


def resolve_model_path(args):
    if args.model:
        return Path(args.model)
    if args.precision == "fp16-int8":
        return DEFAULT_FP16_INT8_MODEL
    return DEFAULT_FP16_MODEL


def main():
    args = parse_args()
    model_path = resolve_model_path(args)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"OpenVINO model not found: {model_path}\n"
            "Convert it first, for example:\n"
            "  python tools/convert_kpr_openvino.py --precision all"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kpr_cfg = build_config(config_path=args.config)
    kpr_cfg.use_gpu = False

    # NPU cannot compile the dynamic-shape IR, so it needs a fixed batch. Force
    # batch_size=1 and reshape the model to a static batch of 1 on NPU.
    batch_size = args.batch_size
    static_batch = None
    if args.device.upper() == "NPU":
        if batch_size != 1:
            print("NPU requires a fixed batch; forcing --batch-size 1 for the static-shape model.")
            batch_size = 1
        static_batch = 1

    extractor = KPROpenVINOFeatureExtractor(
        kpr_cfg,
        model_path=model_path,
        image_size=(kpr_cfg.data.height, kpr_cfg.data.width),
        pixel_mean=kpr_cfg.data.norm_mean,
        pixel_std=kpr_cfg.data.norm_std,
        device=args.device,
        batch_size=batch_size,
        static_batch=static_batch,
    )

    samples_grp_1, samples_grp_2 = load_demo_groups(args.demo_dir)
    display_kpr_reid_samples_grid(samples_grp_1 + samples_grp_2, display_mode=args.display_mode)

    samples_grp_1, embeddings_grp_1, visibility_scores_grp_1, parts_masks_grp_1 = extractor(samples_grp_1)
    samples_grp_2, embeddings_grp_2, visibility_scores_grp_2, parts_masks_grp_2 = extractor(samples_grp_2)

    display_kpr_reid_samples_grid(
        samples_grp_1 + samples_grp_2,
        display_mode=args.display_mode,
        save_path=str(output_dir / f"samples_grid_ov_{args.precision}_{args.device}.png"),
    )

    distance_matrix, body_parts_distmat = compute_distance_matrix_using_bp_features(
        embeddings_grp_1,
        embeddings_grp_2,
        visibility_scores_grp_1,
        visibility_scores_grp_2,
        use_gpu=False,
        use_logger=False,
    )
    distances = distance_matrix.cpu().detach().numpy() / 2

    display_distance_matrix(
        distances,
        samples_grp_1,
        samples_grp_2,
        display_mode=args.display_mode,
        save_path=str(output_dir / f"distance_matrix_ov_{args.precision}_{args.device}.png"),
    )
    print(f"OpenVINO demo completed with {model_path} on {args.device}")
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()