#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import torch

from torchreid.scripts.builder import build_config, build_model
from torchreid.data.datasets.keypoints_to_masks import KeypointsToMasks
from torchreid.data.transforms import build_transforms
from torchreid.tools.openvino_feature_extractor import (
    KPROpenVINOExportWrapper,
    KPROpenVINOFeatureExtractor,
    get_prompt_mask_channels,
    load_demo_groups,
)

DEFAULT_CONFIG = ROOT_DIR / "configs/kpr/imagenet/kpr_occ_posetrack_test.yaml"
DEFAULT_DEMO_DIR = ROOT_DIR / "assets/demo/soccer_players"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "openvino_models"


def parse_args():
    parser = argparse.ArgumentParser(description="Convert KPR PyTorch weights to OpenVINO IR FP16 and FP16-INT8.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="KPR config yaml path.")
    parser.add_argument("--weights", default="", help="Optional PyTorch checkpoint path overriding config model.load_weights.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for converted OpenVINO models.")
    parser.add_argument(
        "--precision",
        choices=["fp16", "fp16-int8", "all"],
        default="all",
        help="Model precision to export. Default: all.",
    )
    parser.add_argument("--demo-dir", default=str(DEFAULT_DEMO_DIR), help="Demo dataset used for INT8 calibration.")
    parser.add_argument("--calib-samples", type=int, default=0, help="Limit INT8 calibration samples. 0 uses all demo samples.")
    parser.add_argument("--batch-size", type=int, default=1, help="Calibration batch size for FP16-INT8 export.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing converted models.")
    return parser.parse_args()


def require_openvino():
    try:
        import openvino as ov
    except ImportError as exc:
        raise SystemExit("OpenVINO is not installed. Install OpenVINO 2026.1 before converting models.") from exc
    return ov


def require_nncf():
    try:
        import nncf
    except ImportError as exc:
        raise SystemExit("NNCF is not installed. Install nncf before exporting FP16-INT8 models.") from exc
    return nncf


def load_cfg(config_path, weights_path):
    cfg = build_config(config_path=config_path)
    cfg.use_gpu = False
    if weights_path:
        if not Path(weights_path).is_file():
            raise FileNotFoundError(f"PyTorch checkpoint not found: {weights_path}")
        cfg.model.load_weights = weights_path
    return cfg


def build_export_wrapper(cfg):
    model = build_model(cfg)
    model.eval()
    return KPROpenVINOExportWrapper(model, cfg).eval()


def make_example_inputs(cfg):
    height, width = cfg.data.height, cfg.data.width
    prompt_channels = get_prompt_mask_channels(cfg)
    images = torch.randn(1, 3, height, width, dtype=torch.float32)
    prompt_masks = torch.zeros(1, prompt_channels, height, width, dtype=torch.float32)
    if not cfg.model.promptable_trans.no_background_token:
        prompt_masks[:, 0] = 1.0
    return images, prompt_masks


def set_tensor_names(port, names):
    try:
        port.get_tensor().set_names(names)
    except AttributeError:
        port.tensor.set_names(names)


def set_model_io_names(ov_model):
    set_tensor_names(ov_model.inputs[0], {"images"})
    set_tensor_names(ov_model.inputs[1], {"prompt_masks"})
    for output, name in zip(ov_model.outputs, ["embeddings", "visibility_scores", "parts_masks"]):
        set_tensor_names(output, {name})


def convert_to_openvino(ov, wrapper, example_inputs, cfg):
    prompt_channels = get_prompt_mask_channels(cfg)
    input_shapes = [
        ("images", [-1, 3, cfg.data.height, cfg.data.width]),
        ("prompt_masks", [-1, prompt_channels, cfg.data.height, cfg.data.width]),
    ]
    with torch.no_grad():
        ov_model = ov.convert_model(wrapper, example_input=example_inputs, input=input_shapes)
    set_model_io_names(ov_model)
    return ov_model


def save_fp16_model(ov, ov_model, output_dir, force):
    fp16_dir = output_dir / "kpr_fp16"
    fp16_xml = fp16_dir / "kpr.xml"
    if fp16_xml.exists() and not force:
        print(f"Skip existing FP16 model: {fp16_xml}")
        return fp16_xml
    fp16_dir.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, str(fp16_xml), compress_to_fp16=True)
    print(f"Saved FP16 OpenVINO model: {fp16_xml}")
    return fp16_xml


def iter_batches(samples, batch_size):
    if batch_size <= 0:
        batch_size = len(samples)
    for start in range(0, len(samples), batch_size):
        yield samples[start : start + batch_size]


def build_calibration_dataset(nncf, cfg, demo_dir, limit_samples, batch_size):
    samples_grp_1, samples_grp_2 = load_demo_groups(demo_dir)
    samples = samples_grp_1 + samples_grp_2
    if limit_samples > 0:
        samples = samples[:limit_samples]
    if not samples:
        raise RuntimeError(f"No calibration samples found under {demo_dir}")

    extractor = KPROpenVINOFeatureExtractor.__new__(KPROpenVINOFeatureExtractor)
    extractor.cfg = cfg
    extractor.prompt_mask_channels = get_prompt_mask_channels(cfg)
    _, extractor.preprocess, extractor.target_preprocess, extractor.prompt_preprocess = build_transforms(
        cfg.data.height,
        cfg.data.width,
        cfg,
        transforms=None,
        norm_mean=cfg.data.norm_mean,
        norm_std=cfg.data.norm_std,
        masks_preprocess=cfg.model.kpr.masks.preprocess,
        softmax_weight=cfg.model.kpr.masks.softmax_weight,
        background_computation_strategy=cfg.model.kpr.masks.background_computation_strategy,
        mask_filtering_threshold=cfg.model.kpr.masks.mask_filtering_threshold,
    )
    extractor.keypoints_to_prompt_masks = KeypointsToMasks(
        mode=cfg.model.kpr.keypoints.prompt_masks,
        vis_thresh=cfg.model.kpr.keypoints.vis_thresh,
        vis_continous=cfg.model.kpr.keypoints.vis_continous,
    )
    extractor.keypoints_to_target_masks = KeypointsToMasks(
        mode=cfg.model.kpr.keypoints.target_masks,
        vis_thresh=cfg.model.kpr.keypoints.vis_thresh,
        vis_continous=False,
    )

    calibration_data = []
    for batch in iter_batches(samples, batch_size):
        images, prompt_masks = extractor.preprocess_samples(batch)
        calibration_data.append(
            {
                "images": images.detach().cpu().numpy(),
                "prompt_masks": prompt_masks.detach().cpu().numpy(),
            }
        )
    return nncf.Dataset(calibration_data)


def save_fp16_int8_model(ov, nncf, ov_model, cfg, args, output_dir):
    int8_dir = output_dir / "kpr_fp16_int8"
    int8_xml = int8_dir / "kpr.xml"
    if int8_xml.exists() and not args.force:
        print(f"Skip existing FP16-INT8 model: {int8_xml}")
        return int8_xml
    int8_dir.mkdir(parents=True, exist_ok=True)
    calibration_dataset = build_calibration_dataset(
        nncf, cfg, args.demo_dir, args.calib_samples, args.batch_size
    )
    quantized_model = nncf.quantize(ov_model, calibration_dataset)
    set_model_io_names(quantized_model)
    ov.save_model(quantized_model, str(int8_xml), compress_to_fp16=True)
    print(f"Saved FP16-INT8 OpenVINO model: {int8_xml}")
    return int8_xml


def main():
    args = parse_args()
    ov = require_openvino()
    output_dir = Path(args.output_dir)
    cfg = load_cfg(args.config, args.weights)
    wrapper = build_export_wrapper(cfg)
    example_inputs = make_example_inputs(cfg)
    ov_model = convert_to_openvino(ov, wrapper, example_inputs, cfg)

    if args.precision in ("fp16", "all"):
        save_fp16_model(ov, ov_model, output_dir, args.force)
    if args.precision in ("fp16-int8", "all"):
        nncf = require_nncf()
        save_fp16_int8_model(ov, nncf, ov_model, cfg, args, output_dir)


if __name__ == "__main__":
    main()