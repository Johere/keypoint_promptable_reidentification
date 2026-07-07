from __future__ import absolute_import

import copy
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as T

from torchreid.data import ImageDataset
from torchreid.data.datasets.keypoints_to_masks import KeypointsToMasks
from torchreid.data.transforms import build_transforms
from torchreid.utils.tools import extract_test_embeddings


def get_prompt_mask_channels(cfg):
    channels = cfg.model.kpr.masks.prompt_parts_num
    if not cfg.model.promptable_trans.no_background_token:
        channels += 1
    if cfg.model.kpr.keypoints.use_negative_keypoints:
        channels += 1
    return channels


class KPROpenVINOExportWrapper(nn.Module):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg

    def forward(self, images, prompt_masks):
        model_output = self.model(images=images, prompt_masks=prompt_masks)
        embeddings, visibility_scores, parts_masks, _ = extract_test_embeddings(
            model_output, self.cfg.model.kpr.test_embeddings
        )
        if self.cfg.test.normalize_feature:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings, visibility_scores, parts_masks


class KPROpenVINOFeatureExtractor(object):
    def __init__(
        self,
        cfg,
        model_path,
        image_size=(256, 128),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        device="CPU",
        ov_config=None,
        batch_size=1,
        static_batch=None,
        cache_dir=None,
    ):
        try:
            import openvino as ov
        except ImportError as exc:
            raise ImportError(
                "OpenVINO is required for demo-ov.py. Install OpenVINO 2026.1 in the active environment."
            ) from exc

        self.cfg = cfg
        self.device = device
        self.batch_size = batch_size
        self.static_batch = static_batch if static_batch and static_batch > 0 else None
        self.prompt_mask_channels = get_prompt_mask_channels(cfg)

        _, self.preprocess, self.target_preprocess, self.prompt_preprocess = build_transforms(
            image_size[0],
            image_size[1],
            cfg,
            transforms=None,
            norm_mean=pixel_mean,
            norm_std=pixel_std,
            masks_preprocess=cfg.model.kpr.masks.preprocess,
            softmax_weight=cfg.model.kpr.masks.softmax_weight,
            background_computation_strategy=cfg.model.kpr.masks.background_computation_strategy,
            mask_filtering_threshold=cfg.model.kpr.masks.mask_filtering_threshold,
        )

        self.keypoints_to_prompt_masks = KeypointsToMasks(
            mode=cfg.model.kpr.keypoints.prompt_masks,
            vis_thresh=cfg.model.kpr.keypoints.vis_thresh,
            vis_continous=cfg.model.kpr.keypoints.vis_continous,
        )
        self.keypoints_to_target_masks = KeypointsToMasks(
            mode=cfg.model.kpr.keypoints.target_masks,
            vis_thresh=cfg.model.kpr.keypoints.vis_thresh,
            vis_continous=False,
        )
        self.to_pil = T.ToPILImage()

        core = ov.Core()
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"OpenVINO model file not found: {model_path}")
        self.ov_model = core.read_model(str(model_path))

        # The exported IR has a dynamic batch dimension. NPU cannot compile dynamic
        # shapes, so reshape to a fixed batch before compiling when static_batch is set.
        if self.static_batch is not None:
            self._reshape_static(image_size)

        compile_config = dict(ov_config or {})
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            compile_config.setdefault("CACHE_DIR", str(cache_dir))
        self.compiled_model = core.compile_model(self.ov_model, device, compile_config)

        self.image_input_name, self.prompt_input_name = self._resolve_input_names()
        self.output_names = self._resolve_output_names()

    def _reshape_static(self, image_size):
        height, width = image_size
        images_port, prompt_port = self.ov_model.inputs[0], self.ov_model.inputs[1]
        self.ov_model.reshape(
            {
                images_port: [self.static_batch, 3, height, width],
                prompt_port: [self.static_batch, self.prompt_mask_channels, height, width],
            }
        )

    def _resolve_input_names(self):
        inputs = self.compiled_model.inputs
        if len(inputs) != 2:
            raise RuntimeError(f"Expected 2 OpenVINO inputs, got {len(inputs)}")

        names = []
        for input_port in inputs:
            try:
                any_name = input_port.get_any_name()
            except RuntimeError:
                any_name = ""
            names.append(any_name)
        if "images" in names and "prompt_masks" in names:
            return "images", "prompt_masks"
        return names[0], names[1]

    def _resolve_output_names(self):
        names = []
        for output_port in self.compiled_model.outputs:
            try:
                names.append(output_port.get_any_name())
            except RuntimeError:
                names.append(None)
        return names

    def preprocess_samples(self, samples):
        samples = [samples] if isinstance(samples, dict) else samples
        batch = {"image": [], "prompt_masks": []}

        for sample in samples:
            preprocessed_sample = ImageDataset.getitem(
                copy.deepcopy(sample),
                self.cfg,
                self.keypoints_to_prompt_masks,
                self.prompt_preprocess,
                self.keypoints_to_target_masks,
                self.target_preprocess,
                self.preprocess,
                load_masks=True,
            )
            batch["image"].append(preprocessed_sample["image"])
            if "prompt_masks" in preprocessed_sample:
                batch["prompt_masks"].append(preprocessed_sample["prompt_masks"])

        images = torch.stack(batch["image"], dim=0)
        if len(batch["prompt_masks"]) > 0:
            prompt_masks = torch.stack(batch["prompt_masks"], dim=0)
        else:
            prompt_masks = self._empty_prompt_masks(images.shape[0], images.shape[2], images.shape[3])
        return images, prompt_masks

    def _pad_to_static(self, images, prompt_masks, real_batch):
        pad_count = self.static_batch - real_batch
        image_pad = images[-1:].repeat(pad_count, 1, 1, 1)
        prompt_pad = prompt_masks[-1:].repeat(pad_count, 1, 1, 1)
        images = torch.cat([images, image_pad], dim=0)
        prompt_masks = torch.cat([prompt_masks, prompt_pad], dim=0)
        return images, prompt_masks

    def _empty_prompt_masks(self, batch_size, height, width):
        prompt_masks = torch.zeros((batch_size, self.prompt_mask_channels, height, width), dtype=torch.float32)
        if not self.cfg.model.promptable_trans.no_background_token:
            prompt_masks[:, 0] = 1.0
        return prompt_masks

    def __call__(self, input):
        samples = [input] if isinstance(input, dict) else input
        if self.batch_size and self.batch_size > 0 and len(samples) > self.batch_size:
            updated_samples = []
            embeddings_batches = []
            visibility_batches = []
            parts_masks_batches = []
            for start in range(0, len(samples), self.batch_size):
                batch = samples[start : start + self.batch_size]
                batch_samples, batch_embeddings, batch_visibility, batch_parts_masks = self._infer_batch(batch)
                updated_samples.extend(batch_samples)
                embeddings_batches.append(batch_embeddings)
                visibility_batches.append(batch_visibility)
                parts_masks_batches.append(batch_parts_masks)
            return (
                updated_samples,
                torch.cat(embeddings_batches, dim=0),
                torch.cat(visibility_batches, dim=0),
                torch.cat(parts_masks_batches, dim=0),
            )

        return self._infer_batch(samples)

    def _infer_batch(self, samples):
        updated_samples = copy.deepcopy(samples)
        images, prompt_masks = self.preprocess_samples(samples)
        real_batch = images.shape[0]

        # With a static (reshaped) model the compiled network only accepts exactly
        # static_batch rows. Pad short batches by repeating the last row, then slice
        # the outputs back to the real batch length.
        if self.static_batch is not None and real_batch < self.static_batch:
            images, prompt_masks = self._pad_to_static(images, prompt_masks, real_batch)

        infer_inputs = {
            self.image_input_name: images.detach().cpu().numpy(),
            self.prompt_input_name: prompt_masks.detach().cpu().numpy(),
        }
        ov_outputs = self.compiled_model(infer_inputs)
        embeddings, visibility_scores, parts_masks = self._unpack_outputs(ov_outputs)

        if self.static_batch is not None and real_batch < self.static_batch:
            embeddings = embeddings[:real_batch]
            visibility_scores = visibility_scores[:real_batch]
            parts_masks = parts_masks[:real_batch]

        for index in range(len(updated_samples)):
            updated_samples[index]["embeddings"] = embeddings[index].detach().cpu().numpy()
            updated_samples[index]["visibility_scores"] = visibility_scores[index].detach().cpu().numpy()
            updated_samples[index]["parts_masks"] = parts_masks[index].detach().cpu().numpy()

        return updated_samples, embeddings, visibility_scores, parts_masks

    def _unpack_outputs(self, ov_outputs):
        outputs = []
        for index, output_port in enumerate(self.compiled_model.outputs):
            output = ov_outputs[output_port]
            tensor = torch.from_numpy(np.asarray(output))
            outputs.append(tensor)
        if len(outputs) != 3:
            raise RuntimeError(f"Expected 3 OpenVINO outputs, got {len(outputs)}")
        return outputs[0], outputs[1], outputs[2]


def load_kpr_samples(images_folder, keypoints_folder):
    image_files = sorted(f for f in os.listdir(images_folder) if f.endswith(".jpg"))
    samples = []

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

        samples.append(
            {
                "image": image,
                "keypoints_xyc": np.asarray(keypoints_xyc[0]),
                "negative_kps": np.asarray(negative_kps),
            }
        )

    return samples


def load_demo_groups(demo_dir):
    demo_dir = Path(demo_dir)
    group1_folder = demo_dir / "group1"
    group2_folder = demo_dir / "group2"
    samples_grp_1 = load_kpr_samples(group1_folder / "images", group1_folder / "keypoints")
    samples_grp_2 = load_kpr_samples(group2_folder / "images", group2_folder / "keypoints")
    return samples_grp_1, samples_grp_2