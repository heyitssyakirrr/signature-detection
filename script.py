"""
Signature Detection Model Test
tech4humans/yolov8s-signature-detector — CPU-only, offline, stand-alone test script.

See signature-detection-test-project.md / signature-detection-test-plan.md for full context.

Basic usage:
    python script.py

All tuning knobs are CLI flags (no code edits needed to re-tune):
    python script.py --conf-threshold 0.35 --iou 0.3 --imgsz 1280
    python script.py --target-min-dim 1000 --max-dim 3000
    python script.py --upload-dir /path/to/cheques --workers 4

Run `python script.py --help` for the full list.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Force fully-offline behaviour before any ML imports touch the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("sigtest")

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ==========================================================================
# Config
# ==========================================================================
@dataclass
class Config:
    """All tunable settings for one run. Everything here is CLI-configurable
    (see build_arg_parser) so re-tuning after reviewing results never requires
    editing this file — important once cheques start arriving from many
    sources at many resolutions, not just this fixed 10-image test set."""

    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    upload_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    model_path: Optional[Path] = None

    # Detection thresholds
    conf_logging_floor: float = 0.001         # log every raw detection down to this confidence
    conf_decision_threshold: float = 0.25    # detections at/above this count as "a signature"
    iou: float = 0.7                        # NMS IoU threshold

    # Inference resolution
    imgsz: int = 1280                   # passed straight to ultralytics; None -> model default
    head_format: str = "v8"             # v8(no objectness), v5(with objectness), auto
    debug_decode: bool = False          # emit raw/decode/NMS diagnostics
    debug_topk: int = 10                # top-k confidences to log when debug_decode is enabled

    # Adaptive pre-resize (applied BEFORE imgsz/inference), since real-world
    # cheques will arrive at wildly different native resolutions/DPI.
    # A cheque scanned at 300x150px loses stroke detail no matter what imgsz
    # you pass; upscaling it first towards target_min_dim recovers some of
    # that. A 6000x3000px phone photo, on the other hand, just wastes CPU
    # time and RAM being fed in full size, so it gets capped by max_dim.
    target_min_dim: int = 1000          # upscale so the shorter side reaches this, if smaller
    max_dim: int = 3000                 # downscale so the longer side never exceeds this
    max_upscale_factor: float = 4.0     # never blow up a tiny/garbage image beyond this factor

    # Throughput
    batch_size: int = 4                 # images per model.predict() call
    workers: int = 1                    # parallel worker processes for image I/O + preprocessing

    log_level: str = "INFO"

    def __post_init__(self):
        if self.upload_dir is None:
            self.upload_dir = self.base_dir.parent / "Upload"
        if self.output_dir is None:
            self.output_dir = self.base_dir.parent / "Output"
        if self.model_path is None:
            self.model_path = self.base_dir.parent / "Model" / "yolov8s_2.torchscript"

    @property
    def sig_images_dir(self) -> Path:
        return self.output_dir / "signature_images"

    @property
    def results_csv(self) -> Path:
        return self.output_dir / "results.csv"

    @property
    def diagnostics_csv(self) -> Path:
        return self.output_dir / "diagnostics.csv"


def build_arg_parser() -> argparse.ArgumentParser:
    # Get defaults from Config class to avoid duplication
    _defaults = Config()
    
    p = argparse.ArgumentParser(description="Signature detection batch test")
    p.add_argument("--upload-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--conf-floor", type=float, default=_defaults.conf_logging_floor, dest="conf_logging_floor",
                    help=f"Log all detections down to this confidence (default: {_defaults.conf_logging_floor})")
    p.add_argument("--conf-threshold", type=float, default=_defaults.conf_decision_threshold, dest="conf_decision_threshold",
                    help=f"Detections at/above this count as a real signature (default: {_defaults.conf_decision_threshold})")
    p.add_argument("--iou", type=float, default=_defaults.iou, help=f"NMS IoU threshold (default: {_defaults.iou})")
    p.add_argument("--imgsz", type=int, default=_defaults.imgsz, help=f"Inference resolution passed to YOLO (default: {_defaults.imgsz})")
    p.add_argument("--head-format", default=_defaults.head_format, choices=["v8", "v5", "auto"],
                    help="TorchScript head decode format: v8(no objectness), v5(with objectness), or auto")
    p.add_argument("--debug-decode", action="store_true",
                    help="Log raw tensor shape and confidence flow before/after NMS")
    p.add_argument("--debug-topk", type=int, default=_defaults.debug_topk,
                    help=f"How many top confidences to print in decode debug mode (default: {_defaults.debug_topk})")
    p.add_argument("--target-min-dim", type=int, default=_defaults.target_min_dim, dest="target_min_dim",
                    help=f"Upscale images whose shorter side is below this (default: {_defaults.target_min_dim}px)")
    p.add_argument("--max-dim", type=int, default=_defaults.max_dim, dest="max_dim",
                    help=f"Downscale images whose longer side exceeds this (default: {_defaults.max_dim}px)")
    p.add_argument("--max-upscale-factor", type=float, default=_defaults.max_upscale_factor, dest="max_upscale_factor",
                    help=f"Never blow up images beyond this factor (default: {_defaults.max_upscale_factor})")
    p.add_argument("--batch-size", type=int, default=_defaults.batch_size, dest="batch_size",
                    help=f"Images per model.predict() call (default: {_defaults.batch_size})")
    p.add_argument("--workers", type=int, default=_defaults.workers,
                    help=f"Parallel worker processes for image loading/preprocessing (default: {_defaults.workers})")
    p.add_argument("--log-level", default=_defaults.log_level, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def config_from_args(argv=None) -> Config:
    args = build_arg_parser().parse_args(argv)
    return Config(**{k: v for k, v in vars(args).items() if v is not None})


# ==========================================================================
# Image loading + adaptive preprocessing
# ==========================================================================
@dataclass
class PreparedImage:
    """Everything downstream needs for one image: the model-ready RGB image,
    plus enough metadata to explain what preprocessing was applied (this is
    what feeds diagnostics.csv)."""
    path: Path
    image: Image.Image          # final RGB image, resized, ready for inference
    original_size: tuple
    original_mode: str
    final_size: tuple
    scale_factor: float


def load_image_rgb(path: Path) -> Image.Image:
    """Load an image and return a 3-channel RGB PIL Image, regardless of
    source format or color mode (grayscale, CMYK, RGBA, paletted, etc.)."""
    img = Image.open(path)
    img.load()  # force decode now, so errors surface here rather than later
    if img.mode == "RGBA":
        # Flatten transparency onto white rather than silently dropping it,
        # to avoid introducing black backgrounds via naive .convert("RGB").
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img


def adaptive_resize(img: Image.Image, cfg: Config) -> tuple[Image.Image, float]:
    """Normalize resolution so the model sees a comparable amount of detail
    regardless of source scan quality:
      - shorter side below target_min_dim -> upscale (capped at max_upscale_factor)
      - longer side above max_dim -> downscale (keeps CPU/memory cost bounded)
      - otherwise -> left as-is
    Returns (resized_image, scale_factor_applied)."""
    w, h = img.size
    short_side, long_side = min(w, h), max(w, h)

    scale = 1.0
    if short_side < cfg.target_min_dim:
        scale = cfg.target_min_dim / short_side
        scale = min(scale, cfg.max_upscale_factor)
    elif long_side > cfg.max_dim:
        scale = cfg.max_dim / long_side

    if abs(scale - 1.0) < 1e-12:
        return img, 1.0

    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    resample = Image.LANCZOS if scale > 1 else Image.BOX
    resized = img.resize(new_size, resample)
    return resized, scale


def prepare_image(path: Path, cfg: Config) -> PreparedImage:
    # Peek at the source mode/size cheaply (PIL only reads the header here)
    # BEFORE converting to RGB, so diagnostics reflect the true source format
    # (grayscale, CMYK, paletted, etc.) rather than always showing "RGB".
    with Image.open(path) as probe:
        original_size, original_mode = probe.size, probe.mode

    img = load_image_rgb(path)
    resized, scale = adaptive_resize(img, cfg)

    return PreparedImage(
        path=path,
        image=resized,
        original_size=original_size,
        original_mode=original_mode,
        final_size=resized.size,
        scale_factor=scale,
    )


# ==========================================================================
# Inference
# ==========================================================================
def _letterbox_to_square(img: Image.Image, size: int) -> tuple[Image.Image, float, int, int]:
    """Resize with aspect-ratio preserved padding into a square model input."""
    src_w, src_h = img.size
    scale = min(size / src_w, size / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resample = Image.LANCZOS if scale > 1 else Image.BOX
    resized = img.resize((new_w, new_h), resample)

    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def _image_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _select_prediction_tensor(raw: Any) -> torch.Tensor:
    if isinstance(raw, torch.Tensor):
        return raw
    if isinstance(raw, dict):
        for key in ("pred", "predictions", "output", "outputs"):
            value = raw.get(key)
            if isinstance(value, torch.Tensor):
                return value
        for value in raw.values():
            if isinstance(value, torch.Tensor):
                return value
    if isinstance(raw, (list, tuple)):
        for value in raw:
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, (list, tuple, dict)):
                nested = _select_prediction_tensor(value)
                if isinstance(nested, torch.Tensor):
                    return nested
    raise RuntimeError(
        "Unable to find tensor predictions in model output. "
        "Expected torch.Tensor or container with a tensor output."
    )


def _boxes_xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = boxes.unbind(dim=1)
    half_w = w / 2
    half_h = h / 2
    return torch.stack((x_c - half_w, y_c - half_h, x_c + half_w, y_c + half_h), dim=1)


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-9)


def _nms_indices(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = _box_iou(boxes[i].unsqueeze(0), boxes[rest]).squeeze(0)
        order = rest[ious <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long)


def _topk_list(tensor: torch.Tensor, topk: int) -> list[float]:
    if tensor.numel() == 0:
        return []
    k = max(1, min(int(topk), int(tensor.numel())))
    vals = torch.topk(tensor, k=k).values
    return [round(float(v), 6) for v in vals.tolist()]


def _normalize_prediction_layout(pred: torch.Tensor) -> torch.Tensor:
    """Normalize model output layout to [B, N, C].

    Common exports:
      - post-NMS: [B, 300, 6]
      - raw YOLOv8: [B, 84, N] or [B, N, 84]
    """
    if pred.dim() != 3:
        return pred

    b, d1, d2 = pred.shape
    del b  # only used for readability

    # Already [B, N, C] for post-NMS or decoded outputs.
    if d2 in (5, 6, 7, 84, 85) and d1 > d2:
        return pred

    # Typical raw-head export layout [B, C, N] where C is small.
    # Include single-class YOLOv8 exports where C=5 (x,y,w,h,cls).
    if d1 in (5, 6, 7, 84, 85) and d2 > d1:
        return pred.transpose(1, 2)

    # Generic safeguard: channel-first layout normally has a tiny channel
    # dimension (<=8) and a very large candidate dimension.
    if d1 <= 8 and d2 > d1:
        return pred.transpose(1, 2)

    # Fallback heuristic from earlier implementation.
    if d2 < 6 and d1 >= 6:
        return pred.transpose(1, 2)

    return pred


def _postprocess_single_prediction(
    pred: torch.Tensor,
    conf_floor: float,
    iou: float,
    imgsz: int,
    head_format: str,
    debug_decode: bool,
    debug_topk: int,
    image_idx: int,
) -> torch.Tensor:
    # Normalize to [N, C]
    if pred.dim() != 2:
        raise RuntimeError(f"Expected per-image prediction tensor with 2 dims, got {tuple(pred.shape)}")
    # If tensor is channel-first [C, N], transpose to [N, C].
    # Keep single-class [N, 5] tensors untouched.
    if pred.shape[0] <= 8 and pred.shape[1] > pred.shape[0]:
        pred = pred.transpose(0, 1)

    if pred.shape[-1] < 5:
        raise RuntimeError(
            f"Unsupported model output shape {tuple(pred.shape)}. "
            "Expected [...,5] (single-class raw), [...,6] (post-NMS), or [...,5+num_classes]."
        )

    decode_mode_used = "direct"

    if pred.shape[-1] == 6:
        boxes = pred[:, :4]
        scores = pred[:, 4]
        cls = pred[:, 5]
    else:
        boxes = pred[:, :4]
        decode_mode = head_format
        if decode_mode == "auto":
            # Common YOLOv8 export is [x,y,w,h,cls1..clsN] (no objectness).
            # If channels are 5 or 84, prefer v8-style decode.
            decode_mode = "v8" if pred.shape[-1] in (5, 84) else "v5"
        decode_mode_used = decode_mode

        if decode_mode == "v8":
            cls_scores = pred[:, 4:]
            if cls_scores.min() < 0 or cls_scores.max() > 1:
                cls_scores = cls_scores.sigmoid()
            scores, cls_idx = cls_scores.max(dim=1)
            cls = cls_idx.to(dtype=pred.dtype)
        else:
            obj = pred[:, 4]
            cls_scores = pred[:, 5:]

            if obj.min() < 0 or obj.max() > 1:
                obj = obj.sigmoid()

            if cls_scores.shape[1] == 0:
                scores = obj
                cls = torch.zeros_like(scores)
            else:
                if cls_scores.min() < 0 or cls_scores.max() > 1:
                    cls_scores = cls_scores.sigmoid()
                cls_conf, cls_idx = cls_scores.max(dim=1)
                scores = obj * cls_conf
                cls = cls_idx.to(dtype=pred.dtype)

        # YOLO family outputs are commonly xywh before decode/export.
        # If x2<=x1/y2<=y1 appears often, treat first 4 values as xywh.
        invalid_xyxy = ((boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])).float().mean()
        if float(invalid_xyxy) > 0.1:
            boxes = _boxes_xywh_to_xyxy(boxes)

    # If coordinates are normalized, map them into model-input pixel space.
    if boxes.numel() and float(boxes.abs().max()) <= 2.0:
        boxes = boxes * float(imgsz)

    if debug_decode:
        logger.info(
            "[decode] image=%d mode=%s pred_shape=%s pre_filter_boxes=%d topk_pre=%s",
            image_idx,
            decode_mode_used,
            tuple(pred.shape),
            int(scores.numel()),
            _topk_list(scores, debug_topk),
        )

    keep_conf = scores >= conf_floor
    boxes = boxes[keep_conf]
    scores = scores[keep_conf]
    cls = cls[keep_conf]

    if debug_decode:
        logger.info(
            "[decode] image=%d conf_floor=%.4f kept_after_floor=%d topk_after_floor=%s",
            image_idx,
            conf_floor,
            int(scores.numel()),
            _topk_list(scores, debug_topk),
        )

    keep = _nms_indices(boxes, scores, iou)
    boxes = boxes[keep]
    scores = scores[keep]
    cls = cls[keep]

    if debug_decode:
        logger.info(
            "[decode] image=%d iou=%.3f kept_after_nms=%d topk_after_nms=%s",
            image_idx,
            iou,
            int(scores.numel()),
            _topk_list(scores, debug_topk),
        )

    return torch.cat((boxes, scores.unsqueeze(1), cls.unsqueeze(1)), dim=1) if boxes.numel() else boxes.new_zeros((0, 6))


def _decode_predictions(raw: Any, cfg: Config) -> list[torch.Tensor]:
    pred = _select_prediction_tensor(raw)

    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
    elif pred.dim() != 3:
        raise RuntimeError(
            f"Unsupported model output rank {pred.dim()} with shape {tuple(pred.shape)}. "
            "Expected rank-2 or rank-3 tensor."
        )

    pred = _normalize_prediction_layout(pred)

    if cfg.debug_decode:
        logger.info("[decode] batch_pred_shape=%s head_format=%s", tuple(pred.shape), cfg.head_format)

    return [
        _postprocess_single_prediction(
            p,
            cfg.conf_logging_floor,
            cfg.iou,
            cfg.imgsz,
            cfg.head_format,
            cfg.debug_decode,
            cfg.debug_topk,
            idx,
        )
        for idx, p in enumerate(pred)
    ]


def run_inference_batch(model, prepared_images: list[PreparedImage], cfg: Config):
    """Run TorchScript model on preprocessed images and return detection lists.
    Expected model artifact: a TorchScript-exported YOLO-like detector."""
    batch_tensors = []
    metas = []

    for prep in prepared_images:
        model_img, ratio, pad_x, pad_y = _letterbox_to_square(prep.image, cfg.imgsz)
        batch_tensors.append(_image_to_tensor(model_img))
        metas.append((ratio, pad_x, pad_y, prep.image.width, prep.image.height))

    inputs = torch.stack(batch_tensors, dim=0)
    with torch.inference_mode():
        raw = model(inputs)

    if cfg.debug_decode:
        logger.info("[decode] model_output_type=%s", type(raw).__name__)

    decoded = _decode_predictions(raw, cfg)

    all_detections = []
    for per_image, (ratio, pad_x, pad_y, src_w, src_h) in zip(decoded, metas):
        detections = []
        for row in per_image:
            x1, y1, x2, y2, conf, _cls = row.tolist()
            x1 = (x1 - pad_x) / ratio
            y1 = (y1 - pad_y) / ratio
            x2 = (x2 - pad_x) / ratio
            y2 = (y2 - pad_y) / ratio

            x1 = max(0.0, min(float(src_w), x1))
            y1 = max(0.0, min(float(src_h), y1))
            x2 = max(0.0, min(float(src_w), x2))
            y2 = max(0.0, min(float(src_h), y2))

            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({"confidence": float(conf), "box": (x1, y1, x2, y2)})

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        all_detections.append(detections)

    return all_detections


def load_torchscript_model(path: Path):
    # Some TorchScript exports (nms=True) embed torchvision::nms. Importing
    # torchvision here registers the custom op in many environments.
    try:
        import torchvision  # noqa: F401
    except Exception:
        pass

    try:
        model = torch.jit.load(str(path), map_location="cpu")
    except Exception as exc:
        if "torchvision::nms" in str(exc):
            raise RuntimeError(
                "TorchScript model requires torchvision::nms but your server "
                "does not provide that operator. Either install a torchvision "
                "build compatible with your torch version, or re-export the "
                "model with nms=False and use this script's Python NMS path."
            ) from exc
        raise RuntimeError(
            "Failed to load model as TorchScript. This script does not use ultralytics. "
            "Please provide a TorchScript-exported YOLO model (e.g. exported with format='torchscript')."
        ) from exc
    model.eval()
    return model


# ==========================================================================
# Thresholding / box-size logging
# ==========================================================================
def compute_box_size(box, image_size):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    img_w, img_h = image_size
    area_pct = (w * h) / (img_w * img_h) * 100 if img_w and img_h else 0.0
    return {"width": round(w, 1), "height": round(h, 1), "area_pct": round(area_pct, 3)}


def split_by_threshold(detections, cfg: Config):
    decision = [d for d in detections if d["confidence"] >= cfg.conf_decision_threshold]
    return decision, detections


# ==========================================================================
# Annotation drawing
# ==========================================================================
def draw_annotations(image: Image.Image, detections, cfg: Config) -> Image.Image:
    """Green/solid = at or above the decision threshold (counted as a real
    signature). Orange/thin = below threshold but above the logging floor —
    kept visible so borderline misses are easy to spot during review."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for d in detections:
        x1, y1, x2, y2 = d["box"]
        conf = d["confidence"]
        is_decision = conf >= cfg.conf_decision_threshold
        color = (0, 200, 0) if is_decision else (255, 140, 0)
        width = 3 if is_decision else 2
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        label = f"{conf:.2f}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        label_y = max(0, y1 - text_h - 4)
        draw.rectangle([x1, label_y, x1 + text_w + 6, label_y + text_h + 4], fill=color)
        draw.text((x1 + 3, label_y + 1), label, fill=(255, 255, 255), font=font)

    return annotated


# ==========================================================================
# Cropping
# ==========================================================================
def crop_signatures(image: Image.Image, decision_detections):
    crops = []
    for d in decision_detections:
        x1, y1, x2, y2 = d["box"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        crops.append(image.crop((x1, y1, x2, y2)))
    return crops


# ==========================================================================
# Per-image result recording
# ==========================================================================
def build_result_row(filename, decision_detections, all_detections, image_size):
    box_sizes = [compute_box_size(d["box"], image_size) for d in all_detections]
    return {
        "filename": filename,
        "signature": len(decision_detections) > 0,
        "num_signatures": len(decision_detections),
        "confidence_scores": [round(d["confidence"], 4) for d in all_detections],
        "box_sizes": [f"{bs['width']}x{bs['height']}px ({bs['area_pct']}%)" for bs in box_sizes],
    }


def build_diagnostics_row(prep: PreparedImage, elapsed_ms: float):
    return {
        "filename": prep.path.name,
        "original_size": f"{prep.original_size[0]}x{prep.original_size[1]}",
        "original_mode": prep.original_mode,
        "resized_size": f"{prep.final_size[0]}x{prep.final_size[1]}",
        "scale_factor": round(prep.scale_factor, 3),
        "processing_ms": round(elapsed_ms, 1),
    }


# ==========================================================================
# Per-image output writing (post-inference)
# ==========================================================================
def write_image_outputs(prep: PreparedImage, detections, cfg: Config):
    """Draw annotations + save crops for one image's detections. Runs on the
    preprocessed (resized) image, since that's the pixel space the boxes are
    in and it typically has equal-or-better detail than the raw source."""
    stem = prep.path.stem
    out_dir = cfg.sig_images_dir / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    decision_detections, all_detections = split_by_threshold(detections, cfg)

    annotated = draw_annotations(prep.image, all_detections, cfg)
    annotated.save(out_dir / f"{stem}_annotated.png")

    for i, crop in enumerate(crop_signatures(prep.image, decision_detections), start=1):
        crop.save(out_dir / f"{stem}_sig{i}.png")

    return build_result_row(prep.path.name, decision_detections, all_detections, prep.image.size)


# ==========================================================================
# Batch orchestration
# ==========================================================================
def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def prepare_images_parallel(paths: list[Path], cfg: Config) -> list[PreparedImage]:
    """Image loading + resizing is I/O/CPU-bound but independent per file —
    safe and worthwhile to parallelize with --workers as volume grows, while
    keeping model inference itself sequential (one shared model instance,
    called in batches) to avoid loading multiple copies of the model into
    memory."""
    prepared = [None] * len(paths)
    if cfg.workers <= 1:
        for i, path in enumerate(paths):
            try:
                prepared[i] = prepare_image(path, cfg)
            except Exception:
                logger.exception("Failed to load/preprocess %s — skipping", path.name)
        return [p for p in prepared if p is not None]

    with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {pool.submit(prepare_image, path, cfg): i for i, path in enumerate(paths)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                prepared[i] = future.result()
            except Exception:
                logger.exception("Failed to load/preprocess %s — skipping", paths[i].name)
    return [p for p in prepared if p is not None]


def run(cfg: Config):
    if not cfg.model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {cfg.model_path}. "
            "Provide a TorchScript model file for offline inference."
        )

    cfg.sig_images_dir.mkdir(parents=True, exist_ok=True)

    model = load_torchscript_model(cfg.model_path)

    image_paths = sorted(
        p for p in cfg.upload_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {cfg.upload_dir}")

    logger.info("Found %d images in %s", len(image_paths), cfg.upload_dir)

    result_rows = []
    diagnostics_rows = []

    for batch_paths in chunked(image_paths, cfg.batch_size):
        prepared = prepare_images_parallel(batch_paths, cfg)
        if not prepared:
            continue

        t0 = time.perf_counter()
        detections_batch = run_inference_batch(model, prepared, cfg)
        elapsed_ms_total = (time.perf_counter() - t0) * 1000
        elapsed_ms_each = elapsed_ms_total / len(prepared)

        for prep, detections in zip(prepared, detections_batch):
            row = write_image_outputs(prep, detections, cfg)
            result_rows.append(row)
            diagnostics_rows.append(build_diagnostics_row(prep, elapsed_ms_each))
            logger.info(
                "%s -> signature=%s num_signatures=%d confidences=%s (scale=%.2fx)",
                row["filename"], row["signature"], row["num_signatures"],
                row["confidence_scores"], prep.scale_factor,
            )

    _write_csv(cfg.results_csv,
               ["filename", "signature", "num_signatures", "confidence_scores", "box_sizes"],
               result_rows)
    _write_csv(cfg.diagnostics_csv,
               ["filename", "original_size", "original_mode", "resized_size", "scale_factor", "processing_ms"],
               diagnostics_rows)

    logger.info("Done. Wrote %d rows to %s and %s", len(result_rows), cfg.results_csv, cfg.diagnostics_csv)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv=None):
    cfg = config_from_args(argv)
    logging.basicConfig(level=getattr(logging, cfg.log_level), format="%(message)s")
    run(cfg)


if __name__ == "__main__":
    main()