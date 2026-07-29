"""
Signature Detection Model Test
mdefrance/yolos-tiny-signature-detection — CPU-only, offline, stand-alone test script.

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
from typing import Optional

# Fully offline: YOLOS is a pure Vision Transformer (no separate CNN backbone
# to resolve, unlike Conditional-DETR/ResNet), so transformers never needs
# to make any network call to load it. Safe to enforce strict offline mode.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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
    conf_logging_floor: float = 0.1     # log every raw detection down to this confidence
    conf_decision_threshold: float = 0.5  # detections at/above this count as "a signature"
    iou: float = 0.7                    # NMS IoU threshold

    # Inference resolution
    imgsz: int = 960                    # passed straight to ultralytics; None -> model default

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
            self.upload_dir = self.base_dir / "upload"
        if self.output_dir is None:
            self.output_dir = self.base_dir / "output"
        if self.model_path is None:
            self.model_path = self.base_dir / "model" / "yolos-tiny-signature"

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
    p = argparse.ArgumentParser(description="Signature detection batch test")
    p.add_argument("--upload-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--conf-floor", type=float, default=0.1, dest="conf_logging_floor",
                    help="Log all detections down to this confidence (default: 0.1)")
    p.add_argument("--conf-threshold", type=float, default=0.5, dest="conf_decision_threshold",
                    help="Detections at/above this count as a real signature (default: 0.5)")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold (default: 0.7)")
    p.add_argument("--imgsz", type=int, default=960, help="Inference resolution passed to YOLO (default: 960)")
    p.add_argument("--target-min-dim", type=int, default=1000, dest="target_min_dim",
                    help="Upscale images whose shorter side is below this (default: 1000px)")
    p.add_argument("--max-dim", type=int, default=3000, dest="max_dim",
                    help="Downscale images whose longer side exceeds this (default: 3000px)")
    p.add_argument("--max-upscale-factor", type=float, default=4.0, dest="max_upscale_factor")
    p.add_argument("--batch-size", type=int, default=4, dest="batch_size")
    p.add_argument("--workers", type=int, default=1,
                    help="Parallel worker processes for image loading/preprocessing (default: 1)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
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

    if scale == 1.0:
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
def run_inference_batch(model, processor, prepared_images: list[PreparedImage], cfg: Config):
    """Run Conditional-DETR on a batch of already-preprocessed images in one
    forward pass (more efficient than one-at-a-time on CPU once volume grows).
    Returns a list of detection-lists, aligned by index with prepared_images.
    Boxes are in the *preprocessed* image's pixel coordinates — annotation
    and cropping happen on that same preprocessed image, so no coordinate
    remapping back to the original scan is needed (see process_batch).

    Note: transformer-based detectors like YOLOS are set-based predictors
    (no NMS step), so cfg.iou and cfg.imgsz (both YOLO-specific knobs) are
    not used here; the processor handles resizing internally.
    """
    import torch

    sources = [p.image for p in prepared_images]
    inputs = processor(images=sources, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([p.image.size[::-1] for p in prepared_images])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=cfg.conf_logging_floor
    )

    all_detections = []
    for r in results:
        detections = []
        for score, box in zip(r["scores"], r["boxes"]):
            xyxy = tuple(box.tolist())
            detections.append({"confidence": float(score), "box": xyxy})
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        all_detections.append(detections)
    return all_detections


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
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
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
            f"Run: hf download mdefrance/yolos-tiny-signature-detection "
            f"--local-dir {cfg.model_path}"
        )

    cfg.sig_images_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoImageProcessor, AutoModelForObjectDetection  # deferred
                                   # import: keeps --help fast, avoids loading torch/
                                   # transformers just to print usage
    processor = AutoImageProcessor.from_pretrained(str(cfg.model_path))
    model = AutoModelForObjectDetection.from_pretrained(str(cfg.model_path))
    model.eval()

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
        detections_batch = run_inference_batch(model, processor, prepared, cfg)
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