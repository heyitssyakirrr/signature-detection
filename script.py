"""
Signature Detection Model Test
tech4humans/yolov8s-signature-detector — CPU-only, offline, stand-alone test script.

See signature-detection-test-project.md / signature-detection-test-plan.md for full context.
Run with: python script.py
"""

import os

# Force fully-offline behaviour before any HF/ultralytics imports touch the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Config — these are the "defaults for this first run", tune after reviewing
# results.csv and the annotated images (see project description Section 6).
# --------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "upload"
OUTPUT_DIR = BASE_DIR / "output"
SIG_IMAGES_DIR = OUTPUT_DIR / "signature_images"
RESULTS_CSV = OUTPUT_DIR / "results.csv"
MODEL_PATH = BASE_DIR / "model" / "yolov8s.pt"

CONF_LOGGING_FLOOR = 0.1   # log every raw detection down to this confidence
CONF_DECISION_THRESHOLD = 0.5  # detections at/above this count as "a signature"
NMS_IOU_THRESHOLD = 0.7    # ultralytics default; log final boxes so overlaps are visible

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# --------------------------------------------------------------------------
# Step 5.1 — Image loading: handle PNG/TIFF, grayscale/color, convert to RGB
# --------------------------------------------------------------------------
def load_image_rgb(path: Path) -> Image.Image:
    """Load an image from disk and return a 3-channel RGB PIL Image,
    regardless of source format or color mode (grayscale PNG, TIFF, etc.)."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# --------------------------------------------------------------------------
# Step 5.2 — Inference: run the model, return raw detections
# --------------------------------------------------------------------------
def run_inference(model: YOLO, image: Image.Image):
    """Run YOLOv8s signature detector on a single RGB image.
    Returns a list of dicts: [{"confidence": float, "box": (x1, y1, x2, y2)}, ...]
    Uses CONF_LOGGING_FLOOR as the model-level conf cutoff so nothing below
    the logging floor is even returned (everything at/above it is logged)."""
    results = model.predict(
        source=image,
        conf=CONF_LOGGING_FLOOR,
        iou=NMS_IOU_THRESHOLD,
        verbose=False,
    )
    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2] in pixel coords
            conf = float(box.conf[0])
            detections.append({"confidence": conf, "box": tuple(xyxy)})
    # Sort by confidence descending — just for readable logs; no left-to-right
    # / top-to-bottom ordering requirement per the project description.
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


# --------------------------------------------------------------------------
# Step 5.3 — Thresholding / box-size logging
# --------------------------------------------------------------------------
def compute_box_size(box, image_size):
    """Return width, height (px) and area as % of full image area."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    img_w, img_h = image_size
    area_pct = (w * h) / (img_w * img_h) * 100 if img_w and img_h else 0.0
    return {"width": round(w, 1), "height": round(h, 1), "area_pct": round(area_pct, 3)}


def split_by_threshold(detections):
    """All detections are already >= CONF_LOGGING_FLOOR (enforced at inference).
    Split into (decision_detections, all_detections) where decision_detections
    are those >= CONF_DECISION_THRESHOLD."""
    decision = [d for d in detections if d["confidence"] >= CONF_DECISION_THRESHOLD]
    return decision, detections


# --------------------------------------------------------------------------
# Step 5.4 — Annotation drawing
# --------------------------------------------------------------------------
def draw_annotations(image: Image.Image, detections) -> Image.Image:
    """Draw all logged detections (>= 0.1 conf) on a copy of the image.
    Detections at/above the 0.5 decision threshold are drawn in green (solid);
    detections below 0.5 but above the logging floor are drawn in orange
    (dashed-look via short segments) so borderline cases are easy to spot
    during review (project doc Step 7: check for below-threshold real
    signatures, and above-threshold false positives)."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()

    for d in detections:
        x1, y1, x2, y2 = d["box"]
        conf = d["confidence"]
        is_decision = conf >= CONF_DECISION_THRESHOLD
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


# --------------------------------------------------------------------------
# Step 5.5 — Cropping (only detections at/above the 0.5 decision threshold)
# --------------------------------------------------------------------------
def crop_signatures(image: Image.Image, decision_detections):
    """Return list of cropped PIL Images, one per decision-threshold detection."""
    crops = []
    for d in decision_detections:
        x1, y1, x2, y2 = d["box"]
        # clamp to image bounds defensively
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        crops.append(image.crop((x1, y1, x2, y2)))
    return crops


# --------------------------------------------------------------------------
# Step 5.6 — Per-image result recording
# --------------------------------------------------------------------------
def build_result_row(filename, decision_detections, all_detections, image_size):
    box_sizes = [compute_box_size(d["box"], image_size) for d in all_detections]
    return {
        "filename": filename,
        "signature": len(decision_detections) > 0,
        "num_signatures": len(decision_detections),
        "confidence_scores": [round(d["confidence"], 4) for d in all_detections],
        "box_sizes": [
            f"{bs['width']}x{bs['height']}px ({bs['area_pct']}%)" for bs in box_sizes
        ],
    }


# --------------------------------------------------------------------------
# Step 5.7 — Batch loop
# --------------------------------------------------------------------------
def process_one_image(model: YOLO, path: Path):
    stem = path.stem
    out_dir = SIG_IMAGES_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    image = load_image_rgb(path)
    detections = run_inference(model, image)
    decision_detections, all_detections = split_by_threshold(detections)

    annotated = draw_annotations(image, all_detections)
    annotated.save(out_dir / f"{stem}_annotated.png")

    crops = crop_signatures(image, decision_detections)
    for i, crop in enumerate(crops, start=1):
        crop.save(out_dir / f"{stem}_sig{i}.png")

    row = build_result_row(path.name, decision_detections, all_detections, image.size)
    return row


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_PATH}. "
            f"See model/DOWNLOAD_INSTRUCTIONS.md to fetch yolov8s.pt first."
        )

    SIG_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    image_paths = sorted(
        p for p in UPLOAD_DIR.iterdir() if p.suffix.lower() in VALID_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {UPLOAD_DIR}")

    rows = []
    for path in image_paths:
        print(f"Processing {path.name} ...")
        row = process_one_image(model, path)
        rows.append(row)
        print(
            f"  -> signature={row['signature']} "
            f"num_signatures={row['num_signatures']} "
            f"confidences={row['confidence_scores']}"
        )

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "signature", "num_signatures", "confidence_scores", "box_sizes"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nDone. Wrote {len(rows)} rows to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
