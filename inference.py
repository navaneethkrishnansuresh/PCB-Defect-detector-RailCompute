#!/usr/bin/env python3
"""Single-image inference for the RailCompute PCB defect detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "best.pt"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "examples" / "output"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

COLORS = [
    (255, 80, 80),
    (255, 170, 40),
    (60, 180, 255),
    (255, 70, 180),
    (80, 230, 120),
    (180, 100, 255),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PCB defect detection on one image.")
    parser.add_argument("--image", required=True, help="Path to one input image")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to YOLO .pt weights")
    parser.add_argument("--output", default=None, help="Output image path")
    parser.add_argument("--conf", type=float, default=0.70, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or CUDA device like '0'")
    parser.add_argument("--no-heatmap", action="store_true", help="Disable heatmap overlay")
    return parser.parse_args()


def resolve_device(value: str) -> str | int:
    if value == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if value.isdigit():
        return int(value)
    return value


def add_heatmap(img: np.ndarray, boxes: list[tuple[int, int, int, int]], confs: list[float]) -> np.ndarray:
    if not boxes:
        return img

    h, w = img.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)

    for (x1, y1, x2, y2), conf in zip(boxes, confs):
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        bw, bh = max(40, x2 - x1), max(40, y2 - y1)
        rx, ry = int(bw * 2.7), int(bh * 2.7)

        x0, x3 = max(0, cx - rx), min(w, cx + rx)
        y0, y3 = max(0, cy - ry), min(h, cy + ry)
        yy, xx = np.mgrid[y0:y3, x0:x3]

        blob = np.exp(
            -(
                ((xx - cx) ** 2) / (2 * (rx / 2.2) ** 2)
                + ((yy - cy) ** 2) / (2 * (ry / 2.2) ** 2)
            )
        )
        heat[y0:y3, x0:x3] += blob * max(conf, 0.7)

    heat = heat / max(float(heat.max()), 1e-6)
    heat_u8 = cv2.GaussianBlur(np.uint8(255 * heat), (0, 0), 20)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    mask = np.clip((heat_u8 / 255.0)[..., None] * 0.55, 0, 0.55)
    return (img * (1 - mask) + heat_color * mask).astype(np.uint8)


def draw_label(img: np.ndarray, x1: int, y1: int, text: str, conf: float, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, min(img.shape[1] / 1200, 0.85))
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    y_text = max(y1 - 10, th + 16)

    cv2.rectangle(img, (x1, y_text - th - 12), (x1 + tw + 16, y_text + 8), color, -1)
    cv2.putText(img, text, (x1 + 8, y_text - 3), font, scale, (20, 20, 20), thick, cv2.LINE_AA)

    bar_width = int((tw + 16) * min(max(conf, 0.0), 1.0))
    cv2.rectangle(img, (x1, y_text + 10), (x1 + tw + 16, y_text + 15), (45, 45, 45), -1)
    cv2.rectangle(img, (x1, y_text + 10), (x1 + bar_width, y_text + 15), color, -1)


def annotate(img: np.ndarray, result: object, show_heatmap: bool) -> tuple[np.ndarray, list[dict[str, object]]]:
    detections: list[dict[str, object]] = []
    boxes_for_heatmap: list[tuple[int, int, int, int]] = []
    confs_for_heatmap: list[float] = []

    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
        name = result.names[cls_id]

        boxes_for_heatmap.append((x1, y1, x2, y2))
        confs_for_heatmap.append(conf)
        detections.append(
            {
                "class_id": cls_id,
                "class_name": name,
                "confidence": round(conf, 4),
                "box_xyxy": [x1, y1, x2, y2],
            }
        )

    if show_heatmap:
        img = add_heatmap(img, boxes_for_heatmap, confs_for_heatmap)

    for det in detections:
        cls_id = int(det["class_id"])
        conf = float(det["confidence"])
        name = str(det["class_name"]).replace("_", " ").title()
        x1, y1, x2, y2 = [int(v) for v in det["box_xyxy"]]
        color = COLORS[cls_id % len(COLORS)]

        glow = np.zeros_like(img)
        cv2.rectangle(glow, (x1, y1), (x2, y2), color, 9)
        glow = cv2.GaussianBlur(glow, (0, 0), 9)
        img = cv2.addWeighted(img, 1.0, glow, 0.45, 0)

        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        img = cv2.addWeighted(overlay, 0.08, img, 0.92, 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1 + 4, y1 + 4), (x2 - 4, y2 - 4), (255, 255, 255), 1)
        draw_label(img, x1, y1, f"{name} {conf:.2f}", conf, color)

    if not detections:
        cv2.putText(
            img,
            "No defect detected",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

    return img, detections


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    weights_path = Path(args.weights)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not image_path.exists() or image_path.suffix.lower() not in IMAGE_EXTS:
        raise FileNotFoundError(f"Input image not found or unsupported: {image_path}")

    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR / f"{image_path.stem}_annotated.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    result = model.predict(
        source=str(image_path),
        conf=args.conf,
        imgsz=args.imgsz,
        device=resolve_device(args.device),
        verbose=False,
    )[0]

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    annotated, detections = annotate(img, result, show_heatmap=not args.no_heatmap)
    cv2.imwrite(str(output_path), annotated)

    print(json.dumps({"output": str(output_path), "detections": detections}, indent=2))


if __name__ == "__main__":
    main()

