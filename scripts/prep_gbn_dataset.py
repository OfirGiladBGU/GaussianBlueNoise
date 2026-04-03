#!/usr/bin/env python3
"""Preprocess problematic images for stippling.

Goal:
- Find images that are likely to stipple poorly (low contrast / weak sharpness)
- Apply a robust enhancement pipeline
- Save before/after metrics to verify whether preprocessing helps

Pipeline (OpenCV-based):
1. Global contrast stretch (percentile normalization)
2. CLAHE (local contrast boost)
3. Unsharp masking (edge emphasis)
4. Optional background suppression to white

Usage examples:
  python preprocess_problematic_stipple_images.py

  python preprocess_problematic_stipple_images.py \
      --input-dir TEST \
      --pattern "a*" \
      --auto-detect \
      --output-dir TEST/preprocessed_for_stipple

  python preprocess_problematic_stipple_images.py \
      --input-dir TEST \
      --pattern "a*" \
      --disable-bg-suppression
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Same style thresholds as analysis script
DYNAMIC_RANGE_MIN = 120
STD_DEV_MIN = 35
LAPLACIAN_VAR_MIN = 80.0


@dataclass
class Metrics:
    dynamic_range: float
    std_dev: float
    lap_var: float
    grad_mean: float
    grad_strong_pct: float


def iter_images(input_dir: Path, pattern: str) -> Iterable[Path]:
    for p in sorted(input_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in VALID_EXT:
            continue
        if p.match(pattern):
            yield p


def compute_metrics(gray: np.ndarray) -> Metrics:
    p2, p98 = np.percentile(gray, [2, 98])
    dynamic_range = float(p98 - p2)
    std_dev = float(gray.std())

    gray_f = gray.astype(np.float32)
    sx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx * sx + sy * sy)

    grad_mean = float(mag.mean())
    grad_strong_pct = float((mag > 30.0).mean() * 100.0)
    lap_var = float(cv2.Laplacian(gray_f, cv2.CV_32F).var())

    return Metrics(
        dynamic_range=dynamic_range,
        std_dev=std_dev,
        lap_var=lap_var,
        grad_mean=grad_mean,
        grad_strong_pct=grad_strong_pct,
    )


def is_problematic(m: Metrics) -> bool:
    return (
        m.dynamic_range < DYNAMIC_RANGE_MIN
        or m.std_dev < STD_DEV_MIN
        or m.lap_var < LAPLACIAN_VAR_MIN
    )


def percentile_stretch(gray: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(gray, [p_low, p_high])
    if hi - lo < 1e-6:
        return gray.copy()
    out = (gray.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_clahe(gray: np.ndarray, clip_limit: float = 3.0, tile: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def unsharp_mask(gray: np.ndarray, sigma: float = 1.5, amount: float = 1.4) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    # gray* (1+amount) - blur*amount
    sharp = cv2.addWeighted(gray.astype(np.float32), 1.0 + amount, blur.astype(np.float32), -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def suppress_background(gray: np.ndarray) -> np.ndarray:
    """Push likely background toward white while keeping foreground structure.

    Uses Otsu segmentation + morphology + border-connected background heuristic.
    """
    g = gray.copy()

    # Otsu split: background usually brighter in these sets
    _, bin_img = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean small noise in mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k, iterations=1)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k, iterations=1)

    # Keep border-connected bright regions as background candidates
    h, w = bin_img.shape
    flood = bin_img.copy()
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 128)
    bg = (flood == 128)

    out = g.astype(np.float32)
    # Blend background toward white (not hard clip, to avoid harsh seams)
    out[bg] = 0.30 * out[bg] + 0.70 * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def preprocess_image(gray: np.ndarray, do_bg_suppression: bool) -> np.ndarray:
    x = percentile_stretch(gray, p_low=1.0, p_high=99.0)
    x = apply_clahe(x, clip_limit=3.0, tile=8)
    x = unsharp_mask(x, sigma=1.4, amount=1.5)
    x = percentile_stretch(x, p_low=0.8, p_high=99.2)
    if do_bg_suppression:
        x = suppress_background(x)
    return x


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Preprocess problematic images for better stippling")
    parser.add_argument("--input-dir", type=Path, default=repo_root / "TEST")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "TEST" / "preprocessed_for_stipple")
    parser.add_argument("--pattern", type=str, default="a*", help="Filename glob pattern, e.g. 'a*' or '*.png'")
    parser.add_argument("--auto-detect", action="store_true", help="Only process images flagged as problematic by metrics")
    parser.add_argument("--disable-bg-suppression", action="store_true", help="Disable optional background suppression step")
    parser.add_argument("--save-compare-panels", action="store_true", help="Save side-by-side before/after PNG panels")
    return parser.parse_args()


def make_compare_panel(before_gray: np.ndarray, after_gray: np.ndarray) -> np.ndarray:
    h = max(before_gray.shape[0], after_gray.shape[0])
    w1, w2 = before_gray.shape[1], after_gray.shape[1]

    canvas = np.full((h, w1 + w2 + 10), 255, dtype=np.uint8)
    canvas[: before_gray.shape[0], :w1] = before_gray
    canvas[: after_gray.shape[0], w1 + 10 : w1 + 10 + w2] = after_gray

    bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    cv2.putText(bgr, "Before", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(bgr, "After", (w1 + 18, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 0), 2, cv2.LINE_AA)
    return bgr


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    do_bg_suppression = not args.disable_bg_suppression

    images = list(iter_images(input_dir, args.pattern))
    if not images:
        print(f"No images found in {input_dir} matching pattern '{args.pattern}'")
        return 1

    compare_dir = output_dir / "compare_panels"
    if args.save_compare_panels:
        compare_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    processed = 0
    skipped = 0

    for path in images:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            skipped += 1
            continue

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        before = compute_metrics(gray)

        if args.auto_detect and not is_problematic(before):
            skipped += 1
            rows.append(
                {
                    "filename": path.name,
                    "status": "skipped_not_problematic",
                    "before_range": f"{before.dynamic_range:.1f}",
                    "before_std": f"{before.std_dev:.1f}",
                    "before_lap": f"{before.lap_var:.1f}",
                    "after_range": "",
                    "after_std": "",
                    "after_lap": "",
                    "improved": "",
                }
            )
            continue

        enhanced = preprocess_image(gray, do_bg_suppression=do_bg_suppression)
        after = compute_metrics(enhanced)

        out_path = output_dir / f"{path.stem}_prep.png"
        cv2.imwrite(str(out_path), enhanced)

        if args.save_compare_panels:
            panel = make_compare_panel(gray, enhanced)
            cv2.imwrite(str(compare_dir / f"{path.stem}_compare.png"), panel)

        improved = (
            (after.dynamic_range > before.dynamic_range)
            and (after.std_dev > before.std_dev)
            and (after.lap_var > before.lap_var)
        )

        rows.append(
            {
                "filename": path.name,
                "status": "processed",
                "before_range": f"{before.dynamic_range:.1f}",
                "before_std": f"{before.std_dev:.1f}",
                "before_lap": f"{before.lap_var:.1f}",
                "after_range": f"{after.dynamic_range:.1f}",
                "after_std": f"{after.std_dev:.1f}",
                "after_lap": f"{after.lap_var:.1f}",
                "improved": str(improved),
            }
        )
        processed += 1

    csv_path = output_dir / "preprocess_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "status",
                "before_range",
                "before_std",
                "before_lap",
                "after_range",
                "after_std",
                "after_lap",
                "improved",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nPreprocessing complete")
    print(f"Input dir:   {input_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Processed:   {processed}")
    print(f"Skipped:     {skipped}")
    print(f"Metrics CSV: {csv_path}")

    if processed > 0:
        before_ranges = [float(r["before_range"]) for r in rows if r["status"] == "processed"]
        after_ranges = [float(r["after_range"]) for r in rows if r["status"] == "processed"]
        before_std = [float(r["before_std"]) for r in rows if r["status"] == "processed"]
        after_std = [float(r["after_std"]) for r in rows if r["status"] == "processed"]
        before_lap = [float(r["before_lap"]) for r in rows if r["status"] == "processed"]
        after_lap = [float(r["after_lap"]) for r in rows if r["status"] == "processed"]

        print("\nAverages (processed only):")
        print(f"  range: {np.mean(before_ranges):.1f} -> {np.mean(after_ranges):.1f}")
        print(f"  std:   {np.mean(before_std):.1f} -> {np.mean(after_std):.1f}")
        print(f"  lap:   {np.mean(before_lap):.1f} -> {np.mean(after_lap):.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
