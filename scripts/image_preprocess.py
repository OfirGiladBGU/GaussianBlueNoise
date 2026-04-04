#!/usr/bin/env python3
"""GBN preprocessing utilities.

This module owns all preprocessing logic used before stippling.
It can also run in debug mode for a single image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def percentile_stretch(gray: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(gray, [p_low, p_high])
    if hi - lo < 1e-6:
        return gray.copy()
    out = (gray.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_clahe(gray: np.ndarray, clip_limit: float = 3.0, tile: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def unsharp_mask(gray: np.ndarray, sigma: float = 1.4, amount: float = 1.5) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    sharp = cv2.addWeighted(
        gray.astype(np.float32),
        1.0 + amount,
        blur.astype(np.float32),
        -amount,
        0,
    )
    return np.clip(sharp, 0, 255).astype(np.uint8)


def suppress_background(gray: np.ndarray) -> np.ndarray:
    """Push likely background toward white while keeping foreground structure."""
    g = gray.copy()
    _, bin_img = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k, iterations=1)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k, iterations=1)

    h, w = bin_img.shape
    flood = bin_img.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 128)

    bg = flood == 128
    out = g.astype(np.float32)
    out[bg] = 0.30 * out[bg] + 0.70 * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def preprocess_image(gray: np.ndarray, do_bg_suppression: bool = True) -> np.ndarray:
    """Enhancement pipeline: stretch -> CLAHE -> unsharp -> stretch -> optional bg suppress."""
    x = percentile_stretch(gray, p_low=1.0, p_high=99.0)
    x = apply_clahe(x, clip_limit=3.0, tile=8)
    x = unsharp_mask(x, sigma=1.4, amount=1.5)
    x = percentile_stretch(x, p_low=0.8, p_high=99.2)
    if do_bg_suppression:
        x = suppress_background(x)
    return x


def load_gray(image_path: Path, image_size: tuple[int, int] | None = None) -> np.ndarray:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if image_size is not None:
        w, h = image_size
        gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return gray


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


def preprocess_path(
    input_path: Path,
    output_path: Path,
    image_size: tuple[int, int] | None = None,
    invert_image: bool = False,
    disable_bg_suppression: bool = False,
) -> np.ndarray:
    gray = load_gray(input_path, image_size=image_size)
    out = preprocess_image(gray, do_bg_suppression=not disable_bg_suppression)
    if invert_image:
        out = 255 - out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), out)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug preprocessing on a single image")
    parser.add_argument("--input-image", type=Path, required=True, help="Input image path")
    parser.add_argument("--output-image", type=Path, required=True, help="Output preprocessed image path")
    parser.add_argument("--compare-output", type=Path, default=None, help="Optional side-by-side debug image")
    parser.add_argument("--image_size", type=int, nargs=2, default=None, metavar=("W", "H"))
    parser.add_argument("--invert_image", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable_bg_suppression", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_size = tuple(args.image_size) if args.image_size is not None else None

    before = load_gray(args.input_image, image_size=image_size)
    after = preprocess_path(
        input_path=args.input_image,
        output_path=args.output_image,
        image_size=image_size,
        invert_image=args.invert_image,
        disable_bg_suppression=args.disable_bg_suppression,
    )

    if args.compare_output is not None:
        panel = make_compare_panel(before, after)
        args.compare_output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.compare_output), panel)

    print(f"Saved preprocessed image: {args.output_image}")
    if args.compare_output is not None:
        print(f"Saved compare panel:     {args.compare_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
