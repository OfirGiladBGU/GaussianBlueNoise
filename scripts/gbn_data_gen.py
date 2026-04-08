#!/usr/bin/env python3
"""GBN dataset generator.

Reads images from <data_path>/original/, prepares source/ and target/ outputs,
and writes prompt.json JSONL entries.

Preprocessing lives in image_preprocess.py and is used when --apply_preprocess.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
GBN_BINARY = REPO_ROOT / "gbn-adaptive-linux"
VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

sys.path.insert(0, str(SCRIPT_DIR))
from image_preprocess import load_gray, preprocess_image  # noqa: E402


def write_pgm(gray: np.ndarray, path: Path) -> None:
    h, w = gray.shape
    with path.open("w", encoding="utf-8") as f:
        f.write("P2\n")
        f.write(f"{w} {h}\n")
        f.write("255\n")
        for row in gray:
            for val in row:
                f.write(f"{int(val)}\n")


def load_points(txt_path: Path) -> np.ndarray:
    pts = []
    with txt_path.open("r", encoding="utf-8") as f:
        _n = int(f.readline().strip())
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                pts.append((float(parts[0]), float(parts[1])))
    return np.asarray(pts, dtype=np.float32)


def normalize_points(points: np.ndarray, width: int, height: int, coord_mode: str) -> np.ndarray:
    if len(points) == 0:
        return points

    pts = points.copy()
    aspect_h_over_w = float(height) / float(width)
    aspect_w_over_h = float(width) / float(height)

    if coord_mode == "unit":
        return pts

    x_max = float(np.max(pts[:, 0]))
    y_max = float(np.max(pts[:, 1]))

    def _near(a: float, b: float, tol: float = 0.08) -> bool:
        return abs(a - b) <= tol

    x_compressed = aspect_w_over_h < 1.0 and _near(x_max, aspect_w_over_h)
    y_compressed = aspect_h_over_w < 1.0 and _near(y_max, aspect_h_over_w)

    if coord_mode in ("aspect", "auto"):
        if x_compressed and aspect_w_over_h > 0:
            pts[:, 0] = pts[:, 0] / aspect_w_over_h
        if y_compressed and aspect_h_over_w > 0:
            pts[:, 1] = pts[:, 1] / aspect_h_over_w

    pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.0)
    return pts


def render_stipple(points: np.ndarray, width: int, height: int, point_size: float, coord_mode: str) -> np.ndarray:
    pts = normalize_points(points, width, height, coord_mode=coord_mode)
    canvas = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(canvas)
    radius = max(0, int(round(point_size / 2.0)))

    for pt in pts:
        x = int(np.clip(round(float(pt[0]) * (width - 1)), 0, width - 1))
        y = int(np.clip(round((1.0 - float(pt[1])) * (height - 1)), 0, height - 1))
        if radius <= 0:
            draw.point((x, y), fill=0)
        else:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=0)

    return np.array(canvas)


def run_gbn(density: np.ndarray, out_txt: Path, n_points: int, n_iters: int) -> None:
    tmp_pgm = out_txt.with_suffix(".tmp.pgm")
    write_pgm(density, tmp_pgm)

    env = os.environ.copy()
    conda_prefix = env.get("CONDA_PREFIX", "")
    if conda_prefix:
        env["LD_LIBRARY_PATH"] = f"{conda_prefix}/lib:{env.get('LD_LIBRARY_PATH', '')}"

    cmd = [str(GBN_BINARY), str(tmp_pgm), str(n_points), str(n_iters), str(out_txt)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    tmp_pgm.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(f"GBN binary failed:\n{proc.stderr}")


def quantize_gray(gray: np.ndarray, n_colors: int) -> np.ndarray:
    """Reduce gray image to n_colors uniform intensity levels."""
    levels = np.linspace(0, 255, n_colors, dtype=np.float32)
    indices = np.argmin(np.abs(gray.astype(np.float32)[..., None] - levels), axis=-1)
    return levels[indices].astype(np.uint8)


def prepare_source_gray(
    input_image: Path,
    image_size: tuple[int, int] | None,
    apply_preprocess: bool,
    disable_bg_suppression: bool,
    invert_image: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (before_preprocess_gray, final_source_gray)."""
    before = load_gray(input_image, image_size=image_size)
    source_gray = before.copy()

    if apply_preprocess:
        source_gray = preprocess_image(source_gray, do_bg_suppression=not disable_bg_suppression)

    if invert_image:
        source_gray = 255 - source_gray

    return before, source_gray


def process_one(
    src_path: Path,
    source_out: Path,
    target_out: Path,
    *,
    image_size: tuple[int, int] | None,
    invert_image: bool,
    invert_density: bool,
    threshold: int,
    apply_preprocess: bool,
    disable_bg_suppression: bool,
    apply_quantization: bool,
    quantization_count: int,
    n_points: int,
    n_iters: int,
    point_size: float,
    coord_mode: str,
    overwrite: bool,
    keep_txt: bool,
) -> str:
    txt_path = target_out.with_suffix(".txt")
    txt_ready = txt_path.exists() if keep_txt else True
    if not overwrite and source_out.exists() and target_out.exists() and txt_ready:
        return "skipped"

    before, source_gray = prepare_source_gray(
        input_image=src_path,
        image_size=image_size,
        apply_preprocess=apply_preprocess,
        disable_bg_suppression=disable_bg_suppression,
        invert_image=invert_image,
    )

    if apply_quantization:
        source_gray = quantize_gray(source_gray, quantization_count)

    source_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(source_out), source_gray)

    density = source_gray.astype(np.float32)
    if invert_density:
        density = 255.0 - density
    density = np.minimum(density, float(threshold))

    d_min, d_max = density.min(), density.max()
    if d_max - d_min > 1e-5:
        density = (density - d_min) * (255.0 / (d_max - d_min))
    density = np.clip(density, 0, 255).astype(np.uint8)

    target_out.parent.mkdir(parents=True, exist_ok=True)
    run_gbn(density, txt_path, n_points=n_points, n_iters=n_iters)

    h_px, w_px = source_gray.shape[:2]
    points = load_points(txt_path)
    rendered = render_stipple(points, w_px, h_px, point_size=point_size, coord_mode=coord_mode)
    cv2.imwrite(str(target_out), rendered)

    if not keep_txt:
        txt_path.unlink(missing_ok=True)

    return "processed"


def main() -> int:
    # ── Configuration (edit values here) ──────────────────────────────────────
    data_path          = Path("/groups/asharf_group/ofirgila/ControlNet/training/FiveK_dataset_nzk_512x512")  # Dataset root that contains original/source/target
    n                  = -1             # Number of images to process; -1 = all
    n_points           = 1024           # GBN point count
    n_iters            = 1000           # GBN optimization iterations
    threshold          = 255            # Density cap before stippling; 255 means no cap
    image_size         = (512, 512)     # (W, H) or None to keep original size
    invert_image       = False          # Invert source image pixels
    invert_density     = False          # Invert density seen by GBN
    point_size         = 1.0            # Rendered stipple point size in pixels
    apply_preprocess   = True           # Apply preprocessing pipeline before stippling
    disable_bg_suppression = False      # Disable bg suppression inside preprocessing
    apply_quantization = False          # Quantize gray levels before stippling
    quantization_count = 4              # Number of gray levels after quantization
    coord_mode         = "auto"         # One of: auto, unit, aspect
    overwrite          = True           # Overwrite existing source/target files
    keep_txt           = False          # Keep GBN txt files (default off for dataset generation)
    # ──────────────────────────────────────────────────────────────────────────

    parser = argparse.ArgumentParser(
        description="Generate source/target stippling dataset with Gaussian Blue Noise",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # fmt: off
    parser.add_argument("--data_path",              type=Path,  default=data_path)
    parser.add_argument("--n",                      type=int,   default=n,                  help="-1 means all images")
    parser.add_argument("--n_points",               type=int,   default=n_points)
    parser.add_argument("--n_iters",                type=int,   default=n_iters)
    parser.add_argument("--threshold",              type=int,   default=threshold)
    parser.add_argument("--image_size",             type=int,   default=image_size,         nargs=2, metavar=("W", "H"))
    parser.add_argument("--invert_image",           action=argparse.BooleanOptionalAction,  default=invert_image)
    parser.add_argument("--invert_density",         action=argparse.BooleanOptionalAction,  default=invert_density)
    parser.add_argument("--point_size",             type=float, default=point_size)
    parser.add_argument("--apply_preprocess",       action=argparse.BooleanOptionalAction,  default=apply_preprocess)
    parser.add_argument("--disable_bg_suppression", action=argparse.BooleanOptionalAction,  default=disable_bg_suppression)
    parser.add_argument("--apply_quantization",     action=argparse.BooleanOptionalAction,  default=apply_quantization)
    parser.add_argument("--quantization_count",     type=int,   default=quantization_count)
    parser.add_argument("--coord_mode",             type=str,   default=coord_mode,         choices=["auto", "unit", "aspect"])
    parser.add_argument("--overwrite",              action=argparse.BooleanOptionalAction,  default=overwrite)
    parser.add_argument("--keep_txt",               action=argparse.BooleanOptionalAction,  default=keep_txt)
    # fmt: on

    args = parser.parse_args()

    if not GBN_BINARY.is_file():
        print(f"Error: GBN binary not found: {GBN_BINARY}", file=sys.stderr)
        return 1

    data_path = args.data_path.resolve()
    original_dir = data_path / "original"
    source_dir = data_path / "source"
    target_dir = data_path / "target"
    json_path = data_path / "prompt.json"

    if not original_dir.is_dir():
        print(f"Error: 'original/' folder not found under: {data_path}", file=sys.stderr)
        return 1

    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    image_size = tuple(args.image_size) if args.image_size is not None else None

    image_files = sorted(
        [
            p.relative_to(original_dir)
            for p in original_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in VALID_EXT
        ]
    )

    if not image_files:
        print(f"No images found under: {original_dir}", file=sys.stderr)
        return 1

    n = len(image_files) if args.n == -1 else min(args.n, len(image_files))

    print(f"Data path: {data_path}")
    print(f"Images found: {len(image_files)} | processing: {n}")

    ok = 0
    skipped = 0
    failed = 0

    for i, rel_path in enumerate(image_files[:n], start=1):
        src = original_dir / rel_path
        source_out = source_dir / rel_path.with_suffix(".png")
        target_out = target_dir / rel_path.with_suffix(".png")

        print(f"[{i}/{n}] {rel_path} ... ", end="", flush=True)
        try:
            status = process_one(
                src_path=src,
                source_out=source_out,
                target_out=target_out,
                image_size=image_size,
                invert_image=args.invert_image,
                invert_density=args.invert_density,
                threshold=args.threshold,
                apply_preprocess=args.apply_preprocess,
                disable_bg_suppression=args.disable_bg_suppression,
                apply_quantization=args.apply_quantization,
                quantization_count=args.quantization_count,
                n_points=args.n_points,
                n_iters=args.n_iters,
                point_size=args.point_size,
                coord_mode=args.coord_mode,
                overwrite=args.overwrite,
                keep_txt=args.keep_txt,
            )
            if status == "skipped":
                skipped += 1
                print("skipped")
            else:
                ok += 1
                print("done")
        except Exception as exc:
            failed += 1
            print("FAILED")
            print(f"  -> {exc}")

    with json_path.open("w", encoding="utf-8") as f:
        for rel_path in image_files[:n]:
            entry = {
                "source": f"source/{rel_path.with_suffix('.png').as_posix()}",
                "target": f"target/{rel_path.with_suffix('.png').as_posix()}",
                "prompt": "Stippling",
            }
            f.write(json.dumps(entry) + "\n")

    print("\nSummary")
    print(f"  processed:   {ok}")
    print(f"  skipped:     {skipped}")
    print(f"  failed:      {failed}")
    print(f"  prompt.json: {json_path}")

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
