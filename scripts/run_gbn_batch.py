#!/usr/bin/env python3
"""Batch-run GBN stippling for a source folder.

Reads images from a source directory, runs Gaussian Blue Noise stippling,
and writes outputs to a target directory.

Default paths:
- source: ../TEST/source
- target: ../TEST/target

Outputs per image:
- <target>/<stem>.png        rendered stippling image
- <target>/<stem>.txt        point coordinates

Usage:
  python run_gbn_batch.py
  python run_gbn_batch.py --points 4096 --iters 300
  python run_gbn_batch.py --source ../TEST/source --target ../TEST/target --overwrite
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
GBN_BINARY = REPO_ROOT / "gbn-adaptive-linux"

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}


def image_to_pgm(input_path: Path, output_path: Path) -> None:
    """Convert input image to ASCII PGM (P2), as expected by gbn-adaptive-linux."""
    img = Image.open(input_path).convert("L")
    pixels = np.array(img, dtype=np.uint8)
    h, w = pixels.shape

    with output_path.open("w", encoding="utf-8") as f:
        f.write("P2\n")
        f.write(f"{w} {h}\n")
        f.write("255\n")
        for row in pixels:
            for val in row:
                f.write(f"{int(val)}\n")


def load_points(txt_path: Path) -> np.ndarray:
    """Load points from the GBN output text format."""
    pts = []
    with txt_path.open("r", encoding="utf-8") as f:
        _n = int(f.readline().strip())
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                pts.append((float(parts[0]), float(parts[1])))
    return np.asarray(pts, dtype=np.float32)


def normalize_points(points: np.ndarray, width: int, height: int, coord_mode: str) -> np.ndarray:
    """Normalize points to unit-square image coordinates.

    coord_mode:
    - unit: assume x,y already in [0,1]
    - aspect: apply aspect correction on the compressed axis
    - auto: infer compressed axis from observed coordinate ranges
    """
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

    # Portrait inputs often come with x in [0, w/h] and y in [0, 1].
    x_compressed = aspect_w_over_h < 1.0 and _near(x_max, aspect_w_over_h)
    # Landscape inputs often come with y in [0, h/w] and x in [0, 1].
    y_compressed = aspect_h_over_w < 1.0 and _near(y_max, aspect_h_over_w)

    if coord_mode == "aspect":
        if x_compressed and aspect_w_over_h > 0:
            pts[:, 0] = pts[:, 0] / aspect_w_over_h
        if y_compressed and aspect_h_over_w > 0:
            pts[:, 1] = pts[:, 1] / aspect_h_over_w
        return pts

    # auto mode
    if x_compressed and aspect_w_over_h > 0:
        pts[:, 0] = pts[:, 0] / aspect_w_over_h
    if y_compressed and aspect_h_over_w > 0:
        pts[:, 1] = pts[:, 1] / aspect_h_over_w

    # Guard against small numeric spillover.
    pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.0)
    return pts


def render_points(points: np.ndarray, output_png: Path, point_size: float, width: int, height: int, coord_mode: str) -> None:
    """Render points to a white image with exact input dimensions."""
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

    canvas.save(output_png)


def run_one(
    input_path: Path,
    target_dir: Path,
    n_points: int,
    n_iters: int,
    point_size: float,
    overwrite: bool,
    coord_mode: str,
) -> str:
    """Run GBN for one image and save outputs in target_dir."""
    stem = input_path.stem
    out_txt = target_dir / f"{stem}.txt"
    out_png = target_dir / f"{stem}.png"
    tmp_pgm = target_dir / f"{stem}__temp.pgm"

    if not overwrite and out_txt.exists() and out_png.exists():
        return "skipped"

    with Image.open(input_path) as src_img:
        width, height = src_img.size

    if input_path.suffix.lower() == ".pgm":
        pgm_path = input_path
    else:
        image_to_pgm(input_path, tmp_pgm)
        pgm_path = tmp_pgm

    env = os.environ.copy()
    conda_prefix = env.get("CONDA_PREFIX", "")
    if conda_prefix:
        env["LD_LIBRARY_PATH"] = f"{conda_prefix}/lib:{env.get('LD_LIBRARY_PATH', '')}"

    cmd = [
        str(GBN_BINARY),
        str(pgm_path),
        str(n_points),
        str(n_iters),
        str(out_txt),
    ]

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        if tmp_pgm.exists():
            tmp_pgm.unlink()
        raise RuntimeError(f"GBN failed for {input_path.name}:\n{proc.stderr}")

    points = load_points(out_txt)
    render_points(points, out_png, point_size=point_size, width=width, height=height, coord_mode=coord_mode)

    if tmp_pgm.exists():
        tmp_pgm.unlink()

    return "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch run Gaussian Blue Noise stippling")
    parser.add_argument("--source", type=Path, default=REPO_ROOT / "TEST" / "source", help="Input folder")
    parser.add_argument("--target", type=Path, default=REPO_ROOT / "TEST" / "target", help="Output folder")
    parser.add_argument("--points", type=int, default=4096, help="Number of points per image")
    parser.add_argument("--iters", type=int, default=1000, help="Optimization iterations")
    parser.add_argument("--point-size", type=float, default=1.0, help="Rendered point size")
    parser.add_argument("--coord-mode", type=str, default="auto", choices=["auto", "unit", "aspect"], help="Coordinate interpretation mode")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N images (0 = all)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not GBN_BINARY.is_file():
        print(f"Error: missing binary: {GBN_BINARY}", file=sys.stderr)
        return 1

    source = args.source.resolve()
    target = args.target.resolve()

    if not source.is_dir():
        print(f"Error: source folder not found: {source}", file=sys.stderr)
        return 1

    target.mkdir(parents=True, exist_ok=True)

    images = [
        p for p in sorted(source.iterdir())
        if p.is_file() and p.suffix.lower() in VALID_EXT
    ]

    if args.limit > 0:
        images = images[: args.limit]

    if not images:
        print(f"No input images found in {source}")
        return 1

    print(f"Source: {source}")
    print(f"Target: {target}")
    print(
        f"Images: {len(images)} | points={args.points} | iters={args.iters} | coord_mode={args.coord_mode}"
    )

    ok = 0
    skipped = 0
    failed = 0

    for i, img in enumerate(images, start=1):
        print(f"[{i}/{len(images)}] {img.name} ... ", end="", flush=True)
        try:
            status = run_one(
                input_path=img,
                target_dir=target,
                n_points=args.points,
                n_iters=args.iters,
                point_size=args.point_size,
                overwrite=args.overwrite,
                coord_mode=args.coord_mode,
            )
            if status == "skipped":
                skipped += 1
                print("skipped")
            else:
                ok += 1
                print("done")
        except Exception as exc:
            failed += 1
            print("failed")
            print(f"  -> {exc}")

    print("\nSummary")
    print(f"  processed: {ok}")
    print(f"  skipped:   {skipped}")
    print(f"  failed:    {failed}")

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
