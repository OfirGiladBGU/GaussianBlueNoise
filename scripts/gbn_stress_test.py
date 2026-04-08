#!/usr/bin/env python3
"""Standalone GBN stress-test generator.

This script is intentionally self-contained and follows generate_stippling.py-style
GBN invocation while exposing gbn_data_gen.py-like arguments.

Pipeline:
1. Read a single image from <data_path>/original/
2. Duplicate it to <data_path>/source/ as <stem>_i.png
3. Run GBN per duplicate and save rendered stippling to <data_path>/target/<stem>_i.png
4. Save HDF5 at <data_path>/db/db_full.hdf5 with schema:
   GBN/<n_points>/data   : (count, n_points, 2)
   GBN/<n_points>/data_t : (count, 2, side, side)
   GBN/<n_points>/prop   : (count, 9)

Notes:
- No preprocessing pipeline is applied in this stress version.
- Quantization can be enabled for stress experiments.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
GBN_BINARY = REPO_ROOT / "gbn-adaptive-linux"
VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}


def write_pgm(gray: np.ndarray, path: Path) -> None:
    h, w = gray.shape
    with path.open("w", encoding="utf-8") as f:
        f.write("P2\n")
        f.write(f"{w} {h}\n")
        f.write("255\n")
        for row in gray:
            for val in row:
                f.write(f"{int(val)}\n")


def load_gray(image_path: Path, image_size: tuple[int, int] | None) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    if image_size is not None:
        img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
    return img


def quantize_gray(gray: np.ndarray, n_colors: int) -> np.ndarray:
    if n_colors < 2:
        raise ValueError("quantization_count must be >= 2")
    levels = np.linspace(0, 255, n_colors, dtype=np.float32)
    idx = np.argmin(np.abs(gray.astype(np.float32)[..., None] - levels), axis=-1)
    return levels[idx].astype(np.uint8)


def load_points(txt_path: Path) -> np.ndarray:
    pts = []
    with txt_path.open("r", encoding="utf-8") as f:
        _n = int(f.readline().strip())
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                pts.append((float(parts[0]), float(parts[1])))
    return np.asarray(pts, dtype=np.float64)


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
    if not GBN_BINARY.is_file():
        raise FileNotFoundError(f"GBN binary not found: {GBN_BINARY}")

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


def make_density(gray: np.ndarray, invert_density: bool, threshold: int) -> np.ndarray:
    density = gray.astype(np.float64)
    if invert_density:
        density = 255.0 - density
    density = np.minimum(density, float(threshold))

    d_min, d_max = density.min(), density.max()
    if d_max - d_min > 1e-9:
        density = (density - d_min) * (255.0 / (d_max - d_min))
    return np.clip(density, 0.0, 255.0).astype(np.uint8)


def points_to_data_t(points: np.ndarray, n_points: int) -> np.ndarray:
    """Build a local data_t tensor with expected shape (2, side, side).

    This keeps schema compatibility without importing any external repository code.
    """
    side = int(round(np.sqrt(n_points)))
    if side * side != n_points:
        raise ValueError("n_points must be a perfect square for data_t conversion")
    if points.shape != (n_points, 2):
        raise ValueError(f"Expected points shape ({n_points}, 2), got {points.shape}")

    gy, gx = np.meshgrid((np.arange(side) + 0.5) / side, (np.arange(side) + 0.5) / side, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)

    order_pts = np.lexsort((points[:, 0], points[:, 1]))
    pts_sorted = points[order_pts]

    offsets = (pts_sorted - grid) * side
    data_t = offsets.T.reshape(2, side, side)
    return data_t.astype(np.float64)


def resolve_single_original(original_dir: Path) -> Path:
    images = sorted([p for p in original_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXT])
    if len(images) != 1:
        raise RuntimeError(
            f"Stress mode expects exactly 1 image in {original_dir}, found {len(images)}"
        )
    return images[0]


def main() -> int:
    # Configuration block (editable)
    data_path = REPO_ROOT / "data_stress1"
    count = 256
    n_points = 1024
    n_iters = 1000
    threshold = 255
    image_size = None
    invert_image = False
    invert_density = False
    point_size = 1.0
    coord_mode = "auto"
    overwrite = True
    keep_txt = False
    apply_quantization = False
    quantization_count = 4

    parser = argparse.ArgumentParser(
        description="Generate standalone GBN stress dataset (source/target + db/db_full.hdf5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path", type=Path, default=data_path)
    parser.add_argument("--count", type=int, default=count, help="Number of stress duplicates (typically 256 or 1024)")
    parser.add_argument("--n_points", type=int, default=n_points)
    parser.add_argument("--n_iters", type=int, default=n_iters)
    parser.add_argument("--threshold", type=int, default=threshold)
    parser.add_argument("--image_size", type=int, nargs=2, default=image_size, metavar=("W", "H"))
    parser.add_argument("--invert_image", action=argparse.BooleanOptionalAction, default=invert_image)
    parser.add_argument("--invert_density", action=argparse.BooleanOptionalAction, default=invert_density)
    parser.add_argument("--point_size", type=float, default=point_size)
    parser.add_argument("--coord_mode", type=str, default=coord_mode, choices=["auto", "unit", "aspect"])
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=overwrite)
    parser.add_argument("--keep_txt", action=argparse.BooleanOptionalAction, default=keep_txt)
    parser.add_argument("--apply_quantization", action=argparse.BooleanOptionalAction, default=apply_quantization)
    parser.add_argument("--quantization_count", type=int, default=quantization_count)

    args = parser.parse_args()

    if args.count <= 0:
        print("Error: --count must be > 0", file=sys.stderr)
        return 1

    data_root = args.data_path.resolve()
    original_dir = data_root / "original"
    source_dir = data_root / "source"
    target_dir = data_root / "target"
    db_dir = data_root / "db"
    hdf5_path = db_dir / "db_full.hdf5"

    if not original_dir.is_dir():
        print(f"Error: missing original/ under {data_root}", file=sys.stderr)
        return 1

    try:
        src_image = resolve_single_original(original_dir)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    image_size_tuple = tuple(args.image_size) if args.image_size is not None else None
    src_gray = load_gray(src_image, image_size=image_size_tuple)

    if args.invert_image:
        src_gray = 255 - src_gray
    if args.apply_quantization:
        src_gray = quantize_gray(src_gray, args.quantization_count)

    density = make_density(src_gray, invert_density=args.invert_density, threshold=args.threshold)

    h_px, w_px = src_gray.shape
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)

    stem = src_image.stem
    all_points = []
    all_data_t = []

    done = 0
    skipped = 0
    failed = 0

    for i in range(1, args.count + 1):
        name = f"{stem}_{i}"
        source_out = source_dir / f"{name}.png"
        target_out = target_dir / f"{name}.png"
        txt_out = target_dir / f"{name}.txt"

        txt_ready = txt_out.exists() if args.keep_txt else True
        if not args.overwrite and source_out.exists() and target_out.exists() and txt_ready:
            skipped += 1
            # Even in skip mode, try to recover points for HDF5 build.
            if txt_out.exists():
                try:
                    pts = load_points(txt_out)
                    if pts.shape == (args.n_points, 2):
                        all_points.append(pts)
                        all_data_t.append(points_to_data_t(pts, args.n_points))
                except Exception:
                    pass
            continue

        cv2.imwrite(str(source_out), src_gray)

        try:
            run_gbn(density, txt_out, n_points=args.n_points, n_iters=args.n_iters)
            pts = load_points(txt_out)
            if pts.shape != (args.n_points, 2):
                raise RuntimeError(
                    f"Unexpected point shape for {name}: {pts.shape}, expected ({args.n_points}, 2)"
                )
            rendered = render_stipple(pts, w_px, h_px, point_size=args.point_size, coord_mode=args.coord_mode)
            cv2.imwrite(str(target_out), rendered)

            all_points.append(pts)
            all_data_t.append(points_to_data_t(pts, args.n_points))

            if not args.keep_txt:
                txt_out.unlink(missing_ok=True)

            done += 1
            print(f"[{i}/{args.count}] {name}: done")
        except Exception as exc:
            failed += 1
            print(f"[{i}/{args.count}] {name}: FAILED -> {exc}")

    if len(all_points) == 0:
        print("Error: no successful samples; HDF5 will not be written.", file=sys.stderr)
        return 2

    data_arr = np.stack(all_points, axis=0).astype(np.float64)
    data_t_arr = np.stack(all_data_t, axis=0).astype(np.float64)
    prop_vec = np.zeros((9,), dtype=np.float64)
    prop_vec[4] = 1.0
    prop_arr = np.tile(prop_vec[None, :], (data_arr.shape[0], 1))

    with h5py.File(hdf5_path, "w") as f:
        g = f.create_group("GBN")
        s = g.create_group(str(args.n_points))
        s.create_dataset("data", data=data_arr)
        s.create_dataset("data_t", data=data_t_arr)
        s.create_dataset("prop", data=prop_arr)

    print("\nSummary")
    print(f"  data root:   {data_root}")
    print(f"  source:      {source_dir}")
    print(f"  target:      {target_dir}")
    print(f"  processed:   {done}")
    print(f"  skipped:     {skipped}")
    print(f"  failed:      {failed}")
    print(f"  hdf5 path:   {hdf5_path}")
    print(f"  hdf5 shape:  data={data_arr.shape}, data_t={data_t_arr.shape}, prop={prop_arr.shape}")

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
