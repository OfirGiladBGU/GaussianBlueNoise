#!/usr/bin/env python3
"""
GBN Stippling Generator
========================

Purpose:
    Generate stippling output from a grayscale density image using the
    compiled gbn-adaptive-linux CUDA binary.

What it does:
    1. Converts input PNG to PGM format (ASCII P2) if needed
    2. Runs gbn-adaptive-linux on the PGM file
    3. Saves point coordinates to text file
    4. Renders stippling visualization to PNG

Usage:
    python generate_stippling.py <input_image> [options]

Arguments:
    input_image     Path to input grayscale image (PNG or PGM)

Options:
    --points N      Number of points to generate (default: 10000)
    --iters N       Number of optimization iterations (default: 1000)
    --output DIR    Output directory (default: ../output)
    --name NAME     Base name for output files (default: derived from input)
    --point-size S  Point size for rendering (default: 1)

Examples:
    # Basic usage
    python generate_stippling.py image.png
    
    # Custom parameters
    python generate_stippling.py image.png --points 5000 --iters 500 --name my_output

Outputs:
    <output>/<name>.txt     - Point coordinates (normalized 0-1)
    <output>/<name>.png     - Rendered stippling image

Requirements:
    - gbn-adaptive-linux binary compiled and available at repo root
    - CUDA-enabled GPU available
    - sd conda environment with numpy, matplotlib, PIL
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
GBN_BINARY = REPO_ROOT / "gbn-adaptive-linux"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output"


def png_to_pgm(input_path: Path, output_path: Path):
    """Convert PNG to ASCII PGM (P2) format."""
    img = Image.open(input_path).convert('L')
    w, h = img.size
    pixels = np.array(img)
    
    with open(output_path, 'w') as f:
        f.write('P2\n')
        f.write(f'{w} {h}\n')
        f.write('255\n')
        for row in pixels:
            for val in row:
                f.write(f'{val}\n')
    
    return w, h


def load_points(txt_path: Path):
    """Load points from GBN output text file."""
    with open(txt_path, 'r') as f:
        n = int(f.readline().strip())
        points = []
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                points.append([float(parts[0]), float(parts[1])])
    return np.array(points)


def render_stippling(points: np.ndarray, output_path: Path, point_size: float = 1):
    """Render points to PNG image."""
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.set_facecolor('white')
    ax.scatter(points[:, 0], points[:, 1], s=point_size, c='black', marker='.')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, facecolor='white', dpi=150)
    plt.close()


def run_gbn(pgm_path: Path, n_points: int, n_iters: int, output_txt: Path):
    """Run gbn-adaptive-linux binary."""
    if not GBN_BINARY.exists():
        raise FileNotFoundError(f"GBN binary not found at {GBN_BINARY}")
    
    # Get conda lib path for cairo
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    env = os.environ.copy()
    if conda_prefix:
        env['LD_LIBRARY_PATH'] = f"{conda_prefix}/lib:{env.get('LD_LIBRARY_PATH', '')}"
    
    cmd = [
        str(GBN_BINARY),
        str(pgm_path),
        str(n_points),
        str(n_iters),
        str(output_txt)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise RuntimeError(f"GBN binary failed with code {result.returncode}")
    
    print(result.stderr)  # GBN prints progress to stderr
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate GBN stippling from density image")
    parser.add_argument("input_image", type=Path, help="Input grayscale image (PNG or PGM)")
    parser.add_argument("--points", type=int, default=10000, help="Number of points (default: 10000)")
    parser.add_argument("--iters", type=int, default=1000, help="Optimization iterations (default: 1000)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--name", type=str, default=None, help="Base name for output files")
    parser.add_argument("--point-size", type=float, default=1, help="Point size for rendering (default: 1)")
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input_image.exists():
        print(f"Error: Input file not found: {args.input_image}")
        sys.exit(1)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Determine output base name
    if args.name:
        base_name = args.name
    else:
        base_name = args.input_image.stem
    
    output_txt = args.output / f"{base_name}.txt"
    output_png = args.output / f"{base_name}.png"
    
    # Convert to PGM if needed
    if args.input_image.suffix.lower() == '.pgm':
        pgm_path = args.input_image
    else:
        pgm_path = args.output / f"{base_name}_temp.pgm"
        print(f"Converting {args.input_image} to PGM...")
        png_to_pgm(args.input_image, pgm_path)
    
    # Run GBN
    print(f"\nGenerating {args.points} points with {args.iters} iterations...")
    run_gbn(pgm_path, args.points, args.iters, output_txt)
    
    # Clean up temp PGM
    if args.input_image.suffix.lower() != '.pgm' and pgm_path.exists():
        pgm_path.unlink()
    
    # Load and render points
    print(f"\nRendering stippling...")
    points = load_points(output_txt)
    render_stippling(points, output_png, args.point_size)
    
    print(f"\n✅ Done!")
    print(f"   Points: {output_txt}")
    print(f"   Image:  {output_png}")
    print(f"   Total points: {len(points)}")


if __name__ == "__main__":
    main()
