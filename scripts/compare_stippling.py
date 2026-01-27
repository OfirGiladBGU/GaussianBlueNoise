#!/usr/bin/env python3
"""
GBN Comparison Visualization Tool
==================================

Purpose:
    Create side-by-side comparison images showing:
    - Input grayscale density map (from PGM or PNG)
    - Ground Truth stippling (reference image, if available)
    - GBN output stippling

What it does:
    1. Loads input density image
    2. Loads ground truth stippling (optional)
    3. Loads GBN output points or renders from text file
    4. Creates a side-by-side comparison figure with labels

Usage:
    python compare_stippling.py <input_image> <gbn_output> [options]

Arguments:
    input_image     Path to input grayscale density image (PNG or PGM)
    gbn_output      Path to GBN output (either .txt points file or .png render)

Options:
    --gt PATH       Path to ground truth stippling image (optional)
    --output PATH   Output comparison image path (default: output/comparison_<name>.png)
    --title TEXT    Title for the comparison (default: derived from filename)
    --point-size S  Point size if rendering from txt (default: 1)

Examples:
    # Two-way comparison (input vs GBN)
    python compare_stippling.py density.png gbn_output.txt
    
    # Three-way comparison (input vs GT vs GBN)
    python compare_stippling.py density.png gbn_output.txt --gt reference.png
    
    # Custom output path
    python compare_stippling.py density.png gbn_output.txt --output my_comparison.png

Outputs:
    Comparison PNG image with side-by-side panels

Requirements:
    - sd conda environment with numpy, matplotlib, PIL
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output"


def load_pgm(pgm_path: Path):
    """Load PGM file and return as grayscale numpy array."""
    with open(pgm_path, 'r') as f:
        magic = f.readline().strip()
        if magic != 'P2':
            raise ValueError(f"Expected P2 PGM format, got {magic}")
        
        # Skip comments
        line = f.readline().strip()
        while line.startswith('#'):
            line = f.readline().strip()
        
        # Get dimensions
        dims = line.split()
        if len(dims) == 2:
            w, h = int(dims[0]), int(dims[1])
            max_val = int(f.readline().strip())
        else:
            w = int(dims[0])
            h = int(f.readline().strip())
            max_val = int(f.readline().strip())
        
        # Read pixel data
        pixels = []
        for line in f:
            pixels.extend([int(x) for x in line.split()])
        
        img = np.array(pixels, dtype=np.uint8).reshape((h, w))
        return img


def load_density_image(image_path: Path):
    """Load density image (PNG or PGM) as grayscale numpy array."""
    if image_path.suffix.lower() == '.pgm':
        return load_pgm(image_path)
    else:
        img = Image.open(image_path).convert('L')
        return np.array(img)


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


def render_points(points: np.ndarray, width: int, height: int, point_size: float = 1):
    """Render points to a matplotlib figure and return as numpy array."""
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    ax.set_facecolor('white')
    ax.scatter(points[:, 0], points[:, 1], s=point_size, c='black', marker='.')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Convert to numpy array (compatible with newer matplotlib)
    fig.canvas.draw()
    img = np.array(fig.canvas.buffer_rgba())[:, :, :3]  # RGB only, drop alpha
    plt.close()
    
    # Convert to grayscale
    gray = np.mean(img, axis=2).astype(np.uint8)
    return gray


def load_gbn_output(gbn_path: Path, ref_width: int, ref_height: int, point_size: float = 1):
    """Load GBN output - either as PNG image or render from text file."""
    if gbn_path.suffix.lower() == '.txt':
        points = load_points(gbn_path)
        return render_points(points, ref_width, ref_height, point_size)
    else:
        img = Image.open(gbn_path).convert('L')
        return np.array(img)


def create_comparison(input_img: np.ndarray, gbn_img: np.ndarray, 
                      gt_img: np.ndarray = None, title: str = "Stippling Comparison",
                      output_path: Path = None):
    """Create side-by-side comparison figure."""
    
    if gt_img is not None:
        # Three-way comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Input density (inverted for visualization - dark = high density)
        axes[0].imshow(255 - input_img, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title('Input Grayscale\n(density map)', fontsize=12)
        axes[0].axis('off')
        
        # Ground truth
        axes[1].imshow(gt_img, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title('Ground Truth\n(reference)', fontsize=12)
        axes[1].axis('off')
        
        # GBN output
        axes[2].imshow(gbn_img, cmap='gray', vmin=0, vmax=255)
        axes[2].set_title('GBN Output\n(generated)', fontsize=12)
        axes[2].axis('off')
    else:
        # Two-way comparison
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Input density (inverted for visualization)
        axes[0].imshow(255 - input_img, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title('Input Grayscale\n(density map)', fontsize=12)
        axes[0].axis('off')
        
        # GBN output
        axes[1].imshow(gbn_img, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title('GBN Output\n(generated)', fontsize=12)
        axes[1].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved comparison to: {output_path}")
    
    plt.close()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Create stippling comparison visualization")
    parser.add_argument("input_image", type=Path, help="Input grayscale density image (PNG or PGM)")
    parser.add_argument("gbn_output", type=Path, help="GBN output (.txt points file or .png render)")
    parser.add_argument("--gt", type=Path, default=None, help="Ground truth stippling image (optional)")
    parser.add_argument("--output", type=Path, default=None, help="Output comparison image path")
    parser.add_argument("--title", type=str, default=None, help="Comparison title")
    parser.add_argument("--point-size", type=float, default=1, help="Point size if rendering from txt")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.input_image.exists():
        print(f"Error: Input file not found: {args.input_image}")
        sys.exit(1)
    
    if not args.gbn_output.exists():
        print(f"Error: GBN output not found: {args.gbn_output}")
        sys.exit(1)
    
    if args.gt and not args.gt.exists():
        print(f"Error: Ground truth file not found: {args.gt}")
        sys.exit(1)
    
    # Load input density image
    print(f"Loading input: {args.input_image}")
    input_img = load_density_image(args.input_image)
    h, w = input_img.shape
    
    # Load GBN output
    print(f"Loading GBN output: {args.gbn_output}")
    gbn_img = load_gbn_output(args.gbn_output, w, h, args.point_size)
    
    # Load ground truth if provided
    gt_img = None
    if args.gt:
        print(f"Loading ground truth: {args.gt}")
        gt_img = np.array(Image.open(args.gt).convert('L'))
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_OUTPUT_DIR / f"comparison_{args.input_image.stem}.png"
    
    # Determine title
    if args.title:
        title = args.title
    else:
        title = f"Stippling Comparison: {args.input_image.stem}"
    
    # Create comparison
    print(f"\nCreating comparison...")
    create_comparison(input_img, gbn_img, gt_img, title, output_path)
    
    print(f"\n✅ Done!")
    print(f"   Output: {output_path}")


if __name__ == "__main__":
    main()
