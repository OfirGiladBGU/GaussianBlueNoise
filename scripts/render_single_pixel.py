#!/usr/bin/env python3
"""
Single-Pixel Point Renderer
============================

Purpose:
    Render point sets as single black pixels on white background,
    matching Rougier's visualization style for fair comparison.

What it does:
    1. Loads point coordinates from txt file (normalized 0-1 range)
    2. Creates white image canvas
    3. Converts normalized coords to pixel coordinates
    4. Draws single black pixels at each point location
    5. Reports coverage statistics

Usage:
    python render_single_pixel.py
    (Hardcoded to process all 3 sample images)

Key Feature:
    Unlike cairo rendering (which draws circles), this creates
    pixel-perfect visualizations suitable for density correlation
    analysis and direct visual comparison with Rougier's results.

Note:
    This rendering style was essential for capacity constraint
    validation, allowing accurate correlation measurements.
"""

import numpy as np
from PIL import Image
import sys

def render_points_as_pixels(points_file, output_image, img_size=512):
    """Render points as single black pixels on white background (like Rougier)."""
    # Load points
    with open(points_file, 'r') as f:
        n_points = int(f.readline().strip())
        points = []
        for line in f:
            x, y = map(float, line.strip().split())
            points.append([x, y])
    
    points = np.array(points)
    print(f"Loaded {len(points)} points from {points_file}")
    print(f"  X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    
    # Create white image
    img = np.ones((img_size, img_size), dtype=np.uint8) * 255
    
    # Convert normalized coords to pixels
    px = (points[:, 0] * img_size).astype(int)
    py = (points[:, 1] * img_size).astype(int)
    
    # Clip to image bounds
    px = np.clip(px, 0, img_size - 1)
    py = np.clip(py, 0, img_size - 1)
    
    # Draw single black pixels
    img[py, px] = 0
    
    # Save
    Image.fromarray(img).save(output_image)
    print(f"Saved to {output_image}")
    print(f"  Black pixels: {(img == 0).sum()}")
    print(f"  Coverage: {(img == 0).sum() / img.size * 100:.3f}%")

if __name__ == "__main__":
    import pathlib
    
    out_dir = pathlib.Path("Python/out")
    
    # Render all 3 images with single-pixel points
    images = [
        "gen_gray_Combined_Shape_1495169812_1630",
        "gen_gray_Radial_Cosine_Gradient_1629573513_7601",
        "gen_gray_Wave_999930824_5143",
    ]
    
    for img_name in images:
        pts_file = out_dir / f"{img_name}.txt"
        out_file = out_dir / f"{img_name}_single_pixel.png"
        
        print(f"\nProcessing {img_name}...")
        render_points_as_pixels(str(pts_file), str(out_file), 512)
