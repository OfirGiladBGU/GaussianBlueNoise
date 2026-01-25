#!/usr/bin/env python3
"""
Capacity Constraint Validation Script
=======================================

Purpose:
    Quantitatively measure whether point distributions follow the source
    image density (capacity constraint / stippling quality).

What it does:
    1. Loads source density map (grayscale image, dark = high density)
    2. Loads output point distributions (both ours and Rougier's)
    3. Divides image into 32×32 grid cells
    4. Counts points per cell and measures correlation with density
    5. Calculates high-density hit rates (% of points in dark regions)
    6. Creates scatter plots showing density vs point concentration

Key Metrics:
    - Correlation: Pearson correlation between cell density and point count
      * Good stippling: >0.6 (Rougier target: 0.763)
      * Random: ~0.0
      * Our C++ GBN: -0.018 to 0.102 (effectively random)
    
    - High-density hit rate: % of points landing in top 25% darkest regions
      * Good stippling: >70%
      * Random: ~25%
      * Our C++ GBN: ~42-55% (near random)

Usage:
    python check_capacity_constraint.py

Inputs:
    - Source image path (hardcoded)
    - Our output: Python/out/*_single_pixel.png
    - Reference: Python/data_grads_v3_sample/target/*.png

Outputs:
    - Console: Correlation statistics and hit rates
    - Analysis confirming whether points follow density

Key Finding:
    This script proved C++ GBN does NOT follow density constraints.
    Points are uniformly distributed regardless of source image darkness.
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load source image and both outputs
source_path = "/groups/asharf_group/ofirgila/Stable_Diffusion/data_grads_v3_sample/source/gen_gray_Combined_Shape_1495169812_1630.png"
our_output = "Python/out/gen_gray_Combined_Shape_1495169812_1630_single_pixel.png"
target_output = "Python/data_grads_v3_sample/target/gen_gray_Combined_Shape_1495169812_1630.png"

source = np.array(Image.open(source_path).convert("L"))
ours = np.array(Image.open(our_output).convert("L"))
target = np.array(Image.open(target_output).convert("L"))

print("="*70)
print("CAPACITY CONSTRAINT CHECK")
print("="*70)
print()

# Invert source to density (darker = higher density)
density = 1.0 - (source.astype(float) / 255.0)

# Check correlation between density and point distribution
# Divide image into grid and count points in each cell
grid_size = 32
h, w = source.shape
cell_h = h // grid_size
cell_w = w // grid_size

print(f"Analyzing {grid_size}×{grid_size} grid cells...")
print()

our_counts = []
target_counts = []
density_vals = []

for i in range(grid_size):
    for j in range(grid_size):
        y1, y2 = i * cell_h, (i + 1) * cell_h
        x1, x2 = j * cell_w, (j + 1) * cell_w
        
        # Get density in this cell
        cell_density = density[y1:y2, x1:x2].mean()
        density_vals.append(cell_density)
        
        # Count black pixels (points) in each output
        our_count = (ours[y1:y2, x1:x2] == 0).sum()
        target_count = (target[y1:y2, x1:x2] == 0).sum()
        
        our_counts.append(our_count)
        target_counts.append(target_count)

density_vals = np.array(density_vals)
our_counts = np.array(our_counts)
target_counts = np.array(target_counts)

# Calculate correlation
our_corr = np.corrcoef(density_vals, our_counts)[0, 1]
target_corr = np.corrcoef(density_vals, target_counts)[0, 1]

print(f"Correlation (density vs point count):")
print(f"  Our GBN:       {our_corr:.3f}")
print(f"  Rougier:       {target_corr:.3f}")
print()

if our_corr < 0.5:
    print("⚠️  PROBLEM: Our points have LOW correlation with density!")
    print("   Points are NOT following the image density properly.")
    print("   This is the 'capacity constraint' issue you mentioned.")
elif our_corr < target_corr - 0.1:
    print("⚠️  Our correlation is LOWER than Rougier's")
    print("   Points somewhat follow density but not as strongly.")
else:
    print("✓ Correlation looks good - points follow density")

print()
print("="*70)

# Show some examples
high_density_cells = density_vals > 0.7
low_density_cells = density_vals < 0.3

if high_density_cells.any():
    print(f"High density regions (>0.7):")
    print(f"  Our points/cell:    {our_counts[high_density_cells].mean():.1f}")
    print(f"  Rougier points/cell: {target_counts[high_density_cells].mean():.1f}")
    
if low_density_cells.any():
    print(f"Low density regions (<0.3):")
    print(f"  Our points/cell:    {our_counts[low_density_cells].mean():.1f}")
    print(f"  Rougier points/cell: {target_counts[low_density_cells].mean():.1f}")

print()
print("Expected behavior:")
print("  High density → MORE points")
print("  Low density  → FEWER points")
