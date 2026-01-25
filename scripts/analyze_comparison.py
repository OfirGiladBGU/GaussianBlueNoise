#!/usr/bin/env python3
"""
Coverage Diagnostic Script (Our GBN vs Rougier)
================================================

Purpose:
    Quick diagnostic to compare coverage (black pixel count) between
    our GBN-rendered output and Rougier's reference stippling. This was
    used early in the analysis to identify potential rendering and
    density-following issues.

What it checks:
    - Counts black pixels in our output vs Rougier's target
    - Computes coverage ratio (Our/Target)
    - Prints guidance based on ratio thresholds

Usage:
    Paths are currently hardcoded to a specific sample. Adjust the
    `source_path`, `our_output`, and `target_output` below as needed.
    Future improvement would add argparse parameters for paths.

Notes:
    - This is a coarse diagnostic (coverage only). For full density
      correlation analysis, use scripts/check_capacity_constraint.py.
    - Our main comparison visuals are in scripts/compare_cpp_results.py.
"""

from PIL import Image
import numpy as np
import sys

# Load images
source_path = "/groups/asharf_group/ofirgila/Stable_Diffusion/data_grads_v3_sample/source/gen_gray_Combined_Shape_1495169812_1630.png"
our_output = "/groups/asharf_group/ofirgila/GaussianBlueNoise/Python/out/gen_gray_Combined_Shape_1495169812_1630.png"
target_output = "/groups/asharf_group/ofirgila/GaussianBlueNoise/Python/data_grads_v3_sample/target/gen_gray_Combined_Shape_1495169812_1630.png"

source = np.array(Image.open(source_path).convert("L"))
ours = np.array(Image.open(our_output).convert("L"))
target = np.array(Image.open(target_output).convert("L"))

print("="*70)
print("DIAGNOSIS: Our GBN vs Rougier Voronoi")
print("="*70)
print()

print("Source image (grayscale input):")
print(f"  Shape: {source.shape}")
print(f"  Brightness: {source.min()} - {source.max()} (mean: {source.mean():.1f})")
print(f"  Dark pixels (< 128): {(source < 128).sum():,} ({(source < 128).sum()/source.size*100:.1f}%)")
print()

print("Our GBN output (stipple):")
print(f"  Shape: {ours.shape}")
print(f"  Black dots (< 128): {(ours < 128).sum():,} pixels")
print(f"  Coverage: {(ours < 128).sum()/ours.size*100:.3f}%")
print()

print("Rougier target (stipple):")
print(f"  Shape: {target.shape}")
print(f"  Black dots (< 128): {(target < 128).sum():,} pixels")
print(f"  Coverage: {(target < 128).sum()/target.size*100:.3f}%")
print()

# Calculate ratio
ratio = (ours < 128).sum() / max((target < 128).sum(), 1)

print("="*70)
print(f"COVERAGE RATIO (Our/Target): {ratio:.3f}x")
print("="*70)
print()

if ratio < 0.5:
    print("⚠️  PROBLEM IDENTIFIED: Our output has FAR FEWER dots!")
    print()
    print("Possible causes:")
    print("  1. RENDERING: Points drawn too small (--black parameter too low)")
    print("  2. DENSITY NOT FOLLOWED: Points not concentrating in dark regions")
    print("  3. INITIALIZATION: Points starting uniformly instead of density-based")
    print()
    print("Rougier's method (Voronoi) has strong 'capacity constraint':")
    print("  - Each Voronoi cell has one point")
    print("  - Cell sizes inversely proportional to density")
    print("  - Small cells in dark areas = many points")
    print("  - Large cells in light areas = few points")
    print()
    print("Our GBN may be missing this constraint!")
    
elif ratio > 2.0:
    print("⚠️  PROBLEM: Too many dots compared to Rougier")
    print("   Points may not be avoiding white/light regions properly")
else:
    print("✓ Coverage ratio is reasonable")
    print("  Problem may be in spatial distribution quality, not quantity")

print()
print("NEXT STEPS:")
print("1. Check if points are actually following density map")
print("2. Increase --black ratio in rendering (try 0.3-0.5)")
print("3. Verify density map inversion is correct")
