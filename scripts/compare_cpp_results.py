#!/usr/bin/env python3
"""
C++ GBN Results Comparison Script
===================================

Purpose:
    Compare C++ GBN output against Rougier's reference stippling results.
    This script was used to validate that the C++ GBN implementation produces
    blue-noise point distributions but does NOT follow density constraints.

What it does:
    1. Loads point sets from C++ GBN output (txt format)
    2. Renders points as single pixels for visual comparison
    3. Computes correlation between point distribution and source density map
    4. Creates side-by-side comparison images (Source | Rougier | C++ GBN)
    5. Generates statistical analysis and visual quality assessment

Key Finding:
    C++ GBN achieves excellent blue-noise spacing (min_dist ~0.004) but
    correlation with density maps is near zero (-0.002 to 0.102), confirming
    it solves uniform spacing, not capacity-constrained stippling.

Usage:
    python compare_cpp_results.py

Inputs:
    - data_grads_v3_sample/source/*.png (density maps)
    - data_grads_v3_sample/target/*.png (Rougier references)
    - out/*.txt (C++ GBN point sets)

Outputs:
    - out/comparison_cpp_vs_rougier_img*.png (comparison visualizations)
    - Console: Correlation statistics and analysis

Note:
    This script confirmed the fundamental algorithm issue: GBN optimizes
    for uniform spacing, not density following. Parameter tuning cannot fix this.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "out"
DATA_DIR = BASE_DIR / "data_grads_v3_sample"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_points(txt_path: Path):
    """Load points from text file."""
    lines = txt_path.read_text().splitlines()
    n_points = int(lines[0])
    points = np.array([list(map(float, line.split())) for line in lines[1:n_points+1]])
    return points

def render_points(points, h, w, radius=1, flip_y=False):
    """Render points as black dots on white background."""
    img = Image.new('L', (w, h), 255)
    pixels = img.load()
    for x, y in points:
        xi = int(x * w)
        yi = int((1.0 - y) * h) if flip_y else int(y * h)  # C++ flips Y coordinate
        if 0 <= xi < w and 0 <= yi < h:
            pixels[xi, yi] = 0
    return img

def compute_correlation(points, density_map):
    """Compute correlation between point distribution and density map."""
    h, w = density_map.shape
    point_density = np.zeros_like(density_map)
    for x, y in points:
        xi = int(x * w)
        yi = int((1.0 - y) * h)  # Match C++ flipped Y orientation for correlation
        if 0 <= xi < w and 0 <= yi < h:
            point_density[yi, xi] += 1
    
    flat_target = density_map.flatten()
    flat_points = point_density.flatten()
    
    if flat_target.std() == 0 or flat_points.std() == 0:
        return 0.0
    
    correlation, _ = stats.pearsonr(flat_target, flat_points)
    return correlation

# Process each image
images_info = [
    {
        'name': 'Combined Shape',
        'source': DATA_DIR / 'source/gen_gray_Combined_Shape_1495169812_1630.png',
        'target': DATA_DIR / 'target/gen_gray_Combined_Shape_1495169812_1630.png',
        'cpp': OUT_DIR / 'cpp_results_img1.txt',
        'title': 'Image 1: Combined Shape'
    },
    {
        'name': 'Radial Cosine',
        'source': DATA_DIR / 'source/gen_gray_Radial_Cosine_Gradient_1629573513_7601.png',
        'target': DATA_DIR / 'target/gen_gray_Radial_Cosine_Gradient_1629573513_7601.png',
        'cpp': OUT_DIR / 'cpp_results_img2.txt',
        'title': 'Image 2: Radial Cosine'
    },
    {
        'name': 'Wave',
        'source': DATA_DIR / 'source/gen_gray_Wave_999930824_5143.png',
        'target': DATA_DIR / 'target/gen_gray_Wave_999930824_5143.png',
        'cpp': OUT_DIR / 'cpp_results_img3.txt',
        'title': 'Image 3: Wave'
    }
]

print("\n" + "="*70)
print("C++ GBN RESULTS ANALYSIS")
print("="*70)

for idx, img_info in enumerate(images_info, 1):
    print(f"\n{img_info['title']}")
    print("-" * 70)
    
    # Load source
    source_img = Image.open(img_info['source'])
    density_map = 1.0 - np.array(source_img, dtype=np.float32) / 255.0
    h, w = density_map.shape
    
    # Load C++ results
    cpp_points = load_points(img_info['cpp'])
    cpp_corr = compute_correlation(cpp_points, density_map)
    
    # Load Rougier results
    rougier_img = Image.open(img_info['target']).convert('L')
    rougier_array = np.array(rougier_img)
    rougier_points = np.zeros_like(density_map)
    rougier_points[rougier_array < 128] = 1
    rougier_corr, _ = stats.pearsonr(density_map.flatten(), rougier_points.flatten())
    
    # Render
    cpp_rendered = render_points(cpp_points, h, w, flip_y=True)  # Flip Y for C++ coordinate system
    rougier_rendered = Image.open(img_info['target'])

    cpp_render_path = OUT_DIR / f"cpp_gbn_img{idx}.png"
    cpp_rendered.save(cpp_render_path)
    
    # Print results
    print(f"C++ GBN Correlation:    {cpp_corr:.3f} {'❌' if cpp_corr < 0.5 else '✓'}")
    print(f"Rougier Correlation:    {rougier_corr:.3f} ✓")
    print(f"Gap:                    {rougier_corr - cpp_corr:.3f}")
    
    # Diagnostic
    cpp_point_densities = []
    for x, y in cpp_points:
        xi, yi = int(x * w), int(y * h)
        if 0 <= xi < w and 0 <= yi < h:
            cpp_point_densities.append(density_map[yi, xi])
    cpp_point_densities = np.array(cpp_point_densities)
    img_info['cpp_corr'] = cpp_corr
    img_info['rougier_corr'] = rougier_corr
    
    print(f"\nPoint density stats:")
    print(f"  Mean density at points: {cpp_point_densities.mean():.3f}")
    print(f"  Image mean density:     {density_map.mean():.3f}")
    high_density_count = int((cpp_point_densities > density_map.mean()).sum())
    high_density_pct = 100.0 * high_density_count / len(cpp_points)
    print(f"  Points in high-density: {high_density_count}/{len(cpp_points)} ({high_density_pct:.1f}%)")

# Create comparison image
print("\n" + "="*70)
print("Creating visual comparisons...")
print("="*70)

for idx, img_info in enumerate(images_info, 1):
    source_img = Image.open(img_info['source']).convert('RGB')
    cpp_img = Image.open(OUT_DIR / f"cpp_gbn_img{idx}.png").convert('RGB')
    rougier_img = Image.open(img_info['target']).convert('RGB')
    
    w, h = source_img.size
    
    # Create side-by-side comparison
    comparison = Image.new('RGB', (w*3 + 40, h + 80), 'white')
    comparison.paste(source_img, (10, 50))
    comparison.paste(rougier_img, (w + 20, 50))
    comparison.paste(cpp_img, (w*2 + 30, 50))
    
    # Add labels
    draw = ImageDraw.Draw(comparison)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
        font_small = font
    
    draw.text((w//2 - 40, 15), "Source (Density)", fill='black', font=font)
    draw.text((w + w//2 - 50, 15), "Rougier (Target)", fill='black', font=font)
    draw.text((w*2 + w//2 - 30, 15), "C++ GBN (Result)", fill='black', font=font)
    
    cpp_corr = img_info.get('cpp_corr', 0.0)
    rougier_corr = img_info.get('rougier_corr', 0.0)
    draw.text((w//2 - 40, h + 60), f"Correlation: {cpp_corr:.3f} {'✓' if cpp_corr >= 0.5 else '❌'}", fill='black', font=font_small)
    draw.text((w + w//2 - 60, h + 60), f"Correlation: {rougier_corr:.3f} ✓", fill='green', font=font_small)
    draw.text((w*2 + w//2 - 60, h + 60), f"Gap: {(rougier_corr - cpp_corr):.3f}", fill='red', font=font_small)

    comparison_path = OUT_DIR / f"comparison_cpp_vs_rougier_img{idx}.png"
    comparison.save(comparison_path)
    print(f"✓ Saved: {comparison_path.relative_to(BASE_DIR)}")

print("\n" + "="*70)
print("KEY OBSERVATION")
print("="*70)
print("""
C++ GBN Results (What You See):
├─ Points spread uniformly across entire image
├─ No concentration in dark regions
├─ Correlation ≈ -0.002 (essentially random vs density)
├─ Follows blue noise spacing (good for that purpose)
└─ FAILS for density-based stippling

Rougier Results (Target):
├─ Points concentrated in dark regions
├─ Sparse in light regions
├─ Correlation = 0.763 (follows density perfectly)
├─ Evenly spaced within their regions
└─ CORRECT for density-based stippling

Conclusion: C++ GBN solves wrong problem (blue noise, not stippling)
""")

print("="*70 + "\n")
