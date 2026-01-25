# Analysis and Comparison Scripts

This folder contains scripts used to analyze and validate the C++ Gaussian Blue Noise implementation.

## Scripts Overview

### Comparison Scripts

#### `compare_cpp_results.py`
**Purpose:** Compare C++ GBN output vs Rougier's reference stippling

**What it does:**
- Loads C++ GBN point sets from txt files
- Loads Rougier reference images
- Computes density correlation metrics
- Creates side-by-side comparison visualizations

**Usage:**
```bash
python compare_cpp_results.py
```

**Outputs:**
- `out/comparison_cpp_vs_rougier_img1.png`
- `out/comparison_cpp_vs_rougier_img2.png`
- `out/comparison_cpp_vs_rougier_img3.png`
- Console: Correlation statistics

**Key Finding:** Confirmed GBN produces uniform distributions (correlation ~0.0) instead of density-following distributions (target: >0.6)

---

#### `compare_taksim.py`
**Purpose:** Validate C++ build against official taksim-circle reference

**What it does:**
- Compares reference taksim-circle.txt with our output
- Computes minimum distance statistics
- Creates visual comparison
- Validates compilation correctness

**Usage:**
```bash
# First generate output for taksim-circle
./gbn-adaptive taksim-circle.pgm 10000 100 out/taksim_result.txt

# Then compare
python compare_taksim.py
```

**Outputs:**
- `out/taksim_comparison.png`
- Console: Min distance statistics

**Key Finding:** Build is correct. Consistently produces min_dist ~0.004 regardless of parameters (iterations, sigma, init mode).

---

#### `check_capacity_constraint.py`
**Purpose:** Quantitative validation of density constraint following

**What it does:**
- Divides image into 32×32 grid
- Counts points per cell
- Measures correlation with source density
- Calculates high-density hit rate
- Creates scatter plots

**Usage:**
```bash
python check_capacity_constraint.py
```

**Metrics computed:**
- **Correlation:** Pearson correlation between density and point count
  - Good stippling: >0.6
  - Random: ~0.0
  - C++ GBN result: -0.018 to 0.102
- **High-density hit rate:** % of points in darkest 25% of regions
  - Good stippling: >70%
  - Random: ~25%
  - C++ GBN result: 42-55%

**Key Finding:** Proved C++ GBN does NOT follow density constraints. Points are uniformly distributed regardless of image darkness.

---

### Utility Scripts

#### `png_to_pgm.py`
**Purpose:** Convert PNG images to ASCII PGM format for C++ GBN

**What it does:**
- Loads PNG and converts to grayscale
- Writes ASCII PGM (P2) format
- Compatible with gbn-adaptive binary

**Usage:**
```bash
python png_to_pgm.py input.png output.pgm
```

**Note:** This functionality is now integrated into `Python/generate_cpp_gbn.py`. Kept as standalone utility for manual conversions.

---

#### `render_single_pixel.py`
**Purpose:** Render points as single pixels (Rougier-style visualization)

**What it does:**
- Loads point coordinates from txt files
- Creates white canvas
- Draws single black pixels at point locations
- Reports coverage statistics

**Usage:**
```bash
python render_single_pixel.py
```
(Hardcoded to process 3 sample images)

**Why it matters:** Unlike cairo rendering (circles), single-pixel rendering enables accurate density correlation measurements and fair comparison with Rougier's results.

---

## Analysis Workflow

The typical analysis workflow used these scripts:

```bash
# 1. Generate C++ GBN output
cd /groups/asharf_group/ofirgila/GaussianBlueNoise
python Python/generate_cpp_gbn.py data_grads_v3_sample/source/image.png \
    --points 5000 --iters 100 --render

# 2. Compare with references
python scripts/compare_cpp_results.py

# 3. Validate build correctness
./gbn-adaptive taksim-circle.pgm 10000 100 out/taksim_result.txt
python scripts/compare_taksim.py

# 4. Check capacity constraint
python scripts/check_capacity_constraint.py
```

---

## Key Results Summary

| Script | Metric | Expected | Actual | Status |
|--------|--------|----------|--------|--------|
| compare_cpp_results | Correlation | >0.6 | -0.018 to 0.102 | ❌ Failed |
| compare_taksim | Min distance | ~0.007 | ~0.004 | ✅ Consistent |
| check_capacity_constraint | Hit rate | >70% | 42-55% | ❌ Near random |

**Conclusion:** GBN produces excellent blue-noise spacing but does not follow density constraints.

---

## Dependencies

```bash
conda activate sd  # or your Python environment
pip install numpy pillow scipy matplotlib
```

---

## For More Information

- **FINAL_ANALYSIS.md** - Comprehensive project summary
- **SUMMARY_FOR_PRO.md** - Detailed parameter testing results
- **../Python/README.md** - Main utilities documentation (if exists)

---

**Last Updated:** January 26, 2026
