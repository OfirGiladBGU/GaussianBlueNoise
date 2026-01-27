# GBN Scripts Documentation

**Updated:** January 27, 2026  
**Status:** ✅ Working with gbn-adaptive-linux (CUDA compiled)

---

## 📁 Main Scripts (USE THESE)

### 1. `generate_stippling.py` ⭐ Main Tool

Generate stippling output from a grayscale density image.

**What it does:**
1. Converts input PNG to PGM format (ASCII P2) if needed
2. Runs `gbn-adaptive-linux` on the PGM file
3. Saves point coordinates to text file
4. Renders stippling visualization to PNG

**Usage:**
```bash
# Basic usage (outputs to ../output/)
python generate_stippling.py <input_image.png>

# Custom parameters
python generate_stippling.py <input_image.png> --points 5000 --iters 500

# Custom output name
python generate_stippling.py <input_image.png> --name my_output

# Custom point size for rendering
python generate_stippling.py <input_image.png> --point-size 2
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--points N` | 10000 | Number of points to generate |
| `--iters N` | 1000 | Optimization iterations |
| `--output DIR` | ../output | Output directory |
| `--name NAME` | input stem | Base name for output files |
| `--point-size S` | 1 | Point size for rendering |

**Outputs:**
- `<output>/<name>.txt` - Point coordinates (normalized 0-1)
- `<output>/<name>.png` - Rendered stippling image

---

### 2. `compare_stippling.py` 📊 Comparison Tool

Create side-by-side comparison images.

**What it does:**
1. Loads input density image (PNG or PGM)
2. Loads ground truth stippling (optional)
3. Loads GBN output (points .txt or rendered .png)
4. Creates a side-by-side comparison figure

**Usage:**
```bash
# Two-way comparison (input vs GBN)
python compare_stippling.py density.png gbn_output.txt

# Three-way comparison (input vs GT vs GBN)
python compare_stippling.py density.png gbn_output.txt --gt reference.png

# Custom output path
python compare_stippling.py density.png gbn_output.txt --output my_comparison.png
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--gt PATH` | None | Ground truth stippling image |
| `--output PATH` | auto | Output comparison image path |
| `--title TEXT` | auto | Title for the comparison |
| `--point-size S` | 1 | Point size if rendering from txt |

**Outputs:**
- `comparison_<name>.png` - Side-by-side comparison image

---

## 📁 Utility Scripts (REFERENCE)

### `png_to_pgm.py`
Convert PNG images to ASCII PGM format for GBN input.

```bash
python png_to_pgm.py input.png output.pgm
```

*Note: This functionality is integrated into `generate_stippling.py`.*

---

### Legacy Scripts (ARCHIVED)

The following scripts were used for previous analysis and may need updating:

- `analyze_comparison.py` - Statistical analysis
- `check_capacity_constraint.py` - Density correlation metrics
- `compare_cpp_results.py` - GBN vs Rougier comparison (old paths)
- `compare_taksim.py` - Build validation
- `render_single_pixel.py` - Single-pixel renderer

---

## 🚀 Quick Start Examples

### Generate stippling for a single image:
```bash
cd /groups/asharf_group/ofirgila/GaussianBlueNoise/scripts
conda activate sd
python generate_stippling.py ../data_grads_v3_sample/source/gen_gray_Wave_999930824_5143.png
```

### Generate and compare with ground truth:
```bash
# Generate
python generate_stippling.py ../data_grads_v3_sample/source/gen_gray_Wave_999930824_5143.png --name wave

# Compare
python compare_stippling.py \
    ../data_grads_v3_sample/source/gen_gray_Wave_999930824_5143.png \
    ../output/wave.txt \
    --gt ../data_grads_v3_sample/target/gen_gray_Wave_999930824_5143.png
```

### Batch process multiple images:
```bash
for img in ../data_grads_v3_sample/source/*.png; do
    python generate_stippling.py "$img" --points 10000 --iters 1000
done
```

---

## 📋 Requirements

- **gbn-adaptive-linux** binary compiled at repo root
- **CUDA-enabled GPU** available
- **sd conda environment** with:
  - numpy
  - matplotlib
  - PIL (Pillow)
  - scipy (for analysis scripts)

### Compilation (if needed):
```bash
ssh <compute-node>
conda activate sd
cd /groups/asharf_group/ofirgila/GaussianBlueNoise
nvcc -arch=sm_89 -Xcompiler "-O3" \
    -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib \
    -o gbn-adaptive-linux gbn-adaptive.cu -lcairo
```

---

## 📊 Performance Reference

| Points | Iterations | Time | Notes |
|--------|------------|------|-------|
| 10000 | 1000 | ~8s | Full quality |
| 10000 | 100 | ~0.9s | Fast, good quality |
| 5000 | 100 | ~0.3s | Quick preview |

*Tested on RTX 6000 (compute 8.9)*
