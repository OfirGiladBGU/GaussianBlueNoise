# GaussianBlueNoise - Complete Documentation

**Project Location:** `/groups/asharf_group/ofirgila/GaussianBlueNoise/`  
**Status:** ✅ Complete - Compiled and working on Linux/CUDA  
**Updated:** January 27, 2026

---

## Table of Contents

1. [Quick Start](#-quick-start)
2. [Project Structure](#-project-structure)
3. [Key Results](#-key-results)
4. [What We Did](#-what-we-did)
5. [Technical Details](#-technical-details)
6. [How to Use](#-how-to-use)
7. [Performance Reference](#-performance-reference)
8. [Requirements](#-requirements)

---

## 🚀 Quick Start

### Generate GBN stippling:
```bash
cd /groups/asharf_group/ofirgila/GaussianBlueNoise/scripts
conda activate sd
python generate_stippling.py ../data_grads_v3_sample/source/image.png --points 10000 --iters 1000
```

### Create comparison visualization:
```bash
python compare_stippling.py input.png ../output/result.txt --gt reference.png
```

### Run directly with binary (on compute node):
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
./gbn-adaptive-linux input.pgm 10000 1000 output.txt
```

---

## 📁 Project Structure

```
GaussianBlueNoise/
├── scripts/                         # Main tools (USE THESE)
│   ├── generate_stippling.py       ⭐ Generate GBN stippling from image
│   ├── compare_stippling.py        📊 Create side-by-side comparisons
│   ├── archive/                    📦 Old scripts
│   └── README.md                   📖 Scripts documentation
│
├── output/                          # Generated outputs
│   ├── *.txt                       (Point coordinates)
│   └── *.png                       (Rendered images)
│
├── data_grads_v3_sample/           # Test data
│   ├── source/                     (3 density map images)
│   ├── pgm/                        (Converted PGM files)
│   └── output/                     (GBN results for samples)
│
├── gbn-adaptive-linux              ⚙️ Compiled CUDA binary
├── gbn-adaptive.cu                 📄 Source code (modified for Linux)
├── gpu-point-set.h                 📄 GPU utilities header
│
├── DOCUMENTATION.md                📖 This file
├── README.md                       📖 Original documentation (do not modify)
└── README.txt                      📖 Original notes (do not modify)
```

---

## 📊 Key Results

### Validation Against Reference

Compared our output against reference `taksim-circle.txt`:

| Metric | Reference | Our Output | Similarity |
|--------|-----------|------------|------------|
| Number of points | 10000 | 10000 | 100% |
| NN distance mean | 0.007130 | 0.007121 | **99.9%** |
| NN distance std | 0.002419 | 0.002397 | **99.1%** |
| NN distance min | 0.003136 | 0.003884 | 76.2% |
| NN distance max | 0.028918 | 0.024041 | 83.1% |
| Histogram intersection | - | - | **98.1%** |
| **Overall similarity** | - | - | **99.0%** |

**Note:** Individual point coordinates differ (expected - algorithm is stochastic) but statistical properties match excellently.

### Sample Images Tested

| Image | Covered Area | Time | Status |
|-------|--------------|------|--------|
| Combined_Shape | 98.83% | 1.69s | ✅ |
| Radial_Cosine_Gradient | 64.24% | 10.16s | ✅ |
| Wave | 85.10% | 2.11s | ✅ |

**Conclusion:** GBN produces excellent density-following blue-noise stippling, achieving **99% statistical similarity** to reference outputs.

---

## ✅ What We Did

### 1. Source Code Modifications

**gbn-adaptive.cu** - Fixed Linux compatibility:
```cpp
// Changed from Windows-specific getopt to system header
#ifdef _WIN32
#include "getopt/getopt.h"
#else
#include <getopt.h>
#endif
```

**gpu-point-set.h** - Fixed compilation warning:
```cpp
// Removed redundant class qualifier in declaration
GPUPointSet(int N, int w = 1, int h = 1);  // was: GPUPointSet::GPUPointSet(...)
```

### 2. Compilation

```bash
# On compute node (cs-6000-04) with RTX 6000 GPU
conda activate sd
nvcc -arch=sm_89 -Xcompiler "-O3" \
    -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib \
    -o gbn-adaptive-linux gbn-adaptive.cu -lcairo
```

**Key requirements:**
- CUDA 12.8 (available on cluster)
- cairo library (installed via `conda install cairo -c conda-forge`)
- Correct GPU architecture flag (`-arch=sm_89` for RTX 6000)

### 3. Validation

Ran the example from README:
```bash
./gbn-adaptive-linux taksim-circle.pgm 10000 1000 test_output_final.txt
```

**Execution time:** 8.15 seconds (matches expected ~7.5s from README)

### 4. Scripts Created

- **`scripts/generate_stippling.py`** - Converts PNG → PGM, runs binary, renders output
- **`scripts/compare_stippling.py`** - Creates side-by-side comparison images

---

## 🔬 Technical Details

### Algorithm Overview

GBN optimizes point positions to match a target density while maintaining blue-noise properties:

1. **Kernel Optimization:** Adjusts shaping factors based on local density
2. **Position Optimization:** Gradient descent to balance attraction (to density) and repulsion (between points)
3. **Iteration:** Repeats until convergence (~1000 iterations)

### Why Previous Analysis Was Wrong

Earlier analysis concluded GBN "doesn't follow density" because:
1. Binary was compiled without correct GPU architecture (produced all zeros)
2. CUDA kernels failed silently
3. Results were actually from uninitialized memory

**Fix:** Compile with `-arch=sm_89` for RTX 6000 (compute capability 8.9).

### What GBN Does Well

1. **Density-Following Distribution** - Points concentrate in dark regions, sparse in light regions
2. **Blue-Noise Quality** - Excellent minimum spacing, no clumping or regular patterns
3. **Performance** - ~8 seconds for 10K points × 1000 iterations
4. **Reproducibility** - 99% similarity to reference outputs

---

## 🎯 How to Use

### Option 1: Python Scripts (Recommended)

```bash
cd /groups/asharf_group/ofirgila/GaussianBlueNoise/scripts
conda activate sd

# Generate stippling
python generate_stippling.py <image.png> --points 10000 --iters 1000 --name output_name

# Create comparison
python compare_stippling.py <input.png> <output.txt> --gt <reference.png>
```

### Option 2: Direct Binary (on compute node)

```bash
ssh <compute-node>
conda activate sd
cd /groups/asharf_group/ofirgila/GaussianBlueNoise
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
./gbn-adaptive-linux input.pgm 10000 1000 output.txt
```

---

## ⚡ Performance Reference

| Points | Iterations | Time | Quality |
|--------|------------|------|---------|
| 10000 | 1000 | ~8s | Full |
| 10000 | 100 | ~0.9s | Good |
| 5000 | 100 | ~0.3s | Preview |

*Tested on RTX 6000 (compute capability 8.9)*

### Recommended Configuration

```bash
./gbn-adaptive-linux <input.pgm> 10000 1000 <output.txt>
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| Points | 10000 | Good balance of quality/speed |
| Iterations | 1000 | Full convergence |
| Input format | PGM (ASCII P2) | Required by binary |

---

## 📋 Requirements

### Runtime:
- CUDA-enabled GPU
- sd conda environment
- cairo library (via conda)
- Python packages: numpy, matplotlib, PIL

### Recompilation (if needed):
```bash
conda activate sd
nvcc -arch=sm_<XX> -Xcompiler "-O3" \
    -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib \
    -o gbn-adaptive-linux gbn-adaptive.cu -lcairo
```

Replace `<XX>` with your GPU's compute capability (e.g., 89 for RTX 6000).

---

## 🔗 External References

- [GBN Paper (ACM TOG 2022)](https://abdallagafar.com/publications/gbn/)
- [Original GBN Repo](https://github.com/Atrix256/GaussianBlueNoise)
- [Scripts Documentation](scripts/README.md)
