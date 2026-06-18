#!/usr/bin/env python3
"""Run gbn_data_gen.py for all ICONS-TIMES-V2 n_points values.

After each run, the output subfolders (target, timestamps) are renamed
with a _GBN_<N_POINTS> postfix. original/ and source/ are left untouched.
"""

import subprocess
import sys
from pathlib import Path

# -- Configuration -------------------------------------------------------------

SCRIPT_PATH = Path(__file__).parent / "gbn_data_gen.py"

DATA_PATH = Path(
    "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion"
    "/experiments/outputs/icons_results_runtimes"
)

# ICONS - TIMES - V2  (actual point counts; comments show NxN equivalent)
N_POINTS = [
    256,    # 16
    576,    # 24
    1024,   # 32
    1600,   # 40
    2304,   # 48
    3136,   # 56
    4096,   # 64
    5184,   # 72
    6400,   # 80
    7744,   # 88
    9216,   # 96
    10816,  # 104
    12544,  # 112
]

# Subfolders produced by each run that should be renamed after completion
# original/ and source/ are left untouched
OUTPUT_SUBDIRS = ["target", "timestamps"]

# -- Main ----------------------------------------------------------------------

def main():
    if not SCRIPT_PATH.exists():
        print(f"ERROR: Script not found at {SCRIPT_PATH}")
        sys.exit(1)

    print(f"Running {len(N_POINTS)} n_points values: {N_POINTS}")
    print(f"Script: {SCRIPT_PATH}\n")

    for n_points in N_POINTS:
        print(f"\n{'='*70}")
        print(f"  N_POINTS = {n_points}")
        print(f"{'='*70}\n")

        cmd = [sys.executable, str(SCRIPT_PATH), "--n_points", str(n_points)]
        print(f"Command: {' '.join(cmd)}\n")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nWarning: n_points={n_points} exited with code {result.returncode}")

        for subdir in OUTPUT_SUBDIRS:
            src = DATA_PATH / subdir
            dst = DATA_PATH / f"{subdir}_GBN_{n_points}"
            if src.exists():
                src.rename(dst)
                print(f"[rename] {subdir}  ->  {dst.name}")

    print(f"\n{'='*70}")
    print("All n_points runs complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
