#!/usr/bin/env python3
"""
PNG to PGM Converter Utility
=============================

Purpose:
    Convert PNG images to ASCII PGM (P2) format for C++ GBN input.
    The C++ gbn-adaptive binary requires PGM format for density maps.

What it does:
    1. Loads PNG image and converts to grayscale
    2. Extracts pixel values (0-255)
    3. Writes ASCII PGM format:
       - P2 header (ASCII grayscale)
       - Width and height
       - Max value (255)
       - Pixel data in text format (70 chars per line)

Usage:
    python png_to_pgm.py input.png output.pgm

Note:
    This functionality is now integrated into generate_cpp_gbn.py.
    Kept as standalone utility for manual conversions if needed.
"""
from PIL import Image
import sys

if len(sys.argv) != 3:
    print("Usage: python png_to_pgm.py input.png output.pgm")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

# Load PNG and convert to grayscale
img = Image.open(input_path).convert('L')

# Get dimensions
w, h = img.size

# Get pixel data
pixels = list(img.getdata())

# Write ASCII PGM format
with open(output_path, 'w') as f:
    f.write(f"P2\n")
    f.write(f"{w} {h}\n")
    f.write(f"255\n")
    
    # Write pixels (ASCII format, 70 chars per line)
    for i, pixel in enumerate(pixels):
        f.write(str(pixel))
        if (i + 1) % 70 == 0:
            f.write('\n')
        else:
            f.write(' ')
    f.write('\n')

print(f"Converted {input_path} to {output_path}")
print(f"Dimensions: {w}x{h}")
