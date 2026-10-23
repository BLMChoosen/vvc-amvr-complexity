#!/usr/bin/env python3
"""
AMVR Dataset Merger
Merges multiple binary dataset files (.bin) into a single master dataset.
"""

import sys
import os
import glob
import struct

def merge_binary_datasets(input_files, output_file):
    if not input_files:
        print("[Merger] No input files provided.")
        return

    valid_files = [f for f in input_files if os.path.exists(f) and os.path.getsize(f) >= 16]
    if not valid_files:
        print("[Merger] No valid dataset files found.")
        return

    # Read master header from the first file
    with open(valid_files[0], 'rb') as f:
        master_header = f.read(16)

    magic = master_header[:4].decode('ascii', errors='ignore')
    if magic != 'AMVR':
        raise ValueError(f"Invalid dataset file {valid_files[0]} (magic: {magic})")

    version, num_features, sample_size = struct.unpack('<III', master_header[4:16])
    print(f"[Merger] Master Header: Magic={magic}, Version={version}, Features={num_features}, SampleSize={sample_size} bytes")

    total_samples = 0
    with open(output_file, 'wb') as out_f:
        # Write 16-byte header once
        out_f.write(master_header)

        for filepath in valid_files:
            file_size = os.path.getsize(filepath)
            payload_size = file_size - 16
            if payload_size <= 0:
                continue

            samples_in_file = payload_size // sample_size
            total_samples += samples_in_file
            print(f"  + Merging '{os.path.basename(filepath)}': {samples_in_file} samples")

            with open(filepath, 'rb') as in_f:
                in_f.seek(16) # Skip 16-byte header
                while True:
                    chunk = in_f.read(64 * 1024 * 1024) # 64MB streaming buffer
                    if not chunk:
                        break
                    out_f.write(chunk)

    print(f"\n[Merger] Success! Merged {len(valid_files)} files into '{output_file}' ({total_samples} total samples, {os.path.getsize(output_file)} bytes).")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python merge_datasets.py <output_file.bin> <input_file1.bin> [input_file2.bin ...]")
        print("Example: python merge_datasets.py master_dataset.bin dataset/*.bin")
        sys.exit(1)

    output = sys.argv[1]
    inputs = []
    for arg in sys.argv[2:]:
        expanded = glob.glob(arg)
        if expanded:
            inputs.extend(expanded)
        else:
            inputs.append(arg)

    merge_binary_datasets(inputs, output)
