#!/usr/bin/env python3
"""
Script to copy audio files based on ur_ref column in CSV file.
Copies files from source_dir to target_dir where filenames match ur_ref + '.wav'
"""

import os
import shutil
import pandas as pd
import argparse

def copy_audio_files(csv_file, source_dir, target_dir):
    """
    Copy audio files based on ur_ref column in CSV.

    Args:
        csv_file (str): Path to the CSV file containing ur_ref column
        source_dir (str): Source directory containing the .wav files
        target_dir (str): Target directory to copy files to
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Check if ur_ref column exists
    if 'ur_ref' not in df.columns:
        raise ValueError(f"Column 'ur_ref' not found in {csv_file}")

    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Get unique ur_ref values
    ur_refs = df['ur_ref'].unique()

    copied_count = 0
    missing_count = 0

    for ur_ref in ur_refs:
        # Construct source and target paths
        source_path = os.path.join(source_dir, f"{ur_ref}.wav")
        target_path = os.path.join(target_dir, f"{ur_ref}.wav")

        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
            print(f"Copied: {ur_ref}.wav")
            copied_count += 1
        else:
            print(f"Warning: Source file not found: {source_path}")
            missing_count += 1

    print(f"\nSummary:")
    print(f"Files copied: {copied_count}")
    print(f"Files missing: {missing_count}")

if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser(description="Copy audio files based on ur_ref column in CSV")
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument("source_dir", help="Source directory containing .wav files")
    parser.add_argument("target_dir", help="Target directory to copy files to")

    args = parser.parse_args()
    """
    copy_audio_files("/mnt/data/Projects/subinphon/dataset/EnglishBH_shortened_harmony.csv", 
                     "/media/ldlmdl/A2AAE4B1AAE482E1/SSD_Documents/subinphon/EnglishBH",
                     "/mnt/data/Projects/subinphon/dataset")