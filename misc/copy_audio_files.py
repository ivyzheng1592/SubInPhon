#!/usr/bin/env python3
"""
Copy audio files referenced by a dataset CSV into a target directory.
"""

import shutil
from pathlib import Path

import pandas as pd


def copy_audio_files(csv_file: Path, source_dir: Path, target_dir: Path, column: str) -> None:
    # Read the dataset CSV and use one reference column to decide which files to copy.
    df = pd.read_csv(csv_file)

    # Check that the chosen reference column exists.
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {csv_file}")

    # Create the target folder if needed.
    target_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    missing_count = 0
    # Copy one WAV file per unique reference value in the selected column.
    for audio_ref in df[column].dropna().astype(str).unique():
        source_path = source_dir / f"{audio_ref}.wav"
        target_path = target_dir / f"{audio_ref}.wav"

        if source_path.exists():
            shutil.copy2(source_path, target_path)
            print(f"Copied: {audio_ref}.wav")
            copied_count += 1
        else:
            print(f"Warning: source file not found: {source_path}")
            missing_count += 1

    print("\nSummary:")
    print(f"Files copied: {copied_count}")
    print(f"Files missing: {missing_count}")


def main() -> None:
    # Edit these paths and settings as needed for the current run.
    # First path: dataset CSV to read.
    # Second path: source folder containing the full audio inventory.
    # Third path: target folder for the copied subset.
    # Final argument: which CSV column to use as the audio reference.
    copy_audio_files(
        Path("/mnt/data/Projects/subinphon/dataset/EnglishBH_shortened_harmony.csv"),
        Path("/mnt/data/Projects/SubInPhon/audio/EnglishBH"),
        Path("/mnt/data/Projects/SubInPhon/audio/EnglishBH_shortened"),
        "ur_var",
    )


if __name__ == "__main__":
    main()
