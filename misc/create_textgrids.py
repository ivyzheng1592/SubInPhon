#!/usr/bin/env python3
"""
Script to create TextGrid files for each .wav file in a folder.
Each TextGrid will have one interval tier containing the filename (without .wav extension).
"""

import os
import argparse
from pathlib import Path
import torchaudio


def create_textgrid(wav_filename, output_path, duration=1.0):
    """
    Create a TextGrid file with one interval tier.

    Args:
        wav_filename (str): Name of the wav file (without path, e.g., "pipi.wav")
        output_path (str): Full path where to save the TextGrid file
        duration (float): Duration of the interval tier (default 1.0 seconds)
    """
    # Extract the label from filename (remove .wav extension)
    label = Path(wav_filename).stem

    # TextGrid content in Praat format
    textgrid_content = f"""File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = {duration}
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "Danielle"
        xmin = 0
        xmax = {duration}
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = {duration}
            text = "{label}"
"""

    # Write the TextGrid file
    with open(output_path, 'w') as f:
        f.write(textgrid_content)


def get_wav_duration(wav_path):
    """
    Get the duration of a .wav file in seconds.

    Args:
        wav_path (str): Path to the .wav file

    Returns:
        float: Duration in seconds
    """
    waveform, sample_rate = torchaudio.load(wav_path)
    duration = waveform.shape[1] / sample_rate
    return duration


def create_textgrids_for_folder(input_folder):
    """
    Create TextGrid files for all .wav files in a folder.

    Args:
        input_folder (str): Folder containing .wav files (TextGrids will be saved here)
    """
    input_folder = Path(input_folder)

    if not input_folder.exists():
        raise ValueError(f"Input folder not found: {input_folder}")

    output_folder = input_folder

    # Find all .wav files
    wav_files = sorted(input_folder.glob("*.wav"))

    if not wav_files:
        print(f"No .wav files found in {input_folder}")
        return

    created_count = 0

    for wav_file in wav_files:
        # Get the duration of the wav file
        wav_duration = get_wav_duration(str(wav_file))

        # Create corresponding TextGrid filename
        textgrid_filename = wav_file.stem + ".TextGrid"
        textgrid_path = output_folder / textgrid_filename

        create_textgrid(wav_file.name, str(textgrid_path), duration=wav_duration)
        print(f"Created: {textgrid_filename} (duration: {wav_duration:.2f}s)")
        created_count += 1

    print(f"\nTotal TextGrid files created: {created_count}")


if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser(
        description="Create TextGrid files for each .wav file in a folder"
    )
    parser.add_argument("input_folder", help="Folder containing .wav files")

    args = parser.parse_args()
    """
    create_textgrids_for_folder("/mnt/data/Projects/subinphon/dataset/EnglishBH_shortened")
