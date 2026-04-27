#!/usr/bin/env python3
"""
Script to create TextGrid files for each .wav file in a folder.
Each TextGrid will have one interval tier containing an English-letter label.
The mapping file path is configured inline in the main() function.
"""

import csv
from pathlib import Path
import torchaudio


def create_textgrid(wav_filename, output_path, duration=1.0, label=None):
    """
    Create a TextGrid file with one interval tier.

    Args:
        wav_filename (str): Name of the wav file (without path, e.g., "pipi.wav")
        output_path (str): Full path where to save the TextGrid file
        duration (float): Duration of the interval tier (default 1.0 seconds)
        label (str, optional): Text label to fill into the interval tier.
            If omitted, the filename stem is used.
    """
    # Extract the label from filename if not provided
    if label is None:
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


def load_textgrid_mapping(mapping_path):
    """
    Load a mapping from wav filename stem to English-letter label.

    Args:
        mapping_path (str): Path to the EnglishBH_textgrid.txt file

    Returns:
        dict[str, str]: Mapping from sound name to English text
    """
    mapping = {}
    with open(mapping_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 2:
                continue
            key = row[0].strip()
            value = row[1].strip()
            if key:
                mapping[key] = value
    return mapping


def create_textgrids_for_folder(wav_folder, mapping_file):
    """
    Create TextGrid files for all .wav files in a folder.

    Args:
        wav_folder (str): Folder containing .wav files (TextGrids will be saved here)
        mapping_file (str): Path to the EnglishBH_textgrid.txt file.
    """
    wav_folder = Path(wav_folder)
    mapping_path = Path(mapping_file)

    if not wav_folder.exists():
        raise ValueError(f"WAV folder not found: {wav_folder}")

    if not mapping_path.exists():
        raise ValueError(f"Mapping file not found: {mapping_path}")

    output_folder = wav_folder

    print(f"Loading English-text mapping from {mapping_path}")
    mapping = load_textgrid_mapping(str(mapping_path))

    # Find all .wav files
    wav_files = sorted(wav_folder.glob("*.wav"))

    if not wav_files:
        print(f"No .wav files found in {wav_folder}")
        return

    created_count = 0

    for wav_file in wav_files:
        # Get the duration of the wav file
        wav_duration = get_wav_duration(str(wav_file))

        # Determine the label from the mapping or fallback to filename stem
        label = mapping.get(wav_file.stem, wav_file.stem)

        # Create corresponding TextGrid filename
        textgrid_filename = wav_file.stem + ".TextGrid"
        textgrid_path = output_folder / textgrid_filename

        create_textgrid(wav_file.name, str(textgrid_path), duration=wav_duration, label=label)
        print(f"Created: {textgrid_filename} (duration: {wav_duration:.2f}s)")
        created_count += 1

    print(f"\nTotal TextGrid files created: {created_count}")


def main():
    # Configure these paths inline:
    wav_folder = "/mnt/data/Projects/subinphon/dataset/EnglishBH_shortened"
    mapping_file = "/mnt/data/Projects/subinphon/dataset/EnglishBH_textgrid.txt"
    create_textgrids_for_folder(wav_folder, mapping_file=mapping_file)


if __name__ == "__main__":
    main()
