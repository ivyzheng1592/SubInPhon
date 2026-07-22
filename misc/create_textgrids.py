#!/usr/bin/env python3
"""
Create TextGrid files for each WAV file in a folder.
"""

import csv
from pathlib import Path

import torchaudio


def create_textgrid(wav_filename: str, output_path: Path, duration: float, label: str) -> None:
    # TextGrid content in Praat format with one interval tier covering the whole file.
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

    # Write the TextGrid file to disk.
    output_path.write_text(textgrid_content, encoding="utf-8")


def get_wav_duration(wav_path: Path) -> float:
    # Measure the wav duration so the TextGrid spans the entire audio file.
    waveform, sample_rate = torchaudio.load(str(wav_path))
    return waveform.shape[1] / sample_rate


def load_textgrid_mapping(mapping_path: Path) -> dict[str, str]:
    # Load a tab-separated mapping from wav stem to the English-letter label.
    mapping: dict[str, str] = {}
    with mapping_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            key = row[0].strip()
            value = row[1].strip()
            if key:
                mapping[key] = value
    return mapping


def create_textgrids_for_folder(wav_folder: Path, mapping_file: Path | None) -> None:
    # Validate that the WAV folder exists before scanning it.
    if not wav_folder.exists():
        raise ValueError(f"WAV folder not found: {wav_folder}")

    mapping = {}
    # Load the optional label mapping; otherwise fall back to wav stem labels.
    if mapping_file is not None:
        if not mapping_file.exists():
            raise ValueError(f"Mapping file not found: {mapping_file}")
        print(f"Loading mapping from {mapping_file}")
        mapping = load_textgrid_mapping(mapping_file)

    # Find all WAV files that need a matching TextGrid.
    wav_files = sorted(wav_folder.glob("*.wav"))
    if not wav_files:
        print(f"No .wav files found in {wav_folder}")
        return

    created_count = 0
    for wav_file in wav_files:
        # Use the actual wav duration and the mapped label when available.
        wav_duration = get_wav_duration(wav_file)
        label = mapping.get(wav_file.stem, wav_file.stem)
        textgrid_path = wav_folder / f"{wav_file.stem}.TextGrid"
        create_textgrid(wav_file.name, textgrid_path, wav_duration, label)
        print(f"Created: {textgrid_path.name} (duration: {wav_duration:.2f}s)")
        created_count += 1

    print(f"\nTotal TextGrid files created: {created_count}")


def main() -> None:
    # Edit these paths as needed for the current run.
    # wav_folder: location of the WAV files that need TextGrids.
    # mapping_file: optional TSV file mapping wav stems to display labels.
    wav_folder = Path("/mnt/data/Projects/subinphon/audio/EnglishBH_shortened").expanduser().resolve()
    mapping_file = Path("/mnt/data/Projects/subinphon/dataset/EnglishBH_textgrid.txt").expanduser().resolve()
    create_textgrids_for_folder(
        wav_folder,
        mapping_file,
    )


if __name__ == "__main__":
    main()
