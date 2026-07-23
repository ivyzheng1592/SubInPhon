"""Create initial TextGrids for the audio stimuli."""

import csv
from pathlib import Path

import torchaudio


def create_textgrid(output_path: Path, duration: float, label: str) -> None:
    """Create one whole-word TextGrid."""
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

    output_path.write_text(textgrid_content, encoding="utf-8")


def get_wav_duration(wav_path: Path) -> float:
    """Measure one WAV file's duration."""
    waveform, sample_rate = torchaudio.load(str(wav_path))
    return waveform.shape[1] / sample_rate


def load_textgrid_mapping(mapping_file: Path) -> dict[str, str]:
    """Load the mapping from WAV stems to TextGrid labels."""
    mapping: dict[str, str] = {}
    with mapping_file.open(encoding="utf-8", newline="") as file:
        for row in csv.reader(file, delimiter="\t"):
            if len(row) >= 2:
                mapping[row[0]] = row[1]
    return mapping


def create_textgrids_for_folder(wav_folder: Path, mapping: dict[str, str]) -> None:
    """Create one TextGrid for every WAV file in a folder."""
    wav_files = sorted(wav_folder.glob("*.wav"))
    for wav_file in wav_files:
        duration = get_wav_duration(wav_file)
        if wav_file.stem not in mapping:
            raise ValueError(f"No mapping for {wav_file.stem}")
        label = mapping[wav_file.stem]
        textgrid_file = wav_folder / f"{wav_file.stem}.TextGrid"
        create_textgrid(textgrid_file, duration, label)
        print(f"Created: {textgrid_file.name} (duration: {duration:.2f}s)")
    print(f"Total TextGrid files created: {len(wav_files)}")


def main() -> None:
    # Set the directory containing the WAV files.
    wav_folder = Path("/mnt/data/Projects/subinphon/audio/EnglishBH_shortened")
    # Set the TextGrid label mapping file.
    mapping_file = Path("/mnt/data/Projects/subinphon/dataset/EnglishBH_textgrid.txt")

    print(" - Reading TextGrid mapping:")
    mapping = load_textgrid_mapping(mapping_file)

    print(" - Creating TextGrids:")
    create_textgrids_for_folder(wav_folder, mapping)


if __name__ == "__main__":
    main()
