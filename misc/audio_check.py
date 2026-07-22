#!/usr/bin/env python3
"""
Measure WAV file lengths and save a summary spreadsheet.
"""

from pathlib import Path

import librosa
import pandas as pd


def check_audio_lengths(audio_dir: Path, output_file: Path, max_name_length: int | None = None) -> None:
    # Get a list of all WAV files in the chosen folder.
    wav_files = sorted(path for path in audio_dir.iterdir() if path.suffix == ".wav")
    rows = []

    # Measure each file and optionally restrict to shorter names for subset checks.
    for index, wav_file in enumerate(wav_files):
        if max_name_length is not None and len(wav_file.name) >= max_name_length:
            continue
        audio, _ = librosa.load(str(wav_file))
        rows.append({"file": wav_file.name, "num_frames": len(audio)})
        if index and index % 500 == 0:
            print(f"Processed {index} files")

    # Write the collected frame counts to an Excel spreadsheet.
    pd.DataFrame(rows).to_excel(output_file, index=False)
    print(f"Saved audio length summary to {output_file}")


def main() -> None:
    # Edit these paths and settings as needed for the current run.
    # audio_dir: folder of WAV files to inspect.
    # output_file: spreadsheet where the frame counts will be saved.
    # max_name_length: optional filename-length filter for quick subset checks.
    check_audio_lengths(
        Path("/media/ldlmdl/A2AAE4B1AAE482E1/SSD_Documents/subinphon/EnglishBH").expanduser().resolve(),
        Path("audio_length.xlsx").expanduser().resolve(),
        max_name_length=None,
    )


if __name__ == "__main__":
    main()
