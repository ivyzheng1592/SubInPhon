#!/usr/bin/env python3
"""
Generate audio stimuli with Amazon Polly from a dataset CSV.

Edit the paths in main() for the current run.
"""

import os
import subprocess
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from pydub import AudioSegment


DEFAULT_COLUMNS = ("ur_var", "sr_var")


def collect_words(csv_file: Path, columns: Iterable[str], limit: int | None = None) -> List[str]:
    # Read the dataset CSV and collect the audio-reference forms used for synthesis.
    df = pd.read_csv(csv_file)

    # Validate that the expected reference columns are present.
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {csv_file}: {', '.join(missing_columns)}"
        )

    # Optionally restrict the script to the first N rows for quick tests.
    if limit is not None:
        df = df.head(limit)

    # Keep unique forms only so the same audio file is not generated twice.
    words: List[str] = []
    seen = set()
    for column in columns:
        for value in df[column].dropna().astype(str):
            if value and value not in seen:
                words.append(value)
                seen.add(value)
    return words


def synthesize_word(word: str, output_mp3: Path, voice_id: str) -> None:
    # Wrap the IPA form in SSML so Polly reads the intended pronunciation.
    ssml = f'<speak><phoneme alphabet="ipa" ph="{word}"></phoneme></speak>'
    cmd = [
        "aws",
        "polly",
        "synthesize-speech",
        "--engine",
        "neural",
        "--text-type",
        "ssml",
        "--text",
        ssml,
        "--output-format",
        "mp3",
        "--voice-id",
        voice_id,
        str(output_mp3),
    ]
    subprocess.run(cmd, check=True)


def convert_mp3_to_wav(mp3_file: str, wav_file: str) -> None:
    # Load one MP3 file and export it as WAV.
    audio = AudioSegment.from_file(mp3_file, format="mp3")
    audio.export(wav_file, format="wav")
    print(f"MP3 file '{mp3_file}' converted to WAV file '{wav_file}'.")


def main() -> None:
    # Edit these paths and settings as needed for the current run.
    # csv_file: stimulus file whose ur_var/sr_var entries will be synthesized.
    # output_dir: folder where MP3 and WAV files will be written.
    # columns: dataset columns to read as audio reference forms.
    # limit: optional row cap for a quick smoke test.
    # voice_id: Polly voice used for synthesis.
    # convert_to_wav: whether generated MP3 files should also be converted to WAV.
    csv_file = Path("EnglishBH_aud_harmony.csv").expanduser().resolve()
    output_dir = Path("audio/English").expanduser().resolve()
    columns = list(DEFAULT_COLUMNS)
    limit = None
    voice_id = "Danielle"
    convert_to_wav = True

    output_dir.mkdir(parents=True, exist_ok=True)

    print(" - Reading stimulus list:")
    words = collect_words(csv_file, columns, limit=limit)
    print(f"Found {len(words)} unique forms in {csv_file.name}")

    print(" - Generating MP3 files with Amazon Polly:")
    # Generate one MP3 file per unique audio reference.
    for word in words:
        mp3_file = output_dir / f"{word}.mp3"
        if mp3_file.exists():
            print(f"Skipping existing MP3: {mp3_file.name}")
            continue
        synthesize_word(word, mp3_file, voice_id)
        print(f"Generated {mp3_file.name}")

    if convert_to_wav:
        print(" - Converting MP3 files to WAV:")
        # Convert all MP3 files in the folder so the training code can use WAV input.
        for mp3_file in sorted(output_dir.glob("*.mp3")):
            wav_file = output_dir / f"{mp3_file.stem}.wav"
            if wav_file.exists():
                print(f"Skipping existing WAV: {wav_file.name}")
                continue
            convert_mp3_to_wav(str(mp3_file), str(wav_file))
    else:
        print(" - Skipping MP3 to WAV conversion.")


if __name__ == "__main__":
    main()
