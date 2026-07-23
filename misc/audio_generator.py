#!/usr/bin/env python3
"""Generate audio stimuli with Amazon Polly."""

import subprocess
from pathlib import Path
from typing import Iterable

import pandas as pd
from pydub import AudioSegment


def collect_words(stimuli_file: Path, columns: Iterable[str]) -> list[str]:
    """Collect unique audio forms from the selected columns."""
    stimuli = pd.read_csv(stimuli_file)
    missing_columns = [column for column in columns if column not in stimuli.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {stimuli_file}: {', '.join(missing_columns)}")

    words: list[str] = []
    seen = set()
    for column in columns:
        for value in stimuli[column].dropna().astype(str):
            if value and value not in seen:
                words.append(value)
                seen.add(value)
    print(f"Found {len(words)} unique forms in {stimuli_file.name}")
    return words


def synthesize_word(word: str, output_file: Path, voice_id: str) -> None:
    """Synthesize one IPA form with Amazon Polly."""
    ssml = f'<speak><phoneme alphabet="ipa" ph="{word}"></phoneme></speak>'
    command = [
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
        str(output_file),
    ]
    subprocess.run(command, check=True)


def convert_mp3_to_wav(mp3_file: str, wav_file: str) -> None:
    """Convert one MP3 file to WAV."""
    audio = AudioSegment.from_file(mp3_file, format="mp3")
    audio.export(wav_file, format="wav")


def generate_audio(words: list[str], audio_dir: Path, voice_id: str) -> None:
    """Generate MP3 files with Amazon Polly."""
    audio_dir.mkdir(parents=True, exist_ok=True)

    for word in words:
        mp3_file = audio_dir / f"{word}.mp3"
        wav_file = audio_dir / f"{word}.wav"
        if mp3_file.exists() or wav_file.exists():
            print(f"Skipping existing audio: {word}")
            continue
        synthesize_word(word, mp3_file, voice_id)
        print(f"Generated {mp3_file.name}")


def convert_audio(audio_dir: Path) -> None:
    """Convert generated MP3 files to WAV."""
    for mp3_file in sorted(audio_dir.glob("*.mp3")):
        wav_file = audio_dir / f"{mp3_file.stem}.wav"
        if wav_file.exists():
            print(f"Skipping existing WAV: {wav_file.name}")
            continue
        convert_mp3_to_wav(str(mp3_file), str(wav_file))
        print(f"Generated {wav_file.name}")


def main() -> None:
    # Set the stimulus list used to collect audio forms.
    stimuli_csv = Path("EnglishBH_aud_harmony.csv").expanduser().resolve()
    # Set the external directory where MP3 and WAV files will be written.
    audio_dir = Path("/mnt/data/Projects/SubInPhon/audio/EnglishBH")
    # Set the dataset columns to read as audio reference forms.
    columns = ("ur_var", "sr_var")
    # Set the Amazon Polly voice.
    voice_id = "Danielle"
    # convert_to_wav: whether generated MP3 files should also be converted to WAV.
    convert_to_wav = True

    print(" - Finding words:")
    words = collect_words(stimuli_csv, columns)

    print(" - Generating MP3 files with Amazon Polly:")
    generate_audio(words, audio_dir, voice_id)

    if not convert_to_wav:
        return

    print(" - Converting MP3 files to WAV:")
    convert_audio(audio_dir)


if __name__ == "__main__":
    main()
