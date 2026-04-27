from pydub import AudioSegment

def convert_mp3_to_wav(mp3_file, wav_file):
    audio = AudioSegment.from_file(mp3_file, format="mp3")
    audio.export(wav_file, format="wav")

    print(f"MP3 file '{mp3_file}' converted to WAV file '{wav_file}'.")