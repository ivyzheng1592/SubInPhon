# created 2025/03/26
# A temporary script to check audio length

import os
import librosa
import pandas as pd

# get a list of all WAV files
aud_dir = "/media/ldlmdl/A2AAE4B1AAE482E1/SSD_Documents/subinphon/EnglishBH"
wav_files = [file for file in os.listdir(aud_dir) if file.endswith('.wav')]

# dictionary for audio length
num_sample_dict = {header: [] for header in ['file', 'num_frames']}

# iterate over all WAV files
for index, file in enumerate(wav_files):
    if len(file) < 9:  # count only the audio files with less than five segments
        file_path = os.path.join(aud_dir, file)
        audio, _ = librosa.load(file_path)
        num_samples = len(audio)
        num_sample_dict['file'].append(file)
        num_sample_dict['num_frames'].append(num_samples)
    if index % 1762 == 0:
        print(f"Went through {index / 1762}% of the files!")

# write dataframe to csv file
num_sample_df = pd.DataFrame(num_sample_dict)
df_path = "audio_length.xlsx"
num_sample_df.to_excel(df_path, index=False)
print(f"All done!")