# created 2025/03/26
# A temporary script to check audio length

import os
import torchaudio
import pandas as pd

# get a list of all WAV files
aud_dir = "audio/English"
wav_files = [file for file in os.listdir(aud_dir) if file.endswith('.wav')]

# dictionary for audio length
num_frame_dict = {header: [] for header in ['file', 'num_frames']}

# iterate over all WAV files
for index, file in enumerate(wav_files):
    file_path = os.path.join(aud_dir, file)
    num_frames = torchaudio.info(file_path).num_frames
    num_frame_dict['file'].append(file)
    num_frame_dict['num_frames'].append(num_frames)
    if index % 1762 == 0:
        print(f"Went through {index / 1762}% of the files!")

# write dataframe to csv file
num_frame_df = pd.DataFrame(num_frame_dict)
df_path = "audio_length.xlsx"
num_frame_df.to_excel(df_path, index=False)
print(f"All done!")