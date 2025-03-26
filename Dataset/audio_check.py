# created 2025/03/26
# A temporary script to check audio length

import os
import random
import torchaudio

# get a list of all WAV files
dir = "audio/English"
wav_files = [file for file in os.listdir(dir) if file.endswith('.wav')]

# randomly select 10% of the WAV files
selected_files = random.sample(wav_files, int(0.1 * len(wav_files)))

# placeholder for the maximum audio length
max_num_frames = 0

# iterate over the selected files
for file in selected_files:
    file_path = os.path.join(dir, file)
    num_frames = torchaudio.info(file_path).num_frames
    print(num_frames)
    if num_frames > max_num_frames:
        max_num_frames = num_frames

print(max_num_frames)