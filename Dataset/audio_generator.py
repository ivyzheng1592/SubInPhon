# created 2025/02/24 from previous project
# A script to generate stimuli with Amazon Polly
# updated 2025/02/26
# Converting Polly mp3 output to wav

import os
from mp3_to_wav import convert_mp3_to_wav

# open file and select the first 200 pairs to generate audio files
fileName = "EnglishBH_aud_harmony.csv"
with open(fileName, "r") as file:
    lines = file.readlines()[1:2]

# store words in a list
print(" - Reading word list:")
words = []
#prev_stem = lines[0].split(",")[0]
#words.append(prev_stem)
for l in lines:
    l = l.split("\n")[0]
    #stem = l.split(",")[0]
    suffixed_ur = l.split(",")[1]
    suffixed_sr = l.split(",")[2]
    #if stem != prev_stem:
        #words.append(stem)
        #prev_stem = stem
    words.append(suffixed_ur)
    if suffixed_sr != suffixed_ur:
        words.append(suffixed_sr)
print(words)

# create audio folder
print(" - Creating audio folder:")
folderName = "audio/English"
if not os.path.exists(folderName):
    os.mkdir(folderName)

# generate audio files
print(" - Generating audio files:")
for w in words:
    cmd = "aws polly synthesize-speech " \
          "--engine neural " \
          "--text-type ssml " \
          "--text '<speak><phoneme alphabet=\"ipa\" ph=\"" + w + "\"></phoneme></speak>' " \
          "--output-format mp3 " \
          "--voice-id Danielle " \
          + folderName + "/" + w + ".mp3"
    os.system(cmd)
    print(cmd)

# convert mp3 files to wav files
print(" - Converting audio files:")
for filename in os.listdir(folderName):
    if filename.endswith(".mp3"):
        mp3_file = os.path.join(folderName, filename)
        wav_file = os.path.join(folderName, os.path.splitext(filename)[0] + ".wav")
        convert_mp3_to_wav(mp3_file, wav_file)