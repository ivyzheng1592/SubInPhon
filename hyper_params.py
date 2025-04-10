"""
data structure:
- "Dataset"
    - {language} + "_" + {datatype} + "_" + {condition} + ".csv"
    - "audio"
        - {language}
            - {src}.mp3
            - {src}.wav
- "Results"
    - "trial_"{trial_num}
        - {datatype}
            - {language} + "_" + {condition} + "_" + "run_"{run_num} + "_seq2seq.pth"
            - {language} + "_" + {condition} + "_" + "run_"{run_num} + "_acc.csv"
            - {language} + "_" + {condition} + "_" + "run_"{run_num} + "_acc_plot.png"
            - {language} + "_" + {condition} + "_" + "run_"{run_num} + "_att_plot"
                - {src}_{trg}.png
"""

# Alphabet
sos_token = "<SOS>"
eos_token = "<EOS>"
unk_token = "<UNK>"
pad_token = "<PAD>"
special_tokens = [sos_token, eos_token, unk_token, pad_token]

# Audio preprocessing hyperparameters
sample_rate = 24000
n_samples = 24000
n_fft = 1024
hop_length = 256
n_mels = 128

# Dataset hyperparameters
data_split_ratio = [0.6, 0.2, 0.2]

# Model hyperparameters
encoder_embedding_dim = 15  # original=300
decoder_embedding_dim = 15  # original=300
hidden_dim = 8  # original=256
n_layers = 1
encoder_dropout = 0.1  # 0.0 is equivalent to Identity function
decoder_dropout = 0.1  # 0.0 is equivalent to Identity function
teacher_forcing_ratio = 0.5  # original=0.5

# Training hyperparameters
n_epochs = 30  # original=50. Note: must be >1 !!!
learning_rate = 5e-4
batch_size = 32  # original=32
#clip = 1.0