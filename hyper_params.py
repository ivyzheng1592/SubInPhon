"""
data structure:
- "subinphon_audio"
    - {language}
        - {src} + ".mp3"
        - {src} + ".wav"
- "Dataset"
    - {language} + "_" + {condition} + ".csv"
- "Results"
    - {trial_num} + "_" + {language}
        - {language} + "_acc.csv"
        - {language} + "_pred.csv"
        - {language} + "_acc_plots"
            - {language} + "_" + {condition} + "_run" + {run_num} + "_acc_plot.png"
        - {language} + "_model_files"
            - {language} + "_" + {condition} + "_run" + {run_num} + "_model_files"
                - {language} + "_" + {condition} + "_run" + {run_num} + "_epoch" + {epoch} + "_seq2seq.pth"
        - {language} + "_att_plots"
            - {language} + "_" + {condition} + "_run" + {run_num} + "_att_plots"
                - {language} + "_" + {condition} + "_run" + {run_num} + "_epoch" + {epoch} + "att_type.csv"
                - {language} + "_" + {condition} + "_run" + {run_num} + "_epoch" + {epoch} + {src}_{trg} + ".png"
        - {language} + "_embed_plots"
            - {language} + "_" + {condition} + "_run" + {run_num} + "_embed_plots"
                - {language} + "_" + {condition} + "_run" + {run_num} + "_epoch" + {epoch} + "_sr_vowel.csv"
                - {language} + "_" + {condition} + "_run" + {run_num} + "_epoch" + {epoch} + "_ur_vowel.csv"
                - {language} + "_" + {condition} + "_run" + {run_num} + "_epoch" + {epoch} + "_sr_vowel.png"
                - {language} + "_" + {condition} + "_run" + {run_num} + "_epoch" + {epoch} + "_ur_vowel.png"
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
data_split_ratio = [0.8, 0.1, 0.1]

# Model hyperparameters
encoder_embedding_dim = 11  # original=300
decoder_embedding_dim = 11  # original=300
hidden_dim = 8  # original=256
n_layers = 1
encoder_dropout = 0.1  # 0.0 is equivalent to Identity function
decoder_dropout = 0.1  # 0.0 is equivalent to Identity function
teacher_forcing_ratio = 0.5  # original=0.5

# Training hyperparameters
n_epochs = 50  # original=50. Note: must be >1 !!!
save_epochs = 10
learning_rate = 1e-4
batch_size = 32  # original=32
#clip = 1.0