"""
data structure:
- "Dataset"
    - {language} + "_" + {condition} + ".csv"
    - "audio"
        - {language}
            - {src} + ".mp3"
            - {src} + ".wav"
- "Results"
    - {trial_num} + "_" + {language}
        - {language} + "_" + {condition} + "_acc.csv"
        - {language} + "_" + {condition} + "_pred.csv"
        - {language} + "_" + {condition} + "_model_files"
            - {language} + "_" + {condition} + "_run" + {run_num} + "_model_files"
                - {language} + "_" + {condition} + "_run" + {run_num} + "_epoch" + {epoch} + "_seq2seq.pth"
        - {language} + "_" + {condition} + "_acc_plots"
            - {language} + "_" + {condition} + "_run" + {run_num} + "_acc_plot.png"
        - {language} + "_" + {condition} + "_att_plots"
            - {language} + "_" + {condition} + "_run" + {run_num} + "_att_plots"
                - {language} + "_" + {condition} + "_run" + {run_num} + "att_type.csv"
                - {language} + "_" + {condition} + "_run" + {run_num} + {src}_{trg} + ".png"
        - {language} + "_" + {condition} + "_embed_plots"
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
data_split_ratio = [0.6, 0.2, 0.2]

# Model hyperparameters
encoder_embedding_dim = 10  # original=300
decoder_embedding_dim = 10  # original=300
hidden_dim = 8  # original=256
n_layers = 1
encoder_dropout = 0.1  # 0.0 is equivalent to Identity function
decoder_dropout = 0.1  # 0.0 is equivalent to Identity function
teacher_forcing_ratio = 0.5  # original=0.5

# Training hyperparameters
n_epochs = 30  # original=50. Note: must be >1 !!!
save_epochs = 5
learning_rate = 5e-4
batch_size = 32  # original=32
#clip = 1.0