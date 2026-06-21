"""
data structure:
When {property} is empty, that segment is omitted from file names.
- audio_root
    - {language_root}  # e.g., EnglishBH
        - {src} + ".mp3"
        - {src} + ".wav"
- "data"
    - {base_lang_name} + "_features.xlsx"
    - {trial_num} + "_" + {lang_name} + "_{property}" + "_generated_data"
        - {lang_name} + "_{property}_" + {directionality} + "_run" + {run_num} + "_harmony.csv"
        - {lang_name} + "_{property}_" + {directionality} + "_run" + {run_num} + "_disharmony.csv"
        - {lang_name} + "_{property}_" + {directionality} + "_run" + {run_num} + "_template_counts.xlsx"
- "results"
    - {trial_num} + "_" + {lang_name} + "_{property}_" + {modality}
        - {lang_name} + "_{property}_" + {modality} + "_acc.csv"
        - {lang_name} + "_{property}_" + {modality} + "_pred.csv"
        - {lang_name} + "_{property}_" + {modality} + "_acc_plots"
            - {lang_name} + "_{property}_" + {modality} + "_" + {directionality} + "_" + {condition} + "_run" + {run_num} + "_acc_plot.png"
        - {lang_name} + "_{property}_" + {modality} + "_model_files"
            - {lang_name} + "_{property}_" + {modality} + "_" + {directionality} + "_" + {condition} + "_run" + {run_num} + "_model_files"
                - {lang_name} + "_{property}_" + {modality} + "_" + {directionality} + "_" + {condition} + "_run" + {run_num} + "_epoch" + {epoch} + "_seq2seq.pth"
        - {lang_name} + "_{property}_" + {modality} + "_att_plots"
            - {lang_name} + "_{property}_" + {modality} + "_" + {directionality} + "_" + {condition} + "_run" + {run_num} + "_att_plots"
                - {lang_name} + "_{property}_" + {modality} + "_" + {directionality} + "_" + {condition} + "_run" + {run_num} + "_epoch" + {epoch} + "_" + {src} + "_" + {trg} + ".png"
                - {lang_name} + "_{property}_" + {modality} + "_" + {directionality} + "_" + {condition} + "_run" + {run_num} + "_epoch" + {epoch} + "_" + {src} + "_" + {trg} + "_txt.png"
                - {lang_name} + "_{property}_" + {modality} + "_" + {directionality} + "_" + {condition} + "_run" + {run_num} + "_epoch" + {epoch} + "_" + {src} + "_" + {trg} + "_aud.png"
        - {lang_name} + "_{property}_" + {modality} + "_embed_plots"
            - {lang_name} + "_{property}_" + {modality} + "_" + {directionality} + "_" + {condition} + "_run" + {run_num} + "_embed.html"
            - {lang_name} + "_{property}_" + {modality} + "_" + {directionality} + "_" + {condition} + "_run" + {run_num} + "_epoch" + {epoch} + "_embed.png"
        - {lang_name} + "_{property}_" + {modality} + "_aud_embed_plots"
            - {lang_name} + "_{property}_" + {modality} + "_" + {directionality} + "_" + {condition} + "_run" + {run_num} + "_aud_embed.csv"
            - {lang_name} + "_{property}_" + {modality} + "_" + {directionality} + "_" + {condition} + "_run" + {run_num} + "_aud_vowel_distance.csv"
            - {lang_name} + "_{property}_" + {modality} + "_" + {directionality} + "_" + {condition} + "_run" + {run_num} + "_pred_embed.png"
            - {lang_name} + "_{property}_" + {modality} + "_" + {directionality} + "_" + {condition} + "_run" + {run_num} + "_aud_embed.html"
            - {lang_name} + "_{property}_" + {modality} + "_" + {directionality} + "_" + {condition} + "_run" + {run_num} + "_aud_vowel_distance.html"
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
n_mels = 128

# Dataset hyperparameters
data_proportion = 1.0  # 1.0 for shortened, 0.1 for full, 0.2 for reduced (nonidentical), 0.0001 for expanded
text_data_split_ratio = [0.8, 0.1, 0.1]
audio_data_split_ratio = [0.8, 0.1, 0.1]

# Model hyperparameters
audio_model = "t1"  # "t1" or "t2"
embedding_dim = 11  # original=300
prenet_dim = 64
text_hidden_dim = 4  # original=256
audio_hidden_dim = 32
text_n_layers = 1
audio_n_layers = 2
aux_n_layers = 1  # auxiliary duration prediction networks
num_heads = 2  # number of multihead attention
cnn_depth = 1
text_dropout = 0.2  # 0.0 is equivalent to Identity function
audio_dropout = 0.2  # 0.0 is equivalent to Identity function
text_teacher_forcing = 0.5  # original=0.5
audio_teacher_forcing = 0.5

# Embedding settings
embedding_init_low = 0.0
embedding_init_high = 1.0  # original=0.01
freeze = False

# Training hyperparameters
n_epochs = 100
save_epochs = 10
learning_rate = 1e-4
batch_size = 32

# Reproducibility
base_seed = 1234  # 1234 for first five runs, 2345 for next five runs

# Audio data root
#audio_root = "/media/ldlmdl/A2AAE4B1AAE482E1/SSD_Documents/subinphon"
audio_root = "/mnt/data/Projects/SubInPhon/dataset"

# Experiment settings
lang_name = "EnglishBH_shortened"
# Property options: "nonidentical," where identical surface vowels are avoided
property = ""
directionality = ["l2r", "r2l"]
conditions = ["harmony", "disharmony"]
# Run mode options: "train and evaluate", "tuning", "inspection"
run_mode = "train and evaluate"
# Prediction logging mode options: "vowel_only_error", "consonant_vowel_error", "all_correct_syll"
pred_log = "vowel_only_error"
device = "cuda"
