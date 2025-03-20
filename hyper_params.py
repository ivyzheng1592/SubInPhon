# Alphabet
sos_token = "<SOS>"
eos_token = "<EOS>"
unk_token = "<UNK>"
pad_token = "<PAD>"
special_tokens = [sos_token, eos_token, unk_token, pad_token]

# Audio
sample_rate = 22050
n_samples = 22050

# Model hyperparameters
encoder_embedding_dim = 30  # original=300
decoder_embedding_dim = 30  # original=300
hidden_dim = 32  # original=256
n_layers = 1
encoder_dropout = 0.1  # 0.0 is equivalent to Identity function
decoder_dropout = 0.1  # 0.0 is equivalent to Identity function
teacher_forcing_ratio = 0.5  #original=0.5

# Training hyperparameters
n_epochs = 10  # original=50. Note: must be >1 !!!
learning_rate = 3e-4
batch_size = 32  # original=32
clip = 1.0