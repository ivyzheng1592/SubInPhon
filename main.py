import os
import torch

from text_dataset_loader import TextDataset
from audio_dataset_loader import AudioDataset
from text_network import TextSeq2Seq
from audio_network import AudioSeq2Seq
from text_run import TextRun
import hyper_params as hp


# a function that loads text dataset, initializes text model
# and completes multiple runs of training and evaluation of one condition
def text_condition(trial_num, language, datatype, condition, n_reps, device):

    print(" - Loading dataset:")
    annotations_file = os.path.join("Dataset", language + "_" + datatype + "_" + condition + ".csv")
    dataset = TextDataset(annotations_file, hp.special_tokens, device=device)
    print(f"The dataset contains {len(dataset)} UR-SR pairs")

    # vocabulary size
    ur_vocab_size = len(dataset.ur_alphabet)
    sr_vocab_size = len(dataset.sr_alphabet)
    print(f"The UR vocabulary size is {ur_vocab_size}")
    print(f"The SR vocabulary size is {sr_vocab_size}")

    print(" - Splitting dataset:")
    train_data, valid_data, test_data = dataset.split_dataset(hp.data_split_ratio)

    print(" - Creating dataloader:")
    train_dataloader = train_data.dataset.get_dataloader(hp.batch_size)
    valid_dataloader = valid_data.dataset.get_dataloader(hp.batch_size)
    test_dataloader = test_data.dataset.get_dataloader(hp.batch_size)

    print(" - Initializing model:")
    # model hyperparameters
    encoder_input_dim = ur_vocab_size
    decoder_input_dim = sr_vocab_size
    output_dim = sr_vocab_size

    # model initialization
    seq2seq = TextSeq2Seq(encoder_input_dim, decoder_input_dim,
                          hp.encoder_embedding_dim, hp.decoder_embedding_dim,
                          hp.n_layers, hp.hidden_dim, output_dim,
                          hp.encoder_dropout, hp.decoder_dropout, device=device)

    print(" - Training and evaluating model:")
    for rep in n_reps:
        run = TextRun(seq2seq, trial_num, language, datatype, condition, rep)
        #run.train(train_dataloader, valid_dataloader)
        #run.test(test_dataloader)
        run.evaluate_one_batch(test_dataloader, dataset)

"""
# a function that loads audio dataset, initializes audio model
# and completes multiple runs of training and evaluation of one condition
def audio_condition(trial_num, language, datatype, condition, n_reps, device):

    print(" - Loading dataset:")
    annotations_file = os.path.join("Dataset", language + "_" + datatype + "_" + condition + ".csv")
    audio_dir = os.path.join("Dataset", "audio", language)
    dataset = AudioDataset(annotations_file, audio_dir, hp.special_tokens,
                           hp.sample_rate, hp.n_samples, hp.n_fft, hp.hop_length, hp.n_mels,
                           wav2mel=True, power2db=True, device=device)
    print(f"The dataset contains {len(dataset)} UR-SR pairs")

    # input shape
    ur, sr = dataset[0]
    ur_shape = ur.shape  # [n_channels, n_freq, n_samples]
    sr_shape = sr.shape
    print(f"The UR input shape is {ur_shape}")
    print(f"The SR input shape is {sr_shape}")

    print(" - Splitting dataset:")
    train_data, valid_data, test_data = dataset.split_dataset(hp.data_split_ratio)

    print(" - Creating dataloader:")
    train_dataloader = train_data.dataset.get_dataloader(hp.batch_size)
    valid_dataloader = valid_data.dataset.get_dataloader(hp.batch_size)
    test_dataloader = test_data.dataset.get_dataloader(hp.batch_size)

    print(" - Initializing model:")
    # model hyperparameters
    encoder_input_dim = ur_shape[1]  # n_freq
    decoder_input_dim = sr_shape[1]
    output_dim = sr_shape[1]

    # model initialization
    seq2seq = AudioSeq2Seq(encoder_input_dim, decoder_input_dim,
                           hp.encoder_embedding_dim, hp.decoder_embedding_dim,
                           hp.n_layers, hp.hidden_dim, output_dim,
                           hp.encoder_dropout, hp.decoder_dropout, device=device)

    print(" - Training and evaluating model:")
    for run in n_reps:
        audio_run(seq2seq, train_dataloader, valid_dataloader, test_dataloader,
                  trial_num, language, datatype, condition, run)
"""

if __name__ == "__main__":

    # defining the current trial of running
    trial_num = "250401"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    # running each condition for x times
    languages = ["English"]
    #datatypes = ["txt", "aud"]
    conditions = ["harmony", "disharmony"]
    n_reps = range(1)
    for language in languages:
        for condition in conditions:
            text_condition(trial_num, language, "txt", condition, n_reps, device)
            #audio_condition(trial_num, language, "aud", condition, n_reps, device)