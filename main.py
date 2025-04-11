import os
import torch

from text_dataset_loader import TextDataset
from audio_dataset_loader import AudioDataset
from text_network import TextSeq2Seq
from audio_network_1 import AudioSeq2Seq
from text_run import TextRun
from audio_run import AudioRun
import hyper_params as hp


# a function that loads text dataset, initializes text model
# and completes multiple runs of training and evaluation of one condition
def text_condition(trial_num, datatype, language, condition, n_run, n_check, device):

    print(" - Loading dataset:")
    annotations_file = os.path.join("Dataset", language + "_" + datatype + "_" + condition + ".csv")
    dataset = TextDataset(annotations_file, hp.special_tokens, device=device)

    print(" - Splitting dataset:")
    train_data, valid_data, test_data = dataset.split_dataset(hp.data_split_ratio)

    print(" - Creating dataloader:")
    train_dataloader = train_data.dataset.get_dataloader(hp.batch_size)
    valid_dataloader = valid_data.dataset.get_dataloader(hp.batch_size)
    test_dataloader = test_data.dataset.get_dataloader(hp.batch_size)

    print(" - Initializing model:")
    # model hyperparameters
    encoder_input_dim = len(dataset.ur_alphabet)
    decoder_input_dim = len(dataset.sr_alphabet)
    output_dim = len(dataset.sr_alphabet)

    # model initialization
    seq2seq = TextSeq2Seq(encoder_input_dim, decoder_input_dim,
                          hp.encoder_embedding_dim, hp.decoder_embedding_dim,
                          hp.n_layers, hp.hidden_dim, output_dim,
                          hp.encoder_dropout, hp.decoder_dropout, device=device)

    print(" - Training and evaluating model:")
    for run in n_run:
        rep = TextRun(seq2seq, dataset, trial_num, datatype, language, condition, run)
        rep.train(train_dataloader, valid_dataloader)
        rep.test(test_dataloader)

    print(" - Inspecting model outputs:")
    for check in n_check:
        rep = TextRun(seq2seq, dataset, trial_num, datatype, language, condition, check)
        rep.evaluate_one_batch(test_dataloader)

"""
# a function that loads audio dataset, initializes audio model
# and completes multiple runs of training and evaluation of one condition
def audio_condition(trial_num, datatype, language, condition, n_run, n_check, device):

    print(" - Loading dataset:")
    annotations_file = os.path.join("Dataset", language + "_" + condition + ".csv")
    audio_dir = os.path.join("Dataset", "audio", language)
    dataset = AudioDataset(annotations_file, audio_dir, hp.special_tokens,
                           hp.sample_rate, hp.n_samples, hp.n_fft, hp.hop_length, hp.n_mels,
                           wav2mel=True, power2db=True, device=device)

    print(" - Splitting dataset:")
    train_data, valid_data, test_data = dataset.split_dataset(hp.data_split_ratio)

    print(" - Creating dataloader:")
    train_dataloader = train_data.dataset.get_dataloader(hp.batch_size)
    valid_dataloader = valid_data.dataset.get_dataloader(hp.batch_size)
    test_dataloader = test_data.dataset.get_dataloader(hp.batch_size)

    print(" - Initializing model:")
    # model hyperparameters
    encoder_input_dim = hp.n_mels
    decoder_input_dim = hp.n_mels
    output_dim = hp.n_mels

    # model initialization
    seq2seq = AudioSeq2Seq(encoder_input_dim, decoder_input_dim,
                           hp.encoder_embedding_dim, hp.decoder_embedding_dim,
                           hp.n_layers, hp.hidden_dim, output_dim,
                           hp.encoder_dropout, hp.decoder_dropout, device=device)

    print(" - Training and evaluating model:")
    for run in n_run:
        rep = AudioRun(seq2seq, trial_num, datatype, language, condition, run)
        rep.train(train_dataloader, valid_dataloader)
        rep.test(test_dataloader)

    #print(" - Inspecting model outputs:")
    #for check in n_check:
        #reps[check].evaluate_one_batch(test_dataloader, dataset)
"""

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    # defining the current trial of running
    trial_num = "250411"
    datatype = "txt"
    if not os.path.exists(os.path.join("Results", trial_num + "_" + datatype)):
        os.mkdir(os.path.join("Results", trial_num + "_" + datatype))

    # running each condition for x times
    languages = ["EnglishBH"]
    conditions = ["harmony", "disharmony"]
    n_run = range(10)  # which run to complete
    n_check = range(0)  # which run to inspect
    for language in languages:
        for condition in conditions:
            text_condition(trial_num, "txt", language, condition, n_run, n_check, device)
            #audio_condition(trial_num, language, "aud", condition, n_run, n_check, device)