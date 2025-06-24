import os
import torch

from Dataset.languages import languages
from text_dataset import TextDataset
from text_network import TextSeq2Seq
from text_run import TextRun
from text_record import TextRecorder
from feature_dataset import FeatureDataset
from feature_network import FeatureSeq2Seq
from audio_dataset_loader import AudioDataset
from audio_network_1 import AudioSeq2Seq
from audio_run import AudioRun
import hyper_params as hp


# a function that loads text dataset, initializes text model for each run of each condition
def text(trial_num, lang_name, conditions, runs, check_epoch, device):

    os.makedirs(os.path.join("Results", trial_num + "_" + lang_name), exist_ok=True)

    for condition in conditions:
        print(" - Instantiating language pattern:")
        language = languages[lang_name]

        print(" - Loading dataset:")
        annotations_file = os.path.join("Dataset", lang_name + "_" + condition + ".csv")
        dataset = TextDataset(annotations_file, hp.special_tokens, device=device)

        for run_num in runs:

            print(" - Splitting dataset:")
            train_data, valid_data, test_data = dataset.split_dataset(hp.data_split_ratio)

            print(" - Creating dataloader:")
            train_dataloader = dataset.get_dataloader(train_data, hp.batch_size)
            valid_dataloader = dataset.get_dataloader(valid_data, hp.batch_size)
            test_dataloader = dataset.get_dataloader(test_data, hp.batch_size)

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

            print(" - Preparing data recorder:")
            recorder = TextRecorder(dataset, trial_num, language, condition, run_num)

            print(" - Training and evaluating model:")
            rep = TextRun(seq2seq, recorder)
            rep.train(train_dataloader, valid_dataloader)
            rep.test(test_dataloader)
            rep.evaluate_attention(test_dataloader, check_epoch)
            rep.evaluate_embedding(check_epoch)


# a function that loads text dataset, initializes text model for each run of each condition
def feature(trial_num, lang_name, conditions, runs, check_epoch, freeze, device):

    os.makedirs(os.path.join("Results", trial_num + "_" + lang_name), exist_ok=True)

    for condition in conditions:
        print(" - Instantiating language pattern:")
        language = languages[lang_name]

        print(" - Loading dataset:")
        annotations_file = os.path.join("Dataset", lang_name + "_" + condition + ".csv")
        feature_file = os.path.join("Dataset", lang_name.split("_")[0] + "_features.xlsx")
        dataset = FeatureDataset(annotations_file, feature_file, hp.special_tokens, device=device)

        for run_num in runs:
            print(" - Splitting dataset:")
            train_data, valid_data, test_data = dataset.split_dataset(hp.data_split_ratio)

            print(" - Creating dataloader:")
            train_dataloader = dataset.get_dataloader(train_data, hp.batch_size)
            valid_dataloader = dataset.get_dataloader(valid_data, hp.batch_size)
            test_dataloader = dataset.get_dataloader(test_data, hp.batch_size)

            print(" - Initializing model:")
            # model hyperparameters
            encoder_input_dim = len(dataset.ur_alphabet)
            decoder_input_dim = len(dataset.sr_alphabet)
            output_dim = len(dataset.sr_alphabet)
            encoder_embedding_weight = dataset.ur_embedding
            decoder_embedding_weight = dataset.sr_embedding

            # model initialization
            seq2seq = FeatureSeq2Seq(encoder_input_dim, decoder_input_dim,
                                     hp.encoder_embedding_dim, hp.decoder_embedding_dim,
                                     encoder_embedding_weight, decoder_embedding_weight,
                                     hp.n_layers, hp.hidden_dim, output_dim,
                                     hp.encoder_dropout, hp.decoder_dropout,
                                     freeze=freeze, device=device)

            print(" - Preparing data recorder:")
            recorder = TextRecorder(dataset, trial_num, language, condition, run_num)

            print(" - Training and evaluating model:")
            rep = TextRun(seq2seq, recorder)
            rep.train(train_dataloader, valid_dataloader)
            rep.test(test_dataloader)
            rep.evaluate_attention(test_dataloader, check_epoch)
            rep.evaluate_embedding(check_epoch)


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

    trial_num = "2506232020_freeze_new"  # time stamp
    lang_name = "EnglishBH_fea"
    conditions = ["harmony", "disharmony"]
    runs = range(3)
    feature(trial_num, lang_name, conditions, runs, check_epoch=hp.n_epochs - 1, freeze=True, device=device)

    trial_num = "2506232020_unfreeze_new"  # time stamp
    lang_name = "EnglishBH_fea"
    conditions = ["harmony", "disharmony"]
    runs = range(3)
    feature(trial_num, lang_name, conditions, runs, check_epoch=hp.n_epochs - 1, freeze=False, device=device)

    trial_num = "2506232020_new"  # time stamp
    lang_name = "EnglishBH_txt"
    conditions = ["harmony", "disharmony"]
    runs = range(3)
    text(trial_num, lang_name, conditions, runs, check_epoch=hp.n_epochs - 1, device=device)

