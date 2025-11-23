import os
import torch
import torch.nn as nn

from Dataset.languages import languages
from text_dataset import TextDataset
from text_network import TextSeq2Seq
from text_run import TextRun
from text_record import TextRecorder
from feature_dataset import FeatureDataset
from feature_network import FeatureSeq2Seq
from audio_dataset import AudioDataset
from audio_network_t1 import AudioSeq2Seq
from audio_run import AudioRun
from audio_record import AudioRecorder
import hyper_params as hp


# a function that loads text dataset, initializes text model for each run of each condition
def text(trial_num, lang_name, conditions, runs, run_mode, device):

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
            seq2seq = TextSeq2Seq(encoder_input_dim, decoder_input_dim, output_dim, device=device)

            # embedding weight initialization
            for name, param in seq2seq.named_parameters():
                if "embedding.weight" in name:
                    nn.init.uniform_(param.data, a=0, b=0.01)

            print(" - Preparing data recorder:")
            recorder = TextRecorder(dataset, trial_num, language, condition, run_num)

            print(" - Training and evaluating model:")
            rep = TextRun(seq2seq, recorder)
            if run_mode == "train and evaluate":
                rep.train(train_dataloader, valid_dataloader)
                rep.test(test_dataloader)
                rep.evaluate_attention(test_dataloader)
                rep.evaluate_embedding()
            elif run_mode == "evaluate attention":
                rep.evaluate_attention(test_dataloader)
            else: # evaluate embedding only
                rep.evaluate_embedding()


# a function that loads text dataset, initializes text model for each run of each condition
def feature(trial_num, lang_name, conditions, runs, run_mode, freeze, device):

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
            seq2seq = FeatureSeq2Seq(encoder_input_dim, decoder_input_dim, output_dim,
                                     encoder_embedding_weight, decoder_embedding_weight,
                                     freeze=freeze, device=device)

            print(" - Preparing data recorder:")
            recorder = TextRecorder(dataset, trial_num, language, condition, run_num)

            print(" - Training and evaluating model:")
            rep = TextRun(seq2seq, recorder)
            if run_mode == "train and evaluation":
                rep.train(train_dataloader, valid_dataloader)
                rep.test(test_dataloader)
                rep.evaluate_attention(test_dataloader)
                rep.evaluate_embedding()
            elif run_mode == "evaluate attention":
                rep.evaluate_attention(test_dataloader)
            else: # evaluate embedding only
                rep.evaluate_embedding()


# a function that loads audio dataset, initializes audio model for each run of each condition
def audio(trial_num, lang_name, conditions, runs, run_mode, device):

    os.makedirs(os.path.join("Results", trial_num + "_" + lang_name), exist_ok=True)

    for condition in conditions:
        print(" - Instantiating language pattern:")
        language = languages[lang_name]

        print(" - Loading dataset:")
        annotations_file = os.path.join("Dataset", lang_name + "_" + condition + ".csv")
        audio_dir = os.path.join("/media/ldlmdl/A2AAE4B1AAE482E1/SSD_Documents/subinphon",
                                 lang_name.split("_")[0])
        dataset = AudioDataset(annotations_file, audio_dir, hp.special_tokens,
                               wav2mel=True, power2db=True, device=device)

        for run_num in runs:

            print(" - Splitting dataset:")
            train_data, valid_data, test_data = dataset.split_dataset(hp.data_split_ratio)

            print(" - Creating dataloader:")
            train_dataloader = dataset.get_dataloader(train_data, hp.batch_size)
            valid_dataloader = dataset.get_dataloader(valid_data, hp.batch_size)
            test_dataloader = dataset.get_dataloader(test_data, hp.batch_size)

            print(" - Initializing model:")
            # model hyperparameters
            encoder_input_dim = hp.n_mels
            decoder_input_dim = len(dataset.sr_alphabet)
            synthsizer_input_dim = hp.n_mels
            text_output_dim = len(dataset.sr_alphabet)
            audio_output_dim = hp.n_mels

            # model initialization
            seq2seq = AudioSeq2Seq(encoder_input_dim, decoder_input_dim, synthsizer_input_dim,
                                   text_output_dim, audio_output_dim, device=device)

            print(" - Preparing data recorder:")
            recorder = AudioRecorder(dataset, trial_num, language, condition, run_num)

            print(" - Training and evaluating model:")
            rep = AudioRun(seq2seq, recorder)
            if run_mode == "train and evaluate":
                rep.train(train_dataloader, valid_dataloader)
                rep.test(test_dataloader)
                rep.evaluate_attention(test_dataloader)
                rep.evaluate_embedding()
            elif run_mode == "evaluate attention":
                rep.evaluate_attention(test_dataloader)
            else: # evaluate embedding only
                rep.evaluate_embedding()


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    trial_num = "2511231600"  # time stamp
    lang_name = "EnglishBH_nonidentical_aud"
    conditions = ["harmony", "disharmony"]
    runs = range(1)
    audio(trial_num, lang_name, conditions, runs, run_mode="train and evaluate", device=device)