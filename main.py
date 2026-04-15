import os
import random
import torch
import torch.nn as nn
import numpy as np

from Dataset.language_registry import languages
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


def _set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# a function that loads text dataset, initializes text model for each run of each condition
def text(trial_num, runs, resume_model_file=None):

    os.makedirs(os.path.join("Results", trial_num + "_" + hp.lang_name + "_txt"),
                exist_ok=True)

    for condition in hp.conditions:
        print(" - Instantiating language pattern:")
        language = languages[hp.lang_name]

        print(" - Loading dataset:")
        annotations_file = os.path.join("Dataset", hp.lang_name + "_" + condition + ".csv")
        dataset = TextDataset(annotations_file, hp.special_tokens, device=hp.device)

        for run_num in runs:
            _set_seed(hp.base_seed + run_num)
            print(" - Splitting dataset:")
            train_data, valid_data, test_data = dataset.split_dataset(hp.text_data_split_ratio)

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
            seq2seq = TextSeq2Seq(encoder_input_dim, decoder_input_dim, output_dim, device=hp.device)

            # embedding weight initialization
            for name, param in seq2seq.named_parameters():
                if "embedding.weight" in name:
                    nn.init.uniform_(param.data, a=hp.embedding_init_low, b=hp.embedding_init_high)

            print(" - Preparing data recorder:")
            recorder = TextRecorder(dataset, trial_num, language, "txt", condition, run_num)

            print(" - Training and evaluating model:")
            rep = TextRun(seq2seq, recorder, resume_model_file=resume_model_file)
            if hp.run_mode == "train and evaluate":
                rep.run(train_dataloader, test_dataloader, "test")
                rep.evaluate_attention(test_dataloader)
                rep.evaluate_embedding()
            elif hp.run_mode == "tuning":
                rep.run(train_dataloader, valid_dataloader, "valid")
                rep.evaluate_attention(valid_dataloader)
                rep.evaluate_embedding()
            else: # evaluate only
                rep.evaluate_attention(test_dataloader)
                rep.evaluate_embedding()


# a function that loads text dataset, initializes text model for each run of each condition
def feature(trial_num, runs, resume_model_file=None):

    os.makedirs(os.path.join("Results", trial_num + "_" + hp.lang_name + "_fea"),
                exist_ok=True)

    for condition in hp.conditions:
        print(" - Instantiating language pattern:")
        language = languages[hp.lang_name]

        print(" - Loading dataset:")
        annotations_file = os.path.join("Dataset", hp.lang_name + "_" + condition + ".csv")
        feature_file = os.path.join("Dataset", hp.lang_name.split("_")[0] + "_features.xlsx")
        dataset = FeatureDataset(annotations_file, feature_file, hp.special_tokens, device=hp.device)

        for run_num in runs:
            _set_seed(hp.base_seed + run_num)
            print(" - Splitting dataset:")
            train_data, valid_data, test_data = dataset.split_dataset(hp.text_data_split_ratio)

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
                                     freeze=hp.freeze, device=hp.device)

            print(" - Preparing data recorder:")
            recorder = TextRecorder(dataset, trial_num, language, "fea", condition, run_num)

            print(" - Training and evaluating model:")
            rep = TextRun(seq2seq, recorder, resume_model_file=resume_model_file)
            if hp.run_mode == "train and evaluate":
                rep.run(train_dataloader, test_dataloader, "test")
                rep.evaluate_attention(test_dataloader)
                rep.evaluate_embedding()
            elif hp.run_mode == "tuning":
                rep.run(train_dataloader, valid_dataloader, "valid")
                rep.evaluate_attention(valid_dataloader)
                rep.evaluate_embedding()
            else: # evaluate only
                rep.evaluate_attention(test_dataloader)
                rep.evaluate_embedding()


# a function that loads audio dataset, initializes audio model for each run of each condition
def audio(trial_num, runs, resume_model_file=None):

    os.makedirs(os.path.join("Results", trial_num + "_" + hp.lang_name + "_aud"),
                exist_ok=True)

    for condition in hp.conditions:
        print(" - Instantiating language pattern:")
        language = languages[hp.lang_name]

        print(" - Loading dataset:")
        annotations_file = os.path.join("Dataset", hp.lang_name + "_" + condition + ".csv")
        audio_dir = os.path.join(hp.audio_root, hp.lang_name.split("_")[0])
        dataset = AudioDataset(annotations_file, audio_dir, hp.special_tokens,
                               wav2mel=True, power2db=True, device=hp.device)

        for run_num in runs:
            _set_seed(hp.base_seed + run_num)
            print(" - Splitting dataset:")
            train_data, valid_data, test_data = dataset.split_dataset(hp.audio_data_split_ratio)

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
                                   text_output_dim, audio_output_dim, device=hp.device)

            print(" - Preparing data recorder:")
            recorder = AudioRecorder(dataset, trial_num, language, "aud", condition, run_num)

            print(" - Training and evaluating model:")
            rep = AudioRun(seq2seq, recorder, resume_model_file=resume_model_file)
            if hp.run_mode == "train and evaluate":
                rep.run(train_dataloader, test_dataloader, "test")
                rep.evaluate_attention(test_dataloader)
                rep.evaluate_embedding()
            elif hp.run_mode == "tuning":
                rep.run(train_dataloader, valid_dataloader, "valid")
                rep.evaluate_attention(valid_dataloader)
                rep.evaluate_embedding()
            else: # evaluate only
                rep.evaluate_attention(test_dataloader)
                rep.evaluate_embedding()


if __name__ == "__main__":

    trial_num = "2512072030_partial_consonant"  # time stamp
    runs = range(2)
    audio(trial_num, runs)
