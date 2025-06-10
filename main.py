import os
import torch

from Dataset.languages import languages
from hyper_params import encoder_embedding_dim
from text_dataset import TextDataset
from text_network import TextSeq2Seq
from text_run import TextRun
from text_record import TextRecorder
from feature_dataset import FeatureDataset
from feature_network import FeatureSeq2Seq
#from feature_run import FeatureRun
from audio_dataset_loader import AudioDataset
from audio_network_1 import AudioSeq2Seq
from audio_run import AudioRun
import hyper_params as hp


# a function that loads text dataset, initializes text model
# and completes multiple runs of training and evaluation of one condition
def text_condition(trial_num, lang_name, condition, n_run, n_check,
                   check_epoch, check_type, device):

    print(" - Instantiating language pattern:")
    language = languages[lang_name]

    print(" - Loading dataset:")
    annotations_file = os.path.join("Dataset", lang_name + "_" + condition + ".csv")
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

    print(" - Preparing data recorder:")
    recorder = TextRecorder(dataset, trial_num, language, condition)

    print(" - Training and evaluating model:")
    for run in n_run:
        rep = TextRun(seq2seq, recorder, trial_num, lang_name, condition, run)
        rep.train(train_dataloader, valid_dataloader)
        rep.test(test_dataloader)

    print(" - Inspecting model outputs:")
    for check in n_check:
        rep = TextRun(seq2seq, recorder, trial_num, lang_name, condition, check)
        rep.evaluate_one_batch(test_dataloader, check_epoch, check_type)

"""
# a function that loads feature dataset, initializes feature model
# and completes multiple runs of training and evaluation of one condition
def feature_condition(trial_num, lang_name, condition, n_run, n_check,
                      check_epoch, check_type, device):

    print(" - Instantiating language pattern:")
    language = languages[lang_name]

    print(" - Loading dataset:")
    annotations_file = os.path.join("Dataset", lang_name + "_" + condition + ".csv")
    feature_file = os.path.join("Dataset", lang_name + "_features.xlsx")
    dataset = FeatureDataset(annotations_file, feature_file, hp.special_tokens, device=device)

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
    encoder_embedding_weight = dataset.ur_embedding
    decoder_embedding_weight = dataset.sr_embedding

    # model initialization
    seq2seq = FeatureSeq2Seq(encoder_input_dim, decoder_input_dim,
                             hp.encoder_embedding_dim, hp.decoder_embedding_dim,
                             encoder_embedding_weight, decoder_embedding_weight,
                             hp.n_layers, hp.hidden_dim, output_dim,
                             hp.encoder_dropout, hp.decoder_dropout, device=device)

    print(" - Preparing data recorder:")
    recorder = TextRecorder(dataset, trial_num, language, condition)

    print(" - Training and evaluating model:")
    for run in n_run:
        rep = FeatureRun(seq2seq, recorder, trial_num, language, condition, run)
        rep.train(train_dataloader, valid_dataloader)
        rep.test(test_dataloader)

    print(" - Inspecting model outputs:")
    for check in n_check:
        rep = FeatureRun(seq2seq, recorder, trial_num, language, condition, check)
        rep.evaluate_one_batch(test_dataloader, check_epoch, check_type)
"""

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

    # hyperparameters of the current trial of running
    trial_num = "250414"  # time stamp
    lang_name = "EnglishBH_txt"
    datatype = lang_name.split("_")[1]  # txt for text, fea for feature, aud for audio
    conditions = ["harmony", "disharmony"]  # condition depends on the language
    n_run = range(0)  # which run to complete
    n_check = range(2)  # which run to inspect
    os.makedirs(os.path.join("Results", trial_num + "_" + lang_name), exist_ok=True)

    # run each condition for x times
    for condition in conditions:
        if datatype == "txt":
            text_condition(trial_num, lang_name, condition, n_run, n_check,
                           check_epoch=hp.n_epochs-1, check_type="both", device=device)
        elif datatype == "fea":
            pass
            #feature_condition(trial_num, lang_name, condition, n_run, n_check,
                              #check_epoch=hp.n_epochs-1, check_type="both", device=device)
        elif datatype == "aud":
            #audio_condition(trial_num, language, "aud", condition, n_run, n_check, device)
            pass