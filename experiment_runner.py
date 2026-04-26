import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split

from Dataset.language_registry import languages
from audio_dataset import AudioDataset
from audio_network_t1 import AudioSeq2Seq
from audio_recorder import AudioRecorder
from audio_trainer import AudioTrainer
from feature_dataset import FeatureDataset
from feature_network import FeatureSeq2Seq
import hyper_params as hp
from text_dataset import TextDataset
from text_network import TextSeq2Seq
from text_recorder import TextRecorder
from text_trainer import TextTrainer


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_annotations_file(
    language,
    trial_num: str,
    run_num: int,
    directionality: str,
    condition: str,
    gen_eval: bool = False,
) -> str:
    output_dir = os.path.join(
        "Results",
        trial_num + "_" + hp.lang_name + "_generated_data",
        "run_" + str(run_num),
        "expanded" if gen_eval else "full",
    )
    os.makedirs(output_dir, exist_ok=True)
    variant = "aud_vowel" if "aud_vowel" in language.variants else None
    sample_proportion = hp.gen_data_proportion if gen_eval else hp.data_proportion
    return language.generate_stimuli(
        seed=hp.base_seed + run_num,
        sample_proportion=sample_proportion,
        property=hp.property,
        directionality=directionality,
        variant=variant,
        output_dir=output_dir,
    )[condition]


def text(trial_num: str, runs: range, resume_model_file: Optional[str] = None) -> None:
    os.makedirs(os.path.join("Results", trial_num + "_" + hp.lang_name + "_txt"), exist_ok=True)

    for directionality in hp.directionality:
        for condition in hp.conditions:
            print(" - Instantiating language pattern:")
            language = languages[hp.lang_name]

            for run_num in runs:
                set_seed(hp.base_seed + run_num)
                print(" - Loading dataset:")
                annotations_file = prepare_annotations_file(
                    language,
                    trial_num,
                    run_num,
                    directionality,
                    condition,
                )
                dataset = TextDataset(annotations_file, hp.special_tokens, device=hp.device)
                print(" - Splitting dataset:")
                train_data, valid_data, test_data = random_split(dataset, hp.text_data_split_ratio)
                gen_test_dataloader = None
                if hp.gen_eval:
                    gen_lang_name = hp.lang_name if hp.lang_name.endswith("_expanded") else hp.lang_name + "_expanded"
                    gen_language = languages[gen_lang_name]
                    gen_annotations_file = prepare_annotations_file(
                        gen_language,
                        trial_num,
                        run_num,
                        directionality,
                        condition,
                        gen_eval=True,
                    )
                    gen_dataset = TextDataset(gen_annotations_file, hp.special_tokens, device=hp.device)
                    gen_dataset.ur_alphabet = dataset.ur_alphabet
                    gen_dataset.sr_alphabet = dataset.sr_alphabet

                print(" - Creating dataloader:")
                train_dataloader = dataset.get_dataloader(train_data, hp.batch_size)
                valid_dataloader = dataset.get_dataloader(valid_data, hp.batch_size)
                test_dataloader = dataset.get_dataloader(test_data, hp.batch_size)
                if hp.gen_eval:
                    gen_test_dataloader = gen_dataset.get_dataloader(gen_dataset, hp.batch_size, shuffle=False)

                print(" - Initializing model:")
                encoder_input_dim = len(dataset.ur_alphabet)
                decoder_input_dim = len(dataset.sr_alphabet)
                output_dim = len(dataset.sr_alphabet)
                seq2seq = TextSeq2Seq(encoder_input_dim, decoder_input_dim, output_dim, device=hp.device)

                for name, param in seq2seq.named_parameters():
                    if "embedding.weight" in name:
                        nn.init.uniform_(param.data, a=hp.embedding_init_low, b=hp.embedding_init_high)

                print(" - Preparing data recorder:")
                recorder = TextRecorder(dataset, trial_num, language, "txt", directionality, condition, run_num)

                print(" - Training and evaluating model:")
                rep = TextTrainer(seq2seq, recorder, resume_model_file=resume_model_file)
                if hp.run_mode == "train and evaluate":
                    rep.run(train_dataloader, test_dataloader, "test", gen_eval_dataloader=gen_test_dataloader)
                    rep.evaluate_attention(test_dataloader)
                    if hp.gen_eval:
                        rep.evaluate_attention(gen_test_dataloader, gen_eval=True)
                    rep.evaluate_embedding()
                elif hp.run_mode == "tuning":
                    rep.run(train_dataloader, valid_dataloader, "valid")
                    rep.evaluate_attention(valid_dataloader)
                    rep.evaluate_embedding()
                else:
                    rep.evaluate_attention(test_dataloader)
                    if hp.gen_eval:
                        rep.evaluate_attention(gen_test_dataloader, gen_eval=True)
                    rep.evaluate_embedding()


def feature(trial_num: str, runs: range, resume_model_file: Optional[str] = None) -> None:
    os.makedirs(os.path.join("Results", trial_num + "_" + hp.lang_name + "_fea"), exist_ok=True)

    for directionality in hp.directionality:
        for condition in hp.conditions:
            print(" - Instantiating language pattern:")
            language = languages[hp.lang_name]
            feature_file = os.path.join("Dataset", "EnglishBH_features.xlsx")

            for run_num in runs:
                set_seed(hp.base_seed + run_num)
                print(" - Loading dataset:")
                annotations_file = prepare_annotations_file(
                    language,
                    trial_num,
                    run_num,
                    directionality,
                    condition,
                )
                dataset = FeatureDataset(annotations_file, feature_file, hp.special_tokens, device=hp.device)
                print(" - Splitting dataset:")
                train_data, valid_data, test_data = random_split(dataset, hp.text_data_split_ratio)
                gen_test_dataloader = None
                if hp.gen_eval:
                    gen_lang_name = hp.lang_name if hp.lang_name.endswith("_expanded") else hp.lang_name + "_expanded"
                    gen_language = languages[gen_lang_name]
                    gen_annotations_file = prepare_annotations_file(
                        gen_language,
                        trial_num,
                        run_num,
                        directionality,
                        condition,
                        gen_eval=True,
                    )
                    gen_dataset = FeatureDataset(gen_annotations_file, feature_file, hp.special_tokens, device=hp.device)
                    gen_dataset.ur_alphabet = dataset.ur_alphabet
                    gen_dataset.sr_alphabet = dataset.sr_alphabet

                print(" - Creating dataloader:")
                train_dataloader = dataset.get_dataloader(train_data, hp.batch_size)
                valid_dataloader = dataset.get_dataloader(valid_data, hp.batch_size)
                test_dataloader = dataset.get_dataloader(test_data, hp.batch_size)
                if hp.gen_eval:
                    gen_test_dataloader = gen_dataset.get_dataloader(gen_dataset, hp.batch_size, shuffle=False)

                print(" - Initializing model:")
                encoder_input_dim = len(dataset.ur_alphabet)
                decoder_input_dim = len(dataset.sr_alphabet)
                output_dim = len(dataset.sr_alphabet)
                encoder_embedding_weight = dataset.ur_embedding
                decoder_embedding_weight = dataset.sr_embedding

                seq2seq = FeatureSeq2Seq(
                    encoder_input_dim,
                    decoder_input_dim,
                    output_dim,
                    encoder_embedding_weight,
                    decoder_embedding_weight,
                    freeze=hp.freeze,
                    device=hp.device,
                )

                print(" - Preparing data recorder:")
                recorder = TextRecorder(dataset, trial_num, language, "fea", directionality, condition, run_num)

                print(" - Training and evaluating model:")
                rep = TextTrainer(seq2seq, recorder, resume_model_file=resume_model_file)
                if hp.run_mode == "train and evaluate":
                    rep.run(train_dataloader, test_dataloader, "test", gen_eval_dataloader=gen_test_dataloader)
                    rep.evaluate_attention(test_dataloader)
                    if hp.gen_eval:
                        rep.evaluate_attention(gen_test_dataloader, gen_eval=True)
                    rep.evaluate_embedding()
                elif hp.run_mode == "tuning":
                    rep.run(train_dataloader, valid_dataloader, "valid")
                    rep.evaluate_attention(valid_dataloader)
                    rep.evaluate_embedding()
                else:
                    rep.evaluate_attention(test_dataloader)
                    if hp.gen_eval:
                        rep.evaluate_attention(gen_test_dataloader, gen_eval=True)
                    rep.evaluate_embedding()


def audio(trial_num: str, runs: range, resume_model_file: Optional[str] = None) -> None:
    os.makedirs(os.path.join("Results", trial_num + "_" + hp.lang_name + "_aud"), exist_ok=True)

    for directionality in hp.directionality:
        for condition in hp.conditions:
            print(" - Instantiating language pattern:")
            language = languages[hp.lang_name]
            audio_dir = os.path.join(hp.audio_root, hp.lang_name)

            for run_num in runs:
                set_seed(hp.base_seed + run_num)
                print(" - Loading dataset:")
                annotations_file = prepare_annotations_file(
                    language,
                    trial_num,
                    run_num,
                    directionality,
                    condition,
                )
                dataset = AudioDataset(
                    annotations_file,
                    audio_dir,
                    hp.special_tokens,
                    wav2mel=True,
                    power2db=True,
                    device=hp.device,
                )
                print(" - Splitting dataset:")
                train_data, valid_data, test_data = random_split(dataset, hp.audio_data_split_ratio)
                gen_test_dataloader = None
                if hp.gen_eval:
                    gen_lang_name = hp.lang_name if hp.lang_name.endswith("_expanded") else hp.lang_name + "_expanded"
                    gen_language = languages[gen_lang_name]
                    gen_annotations_file = prepare_annotations_file(
                        gen_language,
                        trial_num,
                        run_num,
                        directionality,
                        condition,
                        gen_eval=True,
                    )
                    gen_audio_dir = os.path.join(hp.audio_root, gen_lang_name)
                    gen_dataset = AudioDataset(
                        gen_annotations_file,
                        gen_audio_dir,
                        hp.special_tokens,
                        wav2mel=True,
                        power2db=True,
                        device=hp.device,
                    )
                    gen_dataset.ur_alphabet = dataset.ur_alphabet
                    gen_dataset.sr_alphabet = dataset.sr_alphabet

                print(" - Creating dataloader:")
                train_dataloader = dataset.get_dataloader(train_data, hp.batch_size)
                valid_dataloader = dataset.get_dataloader(valid_data, hp.batch_size)
                test_dataloader = dataset.get_dataloader(test_data, hp.batch_size)
                if hp.gen_eval:
                    gen_test_dataloader = gen_dataset.get_dataloader(gen_dataset, hp.batch_size, shuffle=False)

                print(" - Initializing model:")
                encoder_input_dim = hp.n_mels
                decoder_input_dim = len(dataset.sr_alphabet)
                synthsizer_input_dim = hp.n_mels
                text_output_dim = len(dataset.sr_alphabet)
                audio_output_dim = hp.n_mels

                seq2seq = AudioSeq2Seq(
                    encoder_input_dim,
                    decoder_input_dim,
                    synthsizer_input_dim,
                    text_output_dim,
                    audio_output_dim,
                    device=hp.device,
                )

                print(" - Preparing data recorder:")
                recorder = AudioRecorder(dataset, trial_num, language, "aud", directionality, condition, run_num)

                print(" - Training and evaluating model:")
                rep = AudioTrainer(seq2seq, recorder, resume_model_file=resume_model_file)
                if hp.run_mode == "train and evaluate":
                    rep.run(train_dataloader, test_dataloader, "test", gen_eval_dataloader=gen_test_dataloader)
                    rep.evaluate_attention(test_dataloader)
                    if hp.gen_eval:
                        rep.evaluate_attention(gen_test_dataloader, gen_eval=True)
                    rep.evaluate_embedding()
                    rep.evaluate_audio_embedding(test_dataloader, "test")
                elif hp.run_mode == "tuning":
                    rep.run(train_dataloader, valid_dataloader, "valid")
                    rep.evaluate_attention(valid_dataloader)
                    rep.evaluate_embedding()
                    rep.evaluate_audio_embedding(valid_dataloader, "valid")
                else:
                    rep.evaluate_attention(test_dataloader)
                    if hp.gen_eval:
                        rep.evaluate_attention(gen_test_dataloader, gen_eval=True)
                    rep.evaluate_embedding()
                    rep.evaluate_audio_embedding(test_dataloader, "test")


def run_experiment(
    modality: str,
    trial_num: str,
    runs: range,
    resume_model_file: Optional[str] = None,
) -> None:
    if modality == "text":
        text(trial_num, runs, resume_model_file=resume_model_file)
    elif modality == "feature":
        feature(trial_num, runs, resume_model_file=resume_model_file)
    else:
        audio(trial_num, runs, resume_model_file=resume_model_file)
