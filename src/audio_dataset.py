# created 2025/01/08
# updated 2025/03/31 incorporating source and target labels
# A script to load custom dataset with self-defined class inherited from torch Dataset

import os
from typing import Any, List, Tuple
import math
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split, default_collate
from text_dataset import Alphabet
import hyper_params as hp


class AudioDataset(Dataset):
    def __init__(
        self,
        annotations_file: str,
        audio_dir: str,
        special_tokens: List[str],
        wav2mel: bool = True,
        power2db: bool = True,
        device: str = 'cuda',
    ) -> None:
        # get the list of ur and sr words
        # note that word text are separate from word reference for audio extraction
        self.device = device
        self.audio_dir = audio_dir
        self.annotations = pd.read_csv(annotations_file)
        self.ur_words = self.annotations["ur_string"]
        self.sr_words = self.annotations["sr_string"]
        self.ur_refs = self.annotations["ur_var"]
        self.sr_refs = self.annotations["sr_var"]

        # define special characters
        self.specials = special_tokens
        self.pad_idx = self.specials.index("<PAD>")

        # build ur alphabet
        self.ur_alphabet = Alphabet(self.specials)
        self.ur_alphabet.build_alphabet(self.ur_words)

        # build sr alphabet
        self.sr_alphabet = Alphabet(self.specials)
        self.sr_alphabet.build_alphabet(self.sr_words)

        # audio attributes
        self.sample_rate = hp.sample_rate
        self.n_samples = hp.n_samples
        self.n_mels = hp.n_mels

        # audio transformation
        self.wav2mel = wav2mel
        self.power2db = power2db

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # audio: [n_channels, n_samples]
        # retrieve source audio
        src_ref = self.ur_refs[index]
        src_path = os.path.join(self.audio_dir, src_ref + ".wav")
        src_audio, src_sr = torchaudio.load_with_torchcodec(src_path)
        src_audio = src_audio.to(self.device)

        # retrieve target audio
        trg_ref = self.sr_refs[index]
        trg_path = os.path.join(self.audio_dir, trg_ref + ".wav")
        trg_audio, trg_sr = torchaudio.load_with_torchcodec(trg_path)
        trg_audio = trg_audio.to(self.device)

        # pre-process source and target audio
        src_audio = self.resampling(src_audio, src_sr)
        trg_audio = self.resampling(trg_audio, trg_sr)
        src_audio, trg_audio = self.padding(src_audio, trg_audio)

        # transform source and target audio
        if self.wav2mel:  # [n_channels, n_freq, dur]
            src_audio = self.wav_to_mel(src_audio)
            trg_audio = self.wav_to_mel(trg_audio)
        if self.power2db:
            src_audio = self.power_to_db(src_audio)
            trg_audio = self.power_to_db(trg_audio)

        # get source text
        src = self.ur_words[index]
        src_vector = self.ur_alphabet.word2vec(src)
        src_tensor = torch.tensor(src_vector).to(self.device)

        # get target text
        trg = self.sr_words[index]
        trg_vector = self.sr_alphabet.word2vec(trg)
        trg_tensor = torch.tensor(trg_vector).to(self.device)

        return src_tensor, src_audio, trg_tensor, trg_audio

    def resampling(self, signal: torch.Tensor, sr: int) -> torch.Tensor:
        # in this project, we expect all sr == self.sample_rate
        assert (
            sr == self.sample_rate
        ), f"All audio data should have {self.sample_rate} sample rate!"

        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate).to(self.device)
            signal = resampler(signal)
        return signal

    def padding(self, signal1: torch.Tensor, signal2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # in this project, we are padding to a maximum length
        # so we expect all length_signal < self.n_samples
        length_signal1 = signal1.shape[1]
        length_signal2 = signal2.shape[1]
        assert (
            length_signal1 < self.n_samples and length_signal2 < self.n_samples
        ), f"All audio data should have less than {self.n_samples} samples!"

        # in this project, src and trg start with one frame of padded 0s
        # and the rest of padding at the end of the signal
        signal1 = torch.nn.functional.pad(signal1, (1, self.n_samples - length_signal1 - 1))
        signal2 = torch.nn.functional.pad(signal2, (1, self.n_samples - length_signal2 - 1))
        # [1, [1, 1, 1]] -> [1, [0, 1, 1, 1, 0, 0]]
        # [1, [1, 1]] -> [1, [0, 1, 1, 0, 0, 0]]
        return signal1, signal2

    def remove_padding(self, mel: torch.Tensor) -> torch.Tensor:
        # mel = [n_fre, dur]
        # get the values along n_freq dimension that are not 0s
        non_zeros = torch.all(torch.where(torch.eq(mel, -100), False, True), dim=0)

        return non_zeros

    # converting waveform to mel spectrogram
    def wav_to_mel(self, signal: torch.Tensor) -> torch.Tensor:
        mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,  # sampling rate, i.e. 24000 samples in 1s
            n_fft=1024,  # length of the FFT window
            # vowel length normally 50-100ms
            # we select 40ms for each window --> 960 samples --> round up to 1024 samples as it is power of 2
            # the higher n_fft is, the better frequency resolution it gets
            hop_length=256,  # number of samples overlapping between successive frames
            # we select 1024/4 = 256
            # the shorter hop_length, the higher temporal resolution it gets
            n_mels=self.n_mels,  # number of Mel bands to generate, normally 128
        ).to(self.device)
        signal = mel_spectrogram(signal)
        # [n_channels, n_mels, n_samples//hop_length+1]
        return signal

    # converting power scale to decibel scale in spectrogram
    # for readability of the spectrogram figure
    def power_to_db(self, signal: torch.Tensor) -> torch.Tensor:
        db_spectrogram = T.AmplitudeToDB(stype="power").to(self.device)
        signal = db_spectrogram(signal)
        return signal

    def split_dataset(self, data_split_ratio: List[float]) -> Any:
        return random_split(self, data_split_ratio)

    def sample_dataset(self, data_percentage: float) -> Dataset:
        if data_percentage == 1:
            return self

        subset_size = math.ceil(len(self) * data_percentage)
        subset_indices = torch.randperm(len(self))[:subset_size].tolist()
        return Subset(self, subset_indices)

    # a closure of customized collate_fn
    def get_collate_fn(self):
        def collate_fn(
            batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            src_labels = [src_txt for src_txt, _, _, _ in batch]
            src_labels = nn.utils.rnn.pad_sequence(src_labels, batch_first=False, padding_value=self.pad_idx)
            src_audios = [src_aud for _, src_aud, _, _ in batch]
            src_audios = default_collate(src_audios)

            trg_labels = [trg_txt for _, _, trg_txt, _ in batch]
            trg_labels = nn.utils.rnn.pad_sequence(trg_labels, batch_first=False, padding_value=self.pad_idx)
            trg_audios = [trg_aud for _, _, _, trg_aud in batch]
            trg_audios = default_collate(trg_audios)
            return src_labels, src_audios, trg_labels, trg_audios
        return collate_fn

    def get_dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.get_collate_fn(),
            drop_last=True  # drop incomplete batch
        )
        return data_loader


if __name__ == "__main__":
    import utils

    print(" - Loading dataset:")
    audio_dir = os.path.join(hp.audio_root, hp.lang_name)
    annotations_file = (
        "data/" + "_".join(
            part for part in [hp.lang_name, hp.directionality[0], hp.property, hp.conditions[0]] if part
        ) + ".csv"
    )
    annotations = pd.read_csv(annotations_file)
    print(f"Dataset size: {len(annotations)}")
    print(f"Sample data token: {annotations.iloc[0]}")

    print(" - Audio preprocessing:")
    audio_dataset = AudioDataset(annotations_file, audio_dir, hp.special_tokens,
                                 wav2mel=True, power2db=True, device='cpu')
    src_txt, src_aud, trg_txt, trg_aud = audio_dataset[0]
    print(f"Sample source text: {src_txt.shape}")
    print(f"Sample target text: {trg_txt.shape}")
    print(f"Sample source audio: {src_aud.shape}")
    print(f"Sample target audio: {trg_aud.shape}")

    utils.plot_spectrogram(src_aud, trg_aud, "src", "trg")

    print(" - Creating dataloader:")
    audio_dataloader = audio_dataset.get_dataloader(audio_dataset, hp.batch_size)
    dataiter = iter(audio_dataloader)
    source_text, source_audio, target_text, target_audio = next(dataiter)
    print(f"Sample source text: {source_text.shape}")
    print(f"Sample target text: {target_text.shape}")
    print(f"Sample source audio: {source_audio.shape}")
    print(f"Sample target audio: {target_audio.shape}")
