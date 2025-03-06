# 2025/01/08
# A script to load custom dataset with self-defined class inherited from torch Dataset

import os
import pandas as pd
import hi
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import hyper_params as hp


def plot_waveform(waveform, sample_rate):
    pass


def plot_spectrogram(waveform, sample_rate):
    pass


class AudioDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, sample_rate, n_samples, device, wav2mel=True, power2db=True, normalize=True):
        # get the list of ur and sr words
        self.annotations = pd.read_csv(annotations_file)
        self.ur = self.annotations["ur"]
        self.sr = self.annotations["sr"]

        self.audio_dir = audio_dir
        self.device = device

        # audio attributes
        self.sample_rate = sample_rate
        self.n_samples = n_samples

        # audio transformation
        self.wav2mel = wav2mel
        self.power2db = power2db
        self.normalize = normalize

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # audio: torch.Tensor [n_channels, n_samples]
        # retrieve and pre-process source audio
        source_label = self.ur[index]
        source_path = os.path.join(self.audio_dir, (source_label + ".wav"))
        source_audio, source_sr = torchaudio.load(source_path, format="wav")
        source_audio = source_audio.to(self.device)
        source_audio = self._resampling(source_audio, source_sr)
        source_audio = self._padding(source_audio)

        # retrieve and pre-process target audio
        target_label = self.sr[index]
        target_path = os.path.join(self.audio_dir, (target_label + ".wav"))
        target_audio, target_sr = torchaudio.load(target_path, format="wav")
        target_audio = target_audio.to(self.device)
        target_audio = self._resampling(target_audio, target_sr)
        target_audio = self._padding(target_audio)

        # transform source and target audio
        if self.wav2mel:
            source_audio = self._wav_to_mel(source_audio)
            target_audio = self._wav_to_mel(target_audio)

        return source_audio, target_audio

    def _resampling(self, signal, sr):
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            signal = resampler(signal)
        return signal

    def _padding(self, signal):
        length_signal = signal.shape[1]

        # if the signal have more samples than we expect, cut the signal with slicing
        if length_signal > self.n_samples:
            signal = signal[:, :self.n_samples]
            # [1, signal.shape[1]] -> [1, self.n_samples]

        # if the signal have fewer samples than we expect, pad the signal
        if length_signal < self.n_samples:
            # Length of padding at the beginning and end of the signal
            pad_begin_len = random.randint(0, self.n_samples - length_signal)
            pad_end_len = self.n_samples - length_signal - pad_begin_len
            # Padding with 0s
            last_dim_padding = (pad_begin_len, pad_end_len)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
            # [1, [1, 1, 1]] -> [1, [0, 1, 1, 1, 0, 0]]

        return signal

    def _wav_to_mel(self, signal):
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,  # sampling rate, i.e. 24000 samples in 1s
            n_fft=1024,  # length of the FFT window
            # vowel length normally 50-100ms
            # we select 40ms for each window --> 960 points --> round up to 1024 points as it is power of 2
            # the higher n_fft is, the better frequency resolution it gets
            hop_length=256,  # number of samples overlapping between successive frames
            # we select 1024/4 = 256
            # the shorter hop_length, the higher temporal resolution it gets
            n_mels=128,  # number of Mel bands to generate, normally 128
        ).to(self.device)
        signal = mel_spectrogram(signal)
        return signal


def get_dataloader(dataset, batch_size=hp.batch_size, shuffle=True):
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return data_loader


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    print(" - Loading dataset:")
    audio_dir = "Dataset/audio/English"
    annotations_file = "Dataset/English_aud_harmony.csv"
    annotations = pd.read_csv(annotations_file)
    print("Dataset size: ", len(annotations),
          "\nSample data token:", annotations.iloc[0])

    print(" - Audio preprocessing:")
    audio_dataset = AudioDataset(annotations_file, audio_dir, hp.sample_rate, hp.n_samples, device)
    src, trg = audio_dataset[0]
    print("Sample source:", src,
          "\nSample target:", trg)

    print(" - Creating dataloader:")
    audio_dataloader = get_dataloader(audio_dataset)
    dataiter = iter(audio_dataloader)
    source, target = next(dataiter)
    print("Sample source:", source,
          "\nSample target:", target)