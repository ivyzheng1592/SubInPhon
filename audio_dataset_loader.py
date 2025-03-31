# 2025/01/08
# A script to load custom dataset with self-defined class inherited from torch Dataset

import os
import pandas as pd
import random
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt


def plot_waveform(waveform, sample_rate, title="Waveform"):
    waveform = waveform.numpy()  # [n_channels, n_samples]
    time_axis = torch.arange(0, waveform.shape[1]) / sample_rate

    fig, axs = plt.subplots(1, 1)
    axs.set_xlabel("time")
    axs.set_ylabel("amplitude")
    axs.plot(time_axis, waveform[0], linewidth=1)
    axs.grid(visible=True)
    fig.suptitle(title)
    plt.show()

def plot_spectrogram(spectrogram, title="Spectrogram"):
    spectrogram = spectrogram[0]  # [1, n_freq, n_samples]

    fig, axs = plt.subplots(1, 1)
    axs.set_xlabel("frame")
    axs.set_ylabel("mel freq")
    im = axs.imshow(spectrogram, origin='lower', aspect='auto')
    fig.colorbar(im, ax=axs)
    fig.suptitle(title)
    plt.show()


class AudioDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, sample_rate, n_samples, n_fft, hop_length, n_mels,
                 wav2mel=True, power2db=True, normalize=True, device='cuda'):
        # get the list of ur and sr words
        self.annotations = pd.read_csv(annotations_file)
        self.ur = self.annotations["ur"]
        self.sr = self.annotations["sr"]

        self.audio_dir = audio_dir
        self.device = device

        # audio attributes
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

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
        if self.power2db:
            source_audio = self._power_to_db(source_audio)
            target_audio = self._power_to_db(target_audio)

        return source_audio, target_audio

    def _resampling(self, signal, sr):

        # in this project, we expect all sr == self.sample_rate
        assert (
            sr == self.sample_rate
        ), f"All audio data should have {self.sample_rate} sample rate!"

        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate).to(device)
            signal = resampler(signal)
        return signal

    def _padding(self, signal):
        length_signal = signal.shape[1]

        # in this project, we are padding to a maximum length
        # so we expect all length_signal < self.n_samples
        assert (
            length_signal < self.n_samples
        ), f"All audio data should have less than {self.n_samples} samples!"

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
        mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,  # sampling rate, i.e. 24000 samples in 1s
            n_fft=self.n_fft,  # length of the FFT window
            # vowel length normally 50-100ms
            # we select 40ms for each window --> 960 samples --> round up to 1024 samples as it is power of 2
            # the higher n_fft is, the better frequency resolution it gets
            hop_length=self.hop_length,  # number of samples overlapping between successive frames
            # we select 1024/4 = 256
            # the shorter hop_length, the higher temporal resolution it gets
            n_mels=self.n_mels,  # number of Mel bands to generate, normally 128
        ).to(self.device)
        signal = mel_spectrogram(signal)
        return signal

    def _power_to_db(self, signal):
        db_spectrogram = T.AmplitudeToDB(stype="power").to(self.device)
        signal = db_spectrogram(signal)
        return signal

    def split_dataset(self, data_split_ratio):
        return random_split(self, data_split_ratio)

    def get_dataloader(self, batch_size, shuffle=True):
        data_loader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle
        )
        return data_loader


if __name__ == "__main__":
    import hyper_params as hp

    print(" - Loading dataset:")
    audio_dir = "Dataset/audio/English"
    annotations_file = "Dataset/English_aud_harmony.csv"
    annotations = pd.read_csv(annotations_file)
    print("Dataset size: ", len(annotations),
          "\nSample data token:", annotations.iloc[0])

    print(" - Audio preprocessing:")
    audio_dataset = AudioDataset(annotations_file, audio_dir, hp.sample_rate, hp.n_samples,
                                 hp.n_fft, hp.hop_length, hp.n_mels,
                                 wav2mel=True, power2db=True, device='cpu')
    src, trg = audio_dataset[0]
    print("Sample source:", src.shape,
          "\nSample target:", trg.shape)

    plot_spectrogram(src)
    plot_spectrogram(trg)

    print(" - Creating dataloader:")
    audio_dataloader = audio_dataset.get_dataloader(hp.batch_size)
    dataiter = iter(audio_dataloader)
    source, target = next(dataiter)
    print("Sample source:", source.shape,
          "\nSample target:", target.shape)
