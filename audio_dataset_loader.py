# 2025/01/08
# A script to load custom dataset with self-defined class inherited from torch Dataset

import os
import re
import pandas as pd
import random
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from text_dataset_loader import Alphabet


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
    def __init__(self, annotations_file, audio_dir, special_tokens,
                 sample_rate, n_samples, n_fft, hop_length, n_mels,
                 wav2mel=True, power2db=True, normalize=True, device='cuda'):
        # get the list of ur and sr words
        self.device = device
        self.audio_dir = audio_dir
        self.annotations = pd.read_csv(annotations_file)
        self.ur = self.annotations["ur"]
        self.sr = self.annotations["sr"]

        # define special characters
        self.specials = special_tokens
        self.pad_idx = self.specials.index("<PAD>")

        # build ur alphabet
        self.ur_name = re.split('[/_.]', annotations_file)[2] + "_ur"
        self.ur_alphabet = Alphabet(self.ur_name)
        self.ur_alphabet.build_alphabet(self.ur, self.specials)

        # build sr alphabet
        self.sr_name = re.split('[/_.]', annotations_file)[2] + "_sr"
        self.sr_alphabet = Alphabet(self.sr_name)
        self.sr_alphabet.build_alphabet(self.sr, self.specials)

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
        # audio: [n_channels, n_samples]
        # retrieve and pre-process source audio
        src_label = self.ur[index]
        src_path = os.path.join(self.audio_dir, (src_label + ".wav"))
        src_audio, src_sr = torchaudio.load(src_path, format="wav")
        src_audio = src_audio.to(self.device)
        src_audio = self._resampling(src_audio, src_sr)
        src_audio = self._padding(src_audio)

        # retrieve and pre-process target audio
        trg_label = self.sr[index]
        trg_path = os.path.join(self.audio_dir, (trg_label + ".wav"))
        trg_audio, trg_sr = torchaudio.load(trg_path, format="wav")
        trg_audio = trg_audio.to(self.device)
        trg_audio = self._resampling(trg_audio, trg_sr)
        trg_audio = self._padding(trg_audio)

        # transform source and target audio
        if self.wav2mel:
            src_audio = self._wav_to_mel(src_audio)
            trg_audio = self._wav_to_mel(trg_audio)
            # [n_channels, freq, dur]
        if self.power2db:
            src_audio = self._power_to_db(src_audio)
            trg_audio = self._power_to_db(trg_audio)

        # get source text
        src = self.ur[index]
        src_vector = self.ur_alphabet.word2vec(src)
        src_tensor = torch.tensor(src_vector).to(self.device)

        # get target text
        trg = self.sr[index]
        trg_vector = self.sr_alphabet.word2vec(trg)
        trg_tensor = torch.tensor(trg_vector).to(self.device)

        return src_tensor, src_audio, trg_tensor, trg_audio

    def _resampling(self, signal, sr):

        # in this project, we expect all sr == self.sample_rate
        assert (
            sr == self.sample_rate
        ), f"All audio data should have {self.sample_rate} sample rate!"

        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate).to(self.device)
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

    # converting waveform to mel spectrogram
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
        # [n_channels, n_mels, n_samples//hop_length+1]
        return signal

    # converting power scale to decibel scale in spectrogram
    # only for readability of the spectrogram figure
    def _power_to_db(self, signal):
        db_spectrogram = T.AmplitudeToDB(stype="power").to(self.device)
        signal = db_spectrogram(signal)
        return signal

    def split_dataset(self, data_split_ratio):
        return random_split(self, data_split_ratio)

    # a closure of customized collate_fn
    def get_collate_fn(self):
        def collate_fn(batch):
            src_labels = [item[0] for item in batch]
            src_labels = pad_sequence(src_labels, batch_first=False, padding_value=self.pad_idx)
            src_audios = [item[1] for item in batch]

            trg_labels = [item[2] for item in batch]
            trg_labels = pad_sequence(trg_labels, batch_first=False, padding_value=self.pad_idx)
            trg_audios = [item[3] for item in batch]
            return src_labels, src_audios, trg_labels, trg_audios
        return collate_fn

    def get_dataloader(self, batch_size, shuffle=True):
        collate_fn = self.get_collate_fn()

        data_loader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn
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
    audio_dataset = AudioDataset(annotations_file, audio_dir, hp.special_tokens,
                                 hp.sample_rate, hp.n_samples, hp.n_fft, hp.hop_length, hp.n_mels,
                                 wav2mel=True, power2db=True, device='cpu')
    src_txt, src_aud, trg_txt, trg_aud = audio_dataset[0]
    print("Sample source text:", src_txt.shape,
          "\nSample source audio:", src_aud.shape,
          "\nSample target text:", trg_txt.shape,
          "\nSample target audio:", trg_aud.shape)

    plot_spectrogram(src_aud)
    plot_spectrogram(trg_aud)

    print(" - Creating dataloader:")
    audio_dataloader = audio_dataset.get_dataloader(hp.batch_size)
    dataiter = iter(audio_dataloader)
    source, target = next(dataiter)
    print("Sample source:", source.shape,
          "\nSample target:", target.shape)
