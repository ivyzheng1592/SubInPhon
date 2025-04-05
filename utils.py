import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_to_file(data_store, save_file):
    # convert dictionary to pandas dataframe
    data_df = pd.DataFrame(data_store)

    # write to csv file
    data_df.to_csv(save_file, index=False)

def plot_waveform(waveform, sample_rate, title="Waveform"):
    waveform = waveform.cpu().numpy()  # [n_channels, n_samples]
    time_axis = torch.arange(0, waveform.shape[1]) / sample_rate

    fig, axs = plt.subplots()
    axs.set_xlabel("time")
    axs.set_ylabel("amplitude")
    axs.plot(time_axis, waveform[0], linewidth=1)
    axs.grid(visible=True)
    fig.suptitle(title)
    plt.show()

def plot_spectrogram(spectrogram1, spectrogram2, spectrogram1_name, spectrogram2_name,
                     title="Spectrogram"):
    spectrogram1 = spectrogram1[0]  # [1, n_freq, n_samples]
    spectrogram2 = spectrogram2[0]  # [1, n_freq, n_samples]

    fig, (axs1, axs2) = plt.subplots(1, 2, sharey='all')
    axs1.set_xlabel("frame")
    axs2.set_xlabel("frame")
    axs1.set_ylabel("mel freq")
    axs1.set_title(spectrogram1_name)
    axs2.set_title(spectrogram2_name)
    axs1.imshow(spectrogram1, origin='lower', aspect='auto')
    axs2.imshow(spectrogram2, origin='lower', aspect='auto')
    #fig.colorbar()
    fig.suptitle(title)
    plt.show()

def plot_acc(acc_file, acc_plot):
    acc_data = pd.read_csv(acc_file)
    train_data = acc_data[acc_data["record_type"] == "train"]
    valid_data = acc_data[acc_data["record_type"] == "valid"]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all')  # create a 2 * 1 plot
    ax1.plot(train_data["epoch"], train_data["loss"], label="train")
    ax1.plot(valid_data["epoch"], valid_data["loss"], label="valid")
    ax1.legend()
    ax1.set_title("Loss")
    ax2.plot(train_data["epoch"], train_data["acc"], label="train")
    ax2.plot(valid_data["epoch"], valid_data["acc"], label="valid")
    ax2.legend()
    ax2.set_title("Acc")

    plt.savefig(acc_plot)
    plt.show()

def plot_att(ur, sr, attention, att_plot):
    fig, ax = plt.subplots(1, 1)
    attention = attention.cpu().numpy()
    ax.matshow(attention, cmap="bone")
    ax.set_xticks(ticks=np.arange(len(ur)), labels=ur)
    ax.set_yticks(ticks=np.arange(len(sr)), labels=sr)
    plt.savefig(att_plot)
    #plt.show()
    #plt.close()