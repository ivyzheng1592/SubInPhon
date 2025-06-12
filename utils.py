import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from sklearn.decomposition import PCA
from nooverlap import push_text_free


def save_to_file(data_store, save_file):
    # convert dictionary to pandas dataframe
    data_df = pd.DataFrame(data_store)

    # write to new csv file or append to existing file
    if os.path.exists(save_file):
        data_df.to_csv(save_file, header=False, index=False, mode='a')
    else:
        data_df.to_csv(save_file, header=True, index=False, mode='w')

def plot_waveform(waveform, sample_rate, title="Waveform"):
    waveform = waveform.cpu().numpy()  # [n_channels, n_samples]
    time_axis = torch.arange(0, waveform.shape[1]) / sample_rate

    fig, axs = plt.subplots()
    axs.set_xlabel("time")
    axs.set_ylabel("amplitude")
    axs.plot(time_axis, waveform[0], linewidth=1)
    axs.grid(visible=True)
    fig.suptitle(title)
    plt.close()
    #plt.show()

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
    plt.close()
    #plt.show()

def plot_acc(acc_store, acc_plot):
    # read in accuracy data and separate into training and validation
    acc_data = pd.DataFrame(acc_store)
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
    plt.close()
    #plt.show()

def plot_att(ur, sr, attention, att_plot):
    # convert attention data to numpy array
    attention = attention.cpu().numpy()

    fig, ax = plt.subplots(1, 1)
    im = ax.matshow(attention, cmap="bone")
    ax.set_xticks(ticks=np.arange(len(ur)), labels=ur)
    ax.set_yticks(ticks=np.arange(len(sr)), labels=sr)
    fig.colorbar(im)
    plt.savefig(att_plot)
    plt.close()
    #plt.show()

def plot_embed(embed_store, embed_plot, title="vowel embedding"):
    # convert dictionary to pandas dataframe
    embed_df = pd.DataFrame.from_dict(embed_store, orient='index')

    # use PCA to project the data from embedding_dim to 3D
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(embed_df)
    reduced_df = pd.DataFrame(data=reduced_data,
                              columns=['pc1', 'pc2', 'pc3'])
    reduced_df['phone'] = embed_df.index

    # save PCA results to file
    #save_to_file(reduced_df, embed_file)

    # decide a colormap based on whether the plot is on vowels or all phonemes
    if title == "phoneme embedding":
        colormap = colormaps.get_cmap('tab20')
        legend_col = 2
    else:
        colormap = colormaps.get_cmap('tab10')
        legend_col = 1

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in reduced_df.index:
        ax.scatter(xs=reduced_df.loc[i, 'pc1'],
                   ys=reduced_df.loc[i, 'pc2'],
                   zs=reduced_df.loc[i, 'pc3'],
                   c=colormap(i%20), # if there are more than 20 categories, reuse from top
                   label=reduced_df.loc[i, 'phone'])
        ax.text(x=reduced_df.loc[i, 'pc1'],
                y=reduced_df.loc[i, 'pc2'],
                z=reduced_df.loc[i, 'pc3'],
                s=reduced_df.loc[i, 'phone'])
    ax.set_xlabel("pc1")
    ax.set_ylabel("pc2")
    ax.set_zlabel("pc3")
    ax.set_title(title)
    fig.legend(ncols=legend_col)
    push_text_free(fig, ax)
    plt.savefig(embed_plot)
    plt.close()
    #plt.show()