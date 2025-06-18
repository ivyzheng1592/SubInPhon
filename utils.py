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
    # convert dictionary to pandas dataframe
    acc_data = pd.DataFrame(acc_store)

    # separate into training and validation
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

def plot_embed(embed_df, focus_embed_df, embed_plot):
    # convert dictionary to pandas dataframe
    #embed_df = pd.DataFrame.from_dict(embed_store, orient='index')
    #focus_embed_df = pd.DataFrame.from_dict(focus_embed_store, orient='index')

    # use PCA to project the data from embedding_dim to 3D
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(embed_df)
    reduced_df = pd.DataFrame(data=reduced_data,
                              columns=['pc1', 'pc2', 'pc3'])
    reduced_df['phoneme'] = embed_df.index

    focus_pca = PCA(n_components=3)
    focus_reduced_data = focus_pca.fit_transform(focus_embed_df)
    focus_reduced_df = pd.DataFrame(data=focus_reduced_data,
                                    columns=['pc1', 'pc2', 'pc3'])
    focus_reduced_df['phoneme'] = focus_embed_df.index

    fig = plt.figure(figsize=(9,12))
    ax1 = fig.add_subplot(211, projection='3d')
    for i in reduced_df.index:
        ax1.scatter(xs=reduced_df.loc[i, 'pc1'],
                   ys=reduced_df.loc[i, 'pc2'],
                   zs=reduced_df.loc[i, 'pc3'],
                   #color=colormaps.get_cmap('tab20')(i%20), # if there are more than 20 categories, reuse from top
                   label=reduced_df.loc[i, 'phoneme'])
        ax1.text(x=reduced_df.loc[i, 'pc1'],
                 y=reduced_df.loc[i, 'pc2'],
                 z=reduced_df.loc[i, 'pc3'],
                 s=reduced_df.loc[i, 'phoneme'])
    ax1.set_xlabel("pc1")
    ax1.set_ylabel("pc2")
    ax1.set_zlabel("pc3")
    ax1.set_title("Phoneme embedding")
    ax1.legend(ncols=2, loc='center left', bbox_to_anchor=(1.1, 0.5))
    push_text_free(fig, ax1)

    ax2 = fig.add_subplot(212, projection='3d')
    for i in focus_reduced_df.index:
        ax2.scatter(xs=focus_reduced_df.loc[i, 'pc1'],
                    ys=focus_reduced_df.loc[i, 'pc2'],
                    zs=focus_reduced_df.loc[i, 'pc3'],
                    label=focus_reduced_df.loc[i, 'phoneme'])
        ax2.text(x=focus_reduced_df.loc[i, 'pc1'],
                 y=focus_reduced_df.loc[i, 'pc2'],
                 z=focus_reduced_df.loc[i, 'pc3'],
                 s=focus_reduced_df.loc[i, 'phoneme'])
    ax2.set_xlabel("pc1")
    ax2.set_ylabel("pc2")
    ax2.set_zlabel("pc3")
    ax2.set_title("Vowel embedding")
    ax2.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    push_text_free(fig, ax2)
    plt.savefig(embed_plot)
    plt.close()
    #plt.show()

if __name__ == "__main__":
    pass