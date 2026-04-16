import os
from typing import Any, List, Mapping, Sequence, Tuple
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from nooverlap import push_text_free


def save_to_file(data_store: Mapping[str, Any], save_file: str) -> None:
    # convert dictionary to pandas dataframe
    data_df = pd.DataFrame(data_store)

    # write to new csv file or append to existing file
    if os.path.exists(save_file):
        data_df.to_csv(save_file, header=False, index=False, mode='a')
    else:
        data_df.to_csv(save_file, header=True, index=False, mode='w')

def plot_waveform(waveform: torch.Tensor, sample_rate: int, title: str = "Waveform") -> None:
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

def plot_spectrogram(
    spectrogram1: torch.Tensor,
    spectrogram2: torch.Tensor,
    spectrogram1_name: str,
    spectrogram2_name: str,
    title: str = "Spectrogram",
) -> None:
    spectrogram1 = spectrogram1[0]  # [1, n_freq, dur]
    spectrogram2 = spectrogram2[0]  # [1, n_freq, dur]

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

def plot_txt_acc(acc_store: Mapping[str, Any], acc_plot: str) -> None:
    # convert dictionary to pandas dataframe
    acc_data = pd.DataFrame(acc_store)

    # separate into training, validation, and test
    train_data = acc_data[acc_data["record_type"] == "train"]
    valid_data = acc_data[acc_data["record_type"] == "valid"]
    test_data = acc_data[acc_data["record_type"] == "test"]
    generalization_data = acc_data[acc_data["record_type"] == "generalization"]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all', figsize=(6, 6))  # create a 2 * 1 plot
    ax1.plot(train_data["epoch"], train_data["loss"], label="train")
    ax1.plot(valid_data["epoch"], valid_data["loss"], label="valid")
    ax1.plot(test_data["epoch"], test_data["loss"], label="test")
    if len(generalization_data) > 0:
        ax1.plot(generalization_data["epoch"], generalization_data["loss"], label="generalization")
    ax1.legend()
    ax1.set_title("Loss")
    ax1.set_ylim(0, 3)
    ax2.plot(train_data["epoch"], train_data["acc"], label="train")
    ax2.plot(valid_data["epoch"], valid_data["acc"], label="valid")
    ax2.plot(test_data["epoch"], test_data["acc"], label="test")
    if len(generalization_data) > 0:
        ax2.plot(generalization_data["epoch"], generalization_data["acc"], label="generalization")
    ax2.legend()
    ax2.set_title("Acc")
    ax2.set_ylim(0, 1)

    plt.savefig(acc_plot)
    plt.close()
    #plt.show()

def plot_aud_acc(acc_store: Mapping[str, Any], acc_plot: str) -> None:
    # convert dictionary to pandas dataframe
    acc_data = pd.DataFrame(acc_store)

    # separate into training, validation, and test
    train_data = acc_data[acc_data["record_type"] == "train"]
    valid_data = acc_data[acc_data["record_type"] == "valid"]
    test_data = acc_data[acc_data["record_type"] == "test"]
    generalization_data = acc_data[acc_data["record_type"] == "generalization"]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='all', figsize=(6, 8))  # create a 3 * 1 plot
    ax1.plot(train_data["epoch"], train_data["rec_loss"], label="train")
    ax1.plot(valid_data["epoch"], valid_data["rec_loss"], label="valid")
    ax1.plot(test_data["epoch"], test_data["rec_loss"], label="test")
    if len(generalization_data) > 0:
        ax1.plot(generalization_data["epoch"], generalization_data["rec_loss"], label="generalization")
    ax1.legend()
    ax1.set_title("Reconstruction Loss")
    ax1.set_ylim(0, 20)
    ax2.plot(train_data["epoch"], train_data["pred_loss"], label="train")
    ax2.plot(valid_data["epoch"], valid_data["pred_loss"], label="valid")
    ax2.plot(test_data["epoch"], test_data["pred_loss"], label="test")
    if len(generalization_data) > 0:
        ax2.plot(generalization_data["epoch"], generalization_data["pred_loss"], label="generalization")
    ax2.legend()
    ax2.set_title("Prediction Loss")
    ax2.set_ylim(0, 3)
    ax3.plot(train_data["epoch"], train_data["pred_acc"], label="train")
    ax3.plot(valid_data["epoch"], valid_data["pred_acc"], label="valid")
    ax3.plot(test_data["epoch"], test_data["pred_acc"], label="test")
    if len(generalization_data) > 0:
        ax3.plot(generalization_data["epoch"], generalization_data["pred_acc"], label="generalization")
    ax3.legend()
    ax3.set_title("Prediction Acc")
    ax3.set_ylim(0, 1)

    plt.savefig(acc_plot)
    plt.close()
    #plt.show()

def plot_txt_att(ur: Sequence[str], sr: Sequence[str], attention: torch.Tensor, att_plot: str) -> None:
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

def plot_aud_att(
    ur_aud: torch.Tensor,
    sr_txt: Sequence[str],
    sr_aud: torch.Tensor,
    txt_attention: torch.Tensor,
    aud_attention: torch.Tensor,
    txt_att_plot: str,
    aud_att_plot: str,
) -> None:
    # convert data to numpy array
    ur_aud = ur_aud.cpu().numpy()
    sr_aud = sr_aud.cpu().numpy()
    txt_attention = txt_attention.cpu().numpy()
    aud_attention = aud_attention.cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6),
                                   gridspec_kw={'height_ratios': [1, 1]})
    ax1.imshow(ur_aud, origin='lower', aspect='auto')
    ax2.imshow(txt_attention, origin='lower', aspect='auto', cmap="bone")
    ax2.set_yticks(ticks=np.arange(len(sr_txt)), labels=sr_txt)
    plt.tight_layout()
    plt.savefig(txt_att_plot)
    plt.close()
    #plt.show()

def read_vowel_intervals(textgrid_file: str, vowel_labels: Sequence[str]) -> List[Tuple[float, float, str]]:
    vowel_label_set = set(vowel_labels)
    tiers = []
    current_tier = None
    current_interval = {}
    in_interval = False

    with open(textgrid_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith("class = ") and '"IntervalTier"' in line:
                if current_tier is not None:
                    tiers.append(current_tier)
                current_tier = {"name": "", "intervals": []}
                current_interval = {}
                in_interval = False
            elif current_tier is not None and line.startswith("name = "):
                current_tier["name"] = line.split("=", 1)[1].strip().strip('"')
            elif current_tier is not None and line.startswith("intervals ["):
                current_interval = {}
                in_interval = True
            elif current_tier is not None and in_interval and line.startswith("xmin = "):
                value = float(line.split("=", 1)[1].strip())
                if "xmin" not in current_interval:
                    current_interval["xmin"] = value
            elif current_tier is not None and in_interval and line.startswith("xmax = "):
                value = float(line.split("=", 1)[1].strip())
                if "xmin" in current_interval and "xmax" not in current_interval:
                    current_interval["xmax"] = value
            elif current_tier is not None and in_interval and line.startswith("text = "):
                current_interval["text"] = line.split("=", 1)[1].strip().strip('"')
                if {"xmin", "xmax", "text"} <= current_interval.keys():
                    current_tier["intervals"].append(
                        (current_interval["xmin"], current_interval["xmax"], current_interval["text"])
                    )
                    current_interval = {}
                    in_interval = False

    if current_tier is not None:
        tiers.append(current_tier)

    best_intervals = []
    best_match_count = -1
    for tier in tiers:
        matched = [interval for interval in tier["intervals"] if interval[2] in vowel_label_set]
        if len(matched) > best_match_count:
            best_match_count = len(matched)
            best_intervals = matched
    return best_intervals


def interval_to_frame_span(
    start_time: float,
    end_time: float,
    max_frames: int,
    sample_rate: int,
    hop_length: int,
    start_frame_offset: int = 1,
) -> Tuple[int, int]:
    start_frame = int(round(start_time * sample_rate / hop_length)) + start_frame_offset
    end_frame = int(round(end_time * sample_rate / hop_length)) + start_frame_offset
    start_frame = max(0, min(start_frame, max_frames - 1))
    end_frame = max(start_frame + 1, min(end_frame, max_frames))
    return start_frame, end_frame


def plot_audio_embedding(embedding_store: Mapping[str, Sequence[Any]], embed_plot: str, title: str) -> None:
    embed_df = pd.DataFrame(embedding_store)
    feature_cols = [col for col in embed_df.columns if col.startswith("mel_")]
    if len(embed_df) == 0 or len(feature_cols) < 2:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.text(0.5, 0.5, "No vowel tokens extracted", ha="center", va="center")
        ax.set_axis_off()
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(embed_plot, dpi=300)
        plt.close()
        return

    reduced = PCA(n_components=2).fit_transform(embed_df[feature_cols])
    embed_df["pc1"] = reduced[:, 0]
    embed_df["pc2"] = reduced[:, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    for vowel_label in sorted(embed_df["vowel_label"].unique()):
        vowel_data = embed_df[embed_df["vowel_label"] == vowel_label]
        ax.scatter(vowel_data["pc1"], vowel_data["pc2"], s=15, alpha=0.7, label=vowel_label)
    ax.set_xlabel("pc1")
    ax.set_ylabel("pc2")
    ax.set_title(title)
    ax.legend(ncols=2, fontsize=7)
    plt.tight_layout()
    plt.savefig(embed_plot, dpi=300)
    plt.close()

def plot_embed(embed_store: Mapping[str, Any], focus_list: Sequence[str], embed_plot: str) -> None:
    # convert dictionary to pandas dataframe
    embed_df = pd.DataFrame.from_dict(embed_store, orient='index')
    # extract focus embeddings
    focus_embed_df = embed_df[embed_df.index.isin(focus_list)]

    # reorder indices
    embed_new_idx = ['m', 'n', 'ŋ', 'p', 't', 'k', 'b', 'd', 'g', 'f', 's', 'θ', 'ʃ', 'v', 'z', 'ð', 'ʒ', 'h',
                     'i', 'e', 'u', 'o', 'ɪ', 'ɛ', 'ʊ', 'ɔ']
    focus_embed_new_idx = ['i', 'e', 'u', 'o', 'ɪ', 'ɛ', 'ʊ', 'ɔ']
    embed_df = embed_df.reindex(embed_new_idx)
    focus_embed_df = focus_embed_df.reindex(focus_embed_new_idx)

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

    plt.rcParams.update({'font.size': 5})
    fig = plt.figure(figsize=(8,3))

    ax2 = fig.add_subplot(131, projection='3d')
    for i in focus_reduced_df.index:
        ax2.scatter(xs=focus_reduced_df.loc[i, 'pc1'],
                    ys=focus_reduced_df.loc[i, 'pc2'],
                    zs=focus_reduced_df.loc[i, 'pc3'],
                    s=5,
                    c=plt.colormaps.get_cmap('tab20')(i % 18),
                    label=focus_reduced_df.loc[i, 'phoneme'])
        ax2.text(x=focus_reduced_df.loc[i, 'pc1'],
                 y=focus_reduced_df.loc[i, 'pc2'],
                 z=focus_reduced_df.loc[i, 'pc3'],
                 s=focus_reduced_df.loc[i, 'phoneme'],
                 ha='left',
                 va='bottom')
    ax2.set_xlabel("pc1")
    ax2.set_ylabel("pc2")
    ax2.set_zlabel("pc3")
    ax2.set_title("Vowel embedding")
    ax2.legend(loc='center left', bbox_to_anchor=(1.1, 0.5)).remove()
    push_text_free(fig, ax2)

    ax1 = fig.add_subplot(132, projection='3d')
    for i in reduced_df.index:
        ax1.scatter(xs=reduced_df.loc[i, 'pc1'],
                   ys=reduced_df.loc[i, 'pc2'],
                   zs=reduced_df.loc[i, 'pc3'],
                    s=5,
                   c=plt.colormaps.get_cmap('tab20')(i%18), # if there are more than 18 categories, reuse from top
                   label=reduced_df.loc[i, 'phoneme'])
        ax1.text(x=reduced_df.loc[i, 'pc1'],
                 y=reduced_df.loc[i, 'pc2'],
                 z=reduced_df.loc[i, 'pc3'],
                 s=reduced_df.loc[i, 'phoneme'],
                 ha='left',
                 va='bottom')
    ax1.set_xlabel("pc1")
    ax1.set_ylabel("pc2")
    ax1.set_zlabel("pc3")
    ax1.set_title("Phoneme embedding")
    ax1.legend(ncols=2, loc='center left', bbox_to_anchor=(1.3, 0.5))
    push_text_free(fig, ax1)

    plt.savefig(embed_plot, dpi=300)
    plt.close()
    #plt.show()

def plot_embed_updated(
    embed_store: Mapping[str, Mapping[str, Any]],
    focus_list: Sequence[str],
    embed_plot: str,
    focus_embed_plot: str,
) -> None:

    dfs = []
    # iterate through all dictionaries and read into dataframes
    for key, value in embed_store.items():
        # convert dictionary to pandas dataframe
        df = pd.DataFrame.from_dict(value, orient='index')
        # add new columns with phoneme and file name
        df['phoneme'] = df.index
        df['file_name'] = key
        # append to list of dataframes
        dfs.append(df)
    # combine lists of dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Parse metadata from filenames like:
    # EnglishBH_shortened_txt_l2r_harmony_run0_epoch10_embedding.csv
    metadata = combined_df['file_name'].str.extract(
        r'^(?P<language>.+)_(?P<modality>txt|fea|aud)_(?P<directionality>[^_]+)_(?P<condition>[^_]+)_run(?P<run_num>\d+)_epoch(?P<epoch>-?\d+)_(?P<suffix>.+)$'
    )
    if metadata.isnull().any().any():
        bad_files = combined_df.loc[metadata.isnull().any(axis=1), 'file_name'].tolist()
        raise RuntimeError(f"Could not parse embedding metadata from file names: {bad_files}")
    metadata['run_num'] = metadata['run_num'].astype(int)
    metadata['epoch'] = metadata['epoch'].astype(int)
    combined_df = pd.concat([combined_df, metadata], axis=1)

    # extract focus embeddings
    focus_combined_df = combined_df[combined_df['phoneme'].isin(focus_list)]
    focus_combined_df = focus_combined_df.reset_index()

    # use PCA to project the data from embedding_dim to 3D
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(combined_df.loc[:, 0:10])
    reduced_df = pd.DataFrame(data=reduced_data,
                              columns=['pc1', 'pc2', 'pc3'])

    focus_pca = PCA(n_components=3)
    focus_reduced_data = focus_pca.fit_transform(focus_combined_df.loc[:, 0:10])
    focus_reduced_df = pd.DataFrame(data=focus_reduced_data,
                                    columns=['pc1', 'pc2', 'pc3'])

    # combine metalinguistic information
    combined_df = pd.concat([combined_df, reduced_df], axis=1)
    focus_combined_df = pd.concat([focus_combined_df, focus_reduced_df], axis=1)

    # reorder phonemes
    phoneme = pd.CategoricalDtype(categories=['m', 'n', 'ŋ', 'p', 't', 'k', 'b', 'd', 'g',
                                              'f', 's', 'θ', 'ʃ', 'v', 'z', 'ð', 'ʒ', 'h',
                                              'i', 'e', 'u', 'o', 'ɪ', 'ɛ', 'ʊ', 'ɔ'],
                                  ordered=True)
    vowel = pd.CategoricalDtype(categories=['i', 'e', 'u', 'o', 'ɪ', 'ɛ', 'ʊ', 'ɔ'],
                                ordered=True)
    combined_df['phoneme'] = combined_df['phoneme'].astype(phoneme)
    focus_combined_df['phoneme'] = focus_combined_df['phoneme'].astype(vowel)

    # reorder epochs
    combined_df = combined_df.sort_values(by=['phoneme', 'epoch'], ascending=[True, True])
    focus_combined_df = focus_combined_df.sort_values(by=['phoneme', 'epoch'], ascending=[True, True])

    # color palette
    color_palette = [
        'rgb(138, 29, 99)', 'rgb(107, 24, 93)', 'rgb(76, 21, 80)',
        'rgb(250, 205, 145)', 'rgb(246, 173, 119)', 'rgb(240, 142, 98)',
        'rgb(216, 80, 83)', 'rgb(195, 56, 90)', 'rgb(168, 40, 96)',
        'rgb(18, 78, 43)', 'rgb(34, 120, 36)', 'rgb(115, 152, 5)', 'rgb(195, 182, 59)',
        'rgb(140, 193, 186)', 'rgb(60, 154, 171)', 'rgb(30, 110, 161)', 'rgb(38, 62, 144)',
        'rgb(239, 226, 156)',
        'rgb(158,1,66)', 'rgb(213,62,79)', 'rgb(94,79,162)', 'rgb(50,136,189)',
        'rgb(244,109,67)', 'rgb(253,174,97)', 'rgb(102,194,165)', 'rgb(171,221,164)'
    ]
    focus_color_palette = [
        'rgb(158,1,66)', 'rgb(213,62,79)', 'rgb(94,79,162)', 'rgb(50,136,189)',
        'rgb(244,109,67)', 'rgb(253,174,97)', 'rgb(102,194,165)', 'rgb(171,221,164)'
    ]

    # create 3d scatter plot with plotly
    fig = px.scatter_3d(combined_df,
                        x='pc1',
                        y='pc2',
                        z='pc3',
                        color='phoneme',
                        color_discrete_sequence=color_palette,
                        animation_frame='epoch')
    fig.write_html(embed_plot)
    plt.close()
    #plt.show()

    fig = px.scatter_3d(focus_combined_df,
                        x='pc1',
                        y='pc2',
                        z='pc3',
                        color='phoneme',
                        color_discrete_sequence=focus_color_palette,
                        animation_frame='epoch')
    fig.write_html(focus_embed_plot)
    plt.close()
    #plt.show()
