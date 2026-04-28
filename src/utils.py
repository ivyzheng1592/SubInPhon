import os
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from nooverlap import push_text_free
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
import torch


# File I/O

# Save one recorder or embedding store to CSV.
def save_to_file(data_store: Mapping[str, Any], save_file: str) -> None:
    data_df = pd.DataFrame(data_store)

    if os.path.exists(save_file):
        data_df.to_csv(save_file, header=False, index=False, mode="a")
    else:
        data_df.to_csv(save_file, header=True, index=False, mode="w")


# Basic signal plots

# Plot one waveform for quick inspection.
def plot_waveform(waveform: torch.Tensor, sample_rate: int, title: str = "Waveform") -> None:
    waveform = waveform.cpu().numpy()
    time_axis = torch.arange(0, waveform.shape[1]) / sample_rate

    fig, axs = plt.subplots()
    axs.set_xlabel("time")
    axs.set_ylabel("amplitude")
    axs.plot(time_axis, waveform[0], linewidth=1)
    axs.grid(visible=True)
    fig.suptitle(title)
    plt.close()


# Plot two spectrograms side by side.
def plot_spectrogram(
    spectrogram1: torch.Tensor,
    spectrogram2: torch.Tensor,
    spectrogram1_name: str,
    spectrogram2_name: str,
    title: str = "Spectrogram",
) -> None:
    spectrogram1 = spectrogram1[0]
    spectrogram2 = spectrogram2[0]

    fig, (axs1, axs2) = plt.subplots(1, 2, sharey="all")
    axs1.set_xlabel("frame")
    axs2.set_xlabel("frame")
    axs1.set_ylabel("mel freq")
    axs1.set_title(spectrogram1_name)
    axs2.set_title(spectrogram2_name)
    axs1.imshow(spectrogram1, origin="lower", aspect="auto")
    axs2.imshow(spectrogram2, origin="lower", aspect="auto")
    fig.suptitle(title)
    plt.close()


# Accuracy plots

# Plot text training loss and accuracy.
def plot_txt_acc(acc_store: Mapping[str, Any], acc_plot: str) -> None:
    acc_data = pd.DataFrame(acc_store)

    train_data = acc_data[acc_data["record_type"] == "train"]
    valid_data = acc_data[acc_data["record_type"] == "valid"]
    test_data = acc_data[acc_data["record_type"] == "test"]
    gen_data = acc_data[acc_data["record_type"] == "gen"]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex="all", figsize=(6, 6))
    ax1.plot(train_data["epoch"], train_data["loss"], label="train")
    ax1.plot(valid_data["epoch"], valid_data["loss"], label="valid")
    ax1.plot(test_data["epoch"], test_data["loss"], label="test")
    if len(gen_data) > 0:
        ax1.plot(gen_data["epoch"], gen_data["loss"], label="gen")
    ax1.legend()
    ax1.set_title("Loss")
    ax1.set_ylim(-0.1, 3.1)

    ax2.plot(train_data["epoch"], train_data["acc"], label="train")
    ax2.plot(valid_data["epoch"], valid_data["acc"], label="valid")
    ax2.plot(test_data["epoch"], test_data["acc"], label="test")
    if len(gen_data) > 0:
        ax2.plot(gen_data["epoch"], gen_data["acc"], label="gen")
    ax2.legend()
    ax2.set_title("Acc")
    ax2.set_ylim(-0.05, 1.05)

    plt.savefig(acc_plot)
    plt.close()


# Plot audio training losses and prediction accuracy.
def plot_aud_acc(acc_store: Mapping[str, Any], acc_plot: str) -> None:
    acc_data = pd.DataFrame(acc_store)

    train_data = acc_data[acc_data["record_type"] == "train"]
    valid_data = acc_data[acc_data["record_type"] == "valid"]
    test_data = acc_data[acc_data["record_type"] == "test"]
    gen_data = acc_data[acc_data["record_type"] == "gen"]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex="all", figsize=(6, 8))
    ax1.plot(train_data["epoch"], train_data["rec_loss"], label="train")
    ax1.plot(valid_data["epoch"], valid_data["rec_loss"], label="valid")
    ax1.plot(test_data["epoch"], test_data["rec_loss"], label="test")
    if len(gen_data) > 0:
        ax1.plot(gen_data["epoch"], gen_data["rec_loss"], label="gen")
    ax1.legend()
    ax1.set_title("Reconstruction Loss")
    ax1.set_ylim(-0.5, 20.5)

    ax2.plot(train_data["epoch"], train_data["pred_loss"], label="train")
    ax2.plot(valid_data["epoch"], valid_data["pred_loss"], label="valid")
    ax2.plot(test_data["epoch"], test_data["pred_loss"], label="test")
    if len(gen_data) > 0:
        ax2.plot(gen_data["epoch"], gen_data["pred_loss"], label="gen")
    ax2.legend()
    ax2.set_title("Prediction Loss")
    ax2.set_ylim(-0.1, 3.1)

    ax3.plot(train_data["epoch"], train_data["pred_acc"], label="train")
    ax3.plot(valid_data["epoch"], valid_data["pred_acc"], label="valid")
    ax3.plot(test_data["epoch"], test_data["pred_acc"], label="test")
    if len(gen_data) > 0:
        ax3.plot(gen_data["epoch"], gen_data["pred_acc"], label="gen")
    ax3.legend()
    ax3.set_title("Prediction Acc")
    ax3.set_ylim(-0.05, 1.05)

    plt.savefig(acc_plot)
    plt.close()


# Attention plots

# Plot text attention weights.
def plot_txt_att(ur: Sequence[str], sr: Sequence[str], attention: torch.Tensor, att_plot: str) -> None:
    attention = attention.cpu().numpy()

    fig, ax = plt.subplots(1, 1)
    im = ax.matshow(attention, cmap="bone")
    ax.set_xticks(ticks=np.arange(len(ur)), labels=ur)
    ax.set_yticks(ticks=np.arange(len(sr)), labels=sr)
    fig.colorbar(im)
    plt.savefig(att_plot)
    plt.close()


# Plot audio/text attention for one example.
def plot_aud_att(
    ur_aud: torch.Tensor,
    sr_txt: Sequence[str],
    sr_aud: torch.Tensor,
    txt_attention: torch.Tensor,
    aud_attention: torch.Tensor,
    txt_att_plot: str,
    aud_att_plot: str,
) -> None:
    ur_aud = ur_aud.cpu().numpy()
    sr_aud = sr_aud.cpu().numpy()
    txt_attention = txt_attention.cpu().numpy()
    aud_attention = aud_attention.cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6), gridspec_kw={"height_ratios": [1, 1]})
    ax1.imshow(ur_aud, origin="lower", aspect="auto")
    ax2.imshow(txt_attention, origin="lower", aspect="auto", cmap="bone")
    ax2.set_yticks(ticks=np.arange(len(sr_txt)), labels=sr_txt)
    plt.tight_layout()
    plt.savefig(txt_att_plot)
    plt.close()

    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    axs[0, 0].axis("off")
    axs[0, 1].imshow(ur_aud, origin="lower", aspect="auto")
    axs[1, 0].imshow(sr_aud, origin="lower", aspect="auto")
    axs[1, 1].imshow(aud_attention, origin="lower", aspect="auto", cmap="bone")
    plt.tight_layout()
    plt.savefig(aud_att_plot)
    plt.close()


# Embedding plots

# Plot extracted audio vowel embeddings.
def plot_aud_embed(embed_store: Mapping[str, Sequence[Any]], embed_plot: str) -> None:
    embed_df = pd.DataFrame(embed_store)
    feature_cols = [col for col in embed_df.columns if col.startswith("mel_")]

    reduced = PCA(n_components=2).fit_transform(embed_df[feature_cols])
    embed_df["pc1"] = reduced[:, 0]
    embed_df["pc2"] = reduced[:, 1]
    vowel_labels = sorted(embed_df["vowel_label"].unique())
    color_codes = pd.Categorical(embed_df["vowel_label"], categories=vowel_labels).codes
    cmap = plt.colormaps.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (_, row) in enumerate(embed_df.iterrows()):
        ax.scatter(
            row["pc1"], 
            row["pc2"], 
            s=5, 
            alpha=0.7, 
            color=cmap(color_codes[i]))
        ax.text(
            row["pc1"],
            row["pc2"],
            row["word_ref"],
            ha="left",
            va="bottom",
            fontsize=5,
        )
    ax.set_xlabel("pc1")
    ax.set_ylabel("pc2")
    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=5, color=cmap(i))
        for i in range(len(vowel_labels))
    ]
    ax.legend(legend_handles, vowel_labels, ncols=2, fontsize=7)
    plt.tight_layout()
    plt.savefig(embed_plot, dpi=300)
    plt.close()


# Plot one static 3D embedding snapshot.
def plot_embed(embed_store: Mapping[str, Any], focus_list: Sequence[str], embed_plot: str) -> None:
    embed_df = pd.DataFrame.from_dict(embed_store, orient="index")
    focus_embed_df = embed_df[embed_df.index.isin(focus_list)]

    embed_new_idx = [
        "m", "n", "ŋ", "p", "t", "k", "b", "d", "g", "f", "s", "θ", "ʃ", "v", "z", "ð", "ʒ", "h",
        "i", "e", "u", "o", "ɪ", "ɛ", "ʊ", "ɔ",
    ]
    focus_embed_new_idx = ["i", "e", "u", "o", "ɪ", "ɛ", "ʊ", "ɔ"]
    embed_df = embed_df.reindex(embed_new_idx)
    focus_embed_df = focus_embed_df.reindex(focus_embed_new_idx)

    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(embed_df)
    reduced_df = pd.DataFrame(data=reduced_data, columns=["pc1", "pc2", "pc3"])
    reduced_df["phoneme"] = embed_df.index

    focus_pca = PCA(n_components=3)
    focus_reduced_data = focus_pca.fit_transform(focus_embed_df)
    focus_reduced_df = pd.DataFrame(data=focus_reduced_data, columns=["pc1", "pc2", "pc3"])
    focus_reduced_df["phoneme"] = focus_embed_df.index

    plt.rcParams.update({"font.size": 5})
    fig = plt.figure(figsize=(8, 3))

    ax2 = fig.add_subplot(131, projection="3d")
    for i in focus_reduced_df.index:
        ax2.scatter(
            xs=focus_reduced_df.loc[i, "pc1"],
            ys=focus_reduced_df.loc[i, "pc2"],
            zs=focus_reduced_df.loc[i, "pc3"],
            s=5,
            color=plt.colormaps.get_cmap("tab20")(i % 18),
            label=focus_reduced_df.loc[i, "phoneme"],
        )
        ax2.text(
            x=focus_reduced_df.loc[i, "pc1"],
            y=focus_reduced_df.loc[i, "pc2"],
            z=focus_reduced_df.loc[i, "pc3"],
            s=focus_reduced_df.loc[i, "phoneme"],
            ha="left",
            va="bottom",
        )
    ax2.set_xlabel("pc1")
    ax2.set_ylabel("pc2")
    ax2.set_zlabel("pc3")
    ax2.set_title("Vowel embedding")
    ax2.legend(loc="center left", bbox_to_anchor=(1.1, 0.5)).remove()
    push_text_free(fig, ax2)

    ax1 = fig.add_subplot(132, projection="3d")
    for i in reduced_df.index:
        ax1.scatter(
            xs=reduced_df.loc[i, "pc1"],
            ys=reduced_df.loc[i, "pc2"],
            zs=reduced_df.loc[i, "pc3"],
            s=5,
            color=plt.colormaps.get_cmap("tab20")(i % 18),
            label=reduced_df.loc[i, "phoneme"],
        )
        ax1.text(
            x=reduced_df.loc[i, "pc1"],
            y=reduced_df.loc[i, "pc2"],
            z=reduced_df.loc[i, "pc3"],
            s=reduced_df.loc[i, "phoneme"],
            ha="left",
            va="bottom",
        )
    ax1.set_xlabel("pc1")
    ax1.set_ylabel("pc2")
    ax1.set_zlabel("pc3")
    ax1.set_title("Phoneme embedding")
    ax1.legend(ncols=2, loc="center left", bbox_to_anchor=(1.3, 0.5))
    push_text_free(fig, ax1)

    plt.savefig(embed_plot, dpi=300)
    plt.close()


# Plot animated 3D embedding trajectories across epochs.
def plot_embed_updated(
    embed_store: Mapping[str, Mapping[str, Any]],
    focus_list: Sequence[str],
    embed_plot: str,
    focus_embed_plot: str,
) -> None:
    dfs = []
    for key, value in embed_store.items():
        df = pd.DataFrame.from_dict(value, orient="index")
        df["phoneme"] = df.index
        df["file_name"] = key
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)

    metadata = combined_df["file_name"].str.extract(
        r"^(?P<language>.+)_(?P<modality>txt|fea|aud)_(?P<directionality>[^_]+)"
        r"(?:_(?P<property>[^_]+))?_(?P<condition>[^_]+)_run(?P<run_num>\d+)"
        r"_epoch(?P<epoch>-?\d+)_(?P<suffix>.+)$"
    )
    metadata["property"] = metadata["property"].fillna("")
    required_columns = ["language", "modality", "directionality", "condition", "run_num", "epoch", "suffix"]
    if metadata[required_columns].isnull().any().any():
        bad_files = combined_df.loc[metadata[required_columns].isnull().any(axis=1), "file_name"].tolist()
        raise RuntimeError(f"Could not parse embedding metadata from file names: {bad_files}")
    metadata["run_num"] = metadata["run_num"].astype(int)
    metadata["epoch"] = metadata["epoch"].astype(int)
    combined_df = pd.concat([combined_df, metadata], axis=1)

    focus_combined_df = combined_df[combined_df["phoneme"].isin(focus_list)].reset_index()

    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(combined_df.loc[:, 0:10])
    reduced_df = pd.DataFrame(data=reduced_data, columns=["pc1", "pc2", "pc3"])

    focus_pca = PCA(n_components=3)
    focus_reduced_data = focus_pca.fit_transform(focus_combined_df.loc[:, 0:10])
    focus_reduced_df = pd.DataFrame(data=focus_reduced_data, columns=["pc1", "pc2", "pc3"])

    combined_df = pd.concat([combined_df, reduced_df], axis=1)
    focus_combined_df = pd.concat([focus_combined_df, focus_reduced_df], axis=1)

    phoneme = pd.CategoricalDtype(
        categories=[
            "m", "n", "ŋ", "p", "t", "k", "b", "d", "g", "f", "s", "θ", "ʃ", "v", "z", "ð", "ʒ", "h",
            "i", "e", "u", "o", "ɪ", "ɛ", "ʊ", "ɔ",
        ],
        ordered=True,
    )
    vowel = pd.CategoricalDtype(categories=["i", "e", "u", "o", "ɪ", "ɛ", "ʊ", "ɔ"], ordered=True)
    combined_df["phoneme"] = combined_df["phoneme"].astype(phoneme)
    focus_combined_df["phoneme"] = focus_combined_df["phoneme"].astype(vowel)

    combined_df = combined_df.sort_values(by=["phoneme", "epoch"], ascending=[True, True])
    focus_combined_df = focus_combined_df.sort_values(by=["phoneme", "epoch"], ascending=[True, True])

    color_palette = [
        "rgb(138, 29, 99)", "rgb(107, 24, 93)", "rgb(76, 21, 80)",
        "rgb(250, 205, 145)", "rgb(246, 173, 119)", "rgb(240, 142, 98)",
        "rgb(216, 80, 83)", "rgb(195, 56, 90)", "rgb(168, 40, 96)",
        "rgb(18, 78, 43)", "rgb(34, 120, 36)", "rgb(115, 152, 5)", "rgb(195, 182, 59)",
        "rgb(140, 193, 186)", "rgb(60, 154, 171)", "rgb(30, 110, 161)", "rgb(38, 62, 144)",
        "rgb(239, 226, 156)",
        "rgb(158,1,66)", "rgb(213,62,79)", "rgb(94,79,162)", "rgb(50,136,189)",
        "rgb(244,109,67)", "rgb(253,174,97)", "rgb(102,194,165)", "rgb(171,221,164)",
    ]
    focus_color_palette = [
        "rgb(158,1,66)", "rgb(213,62,79)", "rgb(94,79,162)", "rgb(50,136,189)",
        "rgb(244,109,67)", "rgb(253,174,97)", "rgb(102,194,165)", "rgb(171,221,164)",
    ]

    fig = px.scatter_3d(
        combined_df,
        x="pc1",
        y="pc2",
        z="pc3",
        color="phoneme",
        color_discrete_sequence=color_palette,
        animation_frame="epoch",
    )
    fig.write_html(embed_plot)
    plt.close()

    fig = px.scatter_3d(
        focus_combined_df,
        x="pc1",
        y="pc2",
        z="pc3",
        color="phoneme",
        color_discrete_sequence=focus_color_palette,
        animation_frame="epoch",
    )
    fig.write_html(focus_embed_plot)
    plt.close()
