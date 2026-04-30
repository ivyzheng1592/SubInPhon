import os
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from nooverlap import push_text_free
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import torch


# File I/O

# Save one recorder or embedding store to CSV.
def save_to_file(data_store: Mapping[str, Any], save_file: str) -> None:
    data_df = pd.DataFrame(data_store)

    # Append rows to an existing CSV or create a new CSV with headers.
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

    # Plot train, valid, test, and available gen curves.
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

    # Plot reconstruction loss, prediction loss, and prediction accuracy.
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

    # Plot source spectrogram and text-decoder attention.
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6), gridspec_kw={"height_ratios": [1, 1]})
    ax1.imshow(ur_aud, origin="lower", aspect="auto")
    ax2.imshow(txt_attention, origin="lower", aspect="auto", cmap="bone")
    ax2.set_yticks(ticks=np.arange(len(sr_txt)), labels=sr_txt)
    plt.tight_layout()
    plt.savefig(txt_att_plot)
    plt.close()

    # Plot source spectrogram, target spectrogram, and audio attention.
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    axs[0, 0].axis("off")
    axs[0, 1].imshow(ur_aud, origin="lower", aspect="auto")
    axs[1, 0].imshow(sr_aud, origin="lower", aspect="auto")
    axs[1, 1].imshow(aud_attention, origin="lower", aspect="auto", cmap="bone")
    plt.tight_layout()
    plt.savefig(aud_att_plot)
    plt.close()


# Embedding plots

AUD_EMBED_NEW_IDX = ["i", "e", "u", "o", "ɪ", "ɛ", "ʊ", "ɔ"]


# Plot one static 3D embedding snapshot.
def plot_embed(embed_store: Mapping[str, Any], focus_list: Sequence[str], embed_plot: str) -> None:
    embed_df = pd.DataFrame.from_dict(embed_store, orient="index")
    focus_embed_df = embed_df[embed_df.index.isin(focus_list)]

    embed_new_idx = [
        "m", "n", "ŋ", "p", "t", "k", "b", "d", "g", "f", "s", "θ", "ʃ", "v", "z", "ð", "ʒ", "h",
        "i", "e", "u", "o", "ɪ", "ɛ", "ʊ", "ɔ",
    ]
    focus_embed_new_idx = ["i", "e", "u", "o", "ɪ", "ɛ", "ʊ", "ɔ"]
    # Reindex phonemes and vowels into the display order.
    embed_df = embed_df.reindex(embed_new_idx)
    focus_embed_df = focus_embed_df.reindex(focus_embed_new_idx)

    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(embed_df)
    reduced_df = pd.DataFrame(data=reduced_data, columns=["pc1", "pc2", "pc3"])
    reduced_df["phoneme"] = embed_df.index

    # Reduce the focused vowel embeddings separately.
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

    # Extract run metadata from embedding file names.
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

    # Create a vowel-only dataframe for the focused trajectory plot.
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

    # Sort by phoneme and epoch before building animation frames.
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


# Plot interactive audio vowel embeddings for source, target, and predicted spectrograms.
def plot_aud_embed(
    source_embed_store: Mapping[str, Sequence[Any]],
    target_embed_store: Mapping[str, Sequence[Any]],
    pred_embed_store: Mapping[str, Sequence[Any]],
    embed_plot: str,
) -> None:
    source_df = pd.DataFrame(source_embed_store)
    source_df = source_df[source_df["vowel_label"] != "NA"].copy()
    source_df["spectrogram_type"] = "source"
    target_df = pd.DataFrame(target_embed_store)
    target_df = target_df[target_df["vowel_label"] != "NA"].copy()
    target_df["spectrogram_type"] = "target"
    pred_df = pd.DataFrame(pred_embed_store)
    pred_df = pred_df[pred_df["vowel_label"] != "NA"].copy()
    pred_df["spectrogram_type"] = "pred"

    dfs = [source_df, target_df, pred_df]
    combined_df = pd.concat(dfs, ignore_index=True)
    if len(combined_df) == 0:
        return

    # Assign IPA vowel order for sorting.
    vowel_dtype = pd.CategoricalDtype(categories=AUD_EMBED_NEW_IDX, ordered=True)
    spectrogram_types = ["source", "target", "pred"]
    spectrogram_dtype = pd.CategoricalDtype(categories=spectrogram_types, ordered=True)
    combined_df["vowel_label"] = combined_df["vowel_label"].astype(vowel_dtype)
    combined_df["spectrogram_type"] = combined_df["spectrogram_type"].astype(spectrogram_dtype)
    combined_df = combined_df.sort_values(["spectrogram_type", "vowel_label", "word_ref", "vowel_index"])

    # Reduce all spectrogram types in one shared PCA space.
    feature_cols = [col for col in combined_df.columns if col.startswith("mel_")]
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(combined_df[feature_cols])
    reduced_df = pd.DataFrame(data=reduced_data, columns=["pc1", "pc2", "pc3"])
    reduced_df["word_ref"] = combined_df["word_ref"].to_numpy()
    reduced_df["vowel_index"] = combined_df["vowel_index"].to_numpy()
    reduced_df["vowel_label"] = combined_df["vowel_label"].to_numpy()
    reduced_df["spectrogram_type"] = combined_df["spectrogram_type"].to_numpy()
    reduced_df["token_label"] = reduced_df["word_ref"].astype(str) + reduced_df["vowel_index"].astype(str)

    cmap = plt.colormaps.get_cmap("tab20")
    plotted_vowels = [
        vowel_label
        for vowel_label in AUD_EMBED_NEW_IDX
        if vowel_label in set(reduced_df["vowel_label"])
    ]
    colors = {
        vowel_label: "#{:02x}{:02x}{:02x}".format(
            *[int(channel * 255) for channel in cmap(i % 18)[:3]]
        )
        for i, vowel_label in enumerate(AUD_EMBED_NEW_IDX)
    }
    fig = go.Figure()
    for spectrogram_type in spectrogram_types:
        spectrogram_df = reduced_df[reduced_df["spectrogram_type"] == spectrogram_type]
        for vowel_label in plotted_vowels:
            plot_df = spectrogram_df[spectrogram_df["vowel_label"] == vowel_label]
            if len(plot_df) == 0:
                continue
            fig.add_trace(
                go.Scatter3d(
                    x=plot_df["pc1"],
                    y=plot_df["pc2"],
                    z=plot_df["pc3"],
                    mode="markers+text",
                    text=plot_df["token_label"],
                    textposition="top center",
                    name=vowel_label,
                    legendgroup=vowel_label,
                    marker={"size": 3, "color": colors[vowel_label]},
                    customdata=plot_df[["word_ref", "vowel_index", "vowel_label", "spectrogram_type"]],
                    hovertemplate=(
                        "word_ref=%{customdata[0]}<br>"
                        "vowel_index=%{customdata[1]}<br>"
                        "vowel_label=%{customdata[2]}<br>"
                        "spectrogram_type=%{customdata[3]}<br>"
                        "pc1=%{x}<br>pc2=%{y}<br>pc3=%{z}<extra></extra>"
                    ),
                    visible=spectrogram_type == "source",
                    showlegend=spectrogram_type == "source",
                )
            )

    buttons = []
    for spectrogram_type in spectrogram_types:
        visible = [
            trace.customdata[0][3] == spectrogram_type
            for trace in fig.data
        ]
        showlegend = visible.copy()
        buttons.append(
            {
                "label": spectrogram_type,
                "method": "update",
                "args": [
                    {"visible": visible, "showlegend": showlegend},
                    {
                        "title": f"Spectrogram vowel embedding: {spectrogram_type}",
                        "showlegend": True,
                    },
                ],
            }
        )

    fig.update_layout(
        title="Spectrogram vowel embedding: source",
        scene={
            "xaxis_title": "pc1",
            "yaxis_title": "pc2",
            "zaxis_title": "pc3",
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "buttons": buttons,
                "x": 0,
                "y": 1.12,
            },
            {
                "type": "buttons",
                "direction": "right",
                "buttons": [
                    {
                        "label": "show text",
                        "method": "restyle",
                        "args": [{"mode": "markers+text"}],
                    },
                    {
                        "label": "hide text",
                        "method": "restyle",
                        "args": [{"mode": "markers"}],
                    },
                ],
                "x": 0.35,
                "y": 1.12,
            },
        ],
    )
    fig.write_html(embed_plot)


# Plot within-word distance and similarity between first and second vowel embeddings.
def plot_aud_vowel_relation(
    source_embed_store: Mapping[str, Sequence[Any]],
    target_embed_store: Mapping[str, Sequence[Any]],
    pred_embed_store: Mapping[str, Sequence[Any]],
    embed_plot: str,
) -> None:
    source_df = pd.DataFrame(source_embed_store)
    source_df = source_df[source_df["vowel_label"] != "NA"].copy()
    source_df["spectrogram_type"] = "source"
    target_df = pd.DataFrame(target_embed_store)
    target_df = target_df[target_df["vowel_label"] != "NA"].copy()
    target_df["spectrogram_type"] = "target"
    pred_df = pd.DataFrame(pred_embed_store)
    pred_df = pred_df[pred_df["vowel_label"] != "NA"].copy()
    pred_df["spectrogram_type"] = "pred"

    combined_df = pd.concat([source_df, target_df, pred_df], ignore_index=True)
    if len(combined_df) == 0:
        return

    # Assign IPA vowel order and spectrogram type order.
    vowel_dtype = pd.CategoricalDtype(categories=AUD_EMBED_NEW_IDX, ordered=True)
    spectrogram_types = ["source", "target", "pred"]
    spectrogram_dtype = pd.CategoricalDtype(categories=spectrogram_types, ordered=True)
    combined_df["vowel_label"] = combined_df["vowel_label"].astype(vowel_dtype)
    combined_df["spectrogram_type"] = combined_df["spectrogram_type"].astype(spectrogram_dtype)
    combined_df = combined_df.sort_values(["spectrogram_type", "word_ref", "vowel_index"])

    feature_cols = [col for col in combined_df.columns if col.startswith("mel_")]
    pair_rows = []
    for (spectrogram_type, word_ref), word_df in combined_df.groupby(["spectrogram_type", "word_ref"], observed=True):
        word_df = word_df.sort_values("vowel_index")
        if len(word_df) != 2:
            continue

        # Extract the two vowel embeddings for the current word and spectrogram type.
        first_vowel = word_df.iloc[0]
        second_vowel = word_df.iloc[1]
        first_embed = first_vowel[feature_cols].astype(float).to_numpy()
        second_embed = second_vowel[feature_cols].astype(float).to_numpy()

        # Calculate Euclidean distance and cosine similarity between the two vowel embeddings.
        euclidean = np.linalg.norm(first_embed - second_embed)
        cosine = np.dot(first_embed, second_embed) / (
            np.linalg.norm(first_embed) * np.linalg.norm(second_embed)
        )
        pair_rows.append(
            {
                "spectrogram_type": spectrogram_type,
                "word_ref": word_ref,
                "first_vowel": first_vowel["vowel_label"],
                "second_vowel": second_vowel["vowel_label"],
                "vowel_pair": f"{first_vowel['vowel_label']}_{second_vowel['vowel_label']}",
                "euclidean": euclidean,
                "cosine": cosine,
            }
        )
    pair_df = pd.DataFrame(pair_rows)
    if len(pair_df) == 0:
        return

    cmap = plt.colormaps.get_cmap("tab20")
    vowel_pair_order = [f"{vowel_1}_{vowel_2}" for vowel_1 in AUD_EMBED_NEW_IDX for vowel_2 in AUD_EMBED_NEW_IDX]
    plotted_vowel_pairs = [
        vowel_pair
        for vowel_pair in vowel_pair_order
        if vowel_pair in set(pair_df["vowel_pair"])
    ]
    colors = {
        vowel_pair: "#{:02x}{:02x}{:02x}".format(
            *[int(channel * 255) for channel in cmap(i % cmap.N)[:3]]
        )
        for i, vowel_pair in enumerate(plotted_vowel_pairs)
    }

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=[
            "Euclidean distance",
            "Cosine similarity",
        ],
    )

    # Add the overall relation across all words for each spectrogram type.
    mean_pair_df = pair_df.groupby("spectrogram_type", observed=True)[["euclidean", "cosine"]].mean()
    mean_pair_df = mean_pair_df.reindex(spectrogram_types).reset_index()
    fig.add_trace(
        go.Scatter(
            x=mean_pair_df["spectrogram_type"],
            y=mean_pair_df["euclidean"],
            mode="lines+markers",
            name="overall mean",
            legendgroup="overall mean",
            line={"color": "black", "width": 5},
            marker={"size": 8, "color": "black"},
            hovertemplate="spectrogram_type=%{x}<br>mean euclidean=%{y}<extra></extra>",
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=mean_pair_df["spectrogram_type"],
            y=mean_pair_df["cosine"],
            mode="lines+markers",
            name="overall mean",
            legendgroup="overall mean",
            line={"color": "black", "width": 5},
            marker={"size": 8, "color": "black"},
            hovertemplate="spectrogram_type=%{x}<br>mean cosine=%{y}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    word_trace_indices = []

    # Plot individual words as points grouped by source, target, or predicted vowel pair.
    for vowel_pair in plotted_vowel_pairs:
        plot_df = pair_df[pair_df["vowel_pair"] == vowel_pair].sort_values(["spectrogram_type", "word_ref"])
        vowel_pair_mean_df = plot_df.groupby("spectrogram_type", observed=True)[["euclidean", "cosine"]].mean()
        vowel_pair_mean_df = vowel_pair_mean_df.reindex(spectrogram_types).reset_index()

        fig.add_trace(
            go.Scatter(
                x=plot_df["spectrogram_type"],
                y=plot_df["euclidean"],
                mode="markers+text",
                text=plot_df["word_ref"],
                textposition="top center",
                name=vowel_pair,
                legendgroup=vowel_pair,
                marker={"size": 5, "color": colors[vowel_pair], "opacity": 0.65},
                customdata=plot_df[["word_ref", "vowel_pair", "first_vowel", "second_vowel"]],
                hovertemplate=(
                    "word_ref=%{customdata[0]}<br>"
                    "vowel_pair=%{customdata[1]}<br>"
                    "first_vowel=%{customdata[2]}<br>"
                    "second_vowel=%{customdata[3]}<br>"
                    "spectrogram_type=%{x}<br>"
                    "euclidean=%{y}<extra></extra>"
                ),
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        word_trace_indices.append(len(fig.data) - 1)
        fig.add_trace(
            go.Scatter(
                x=plot_df["spectrogram_type"],
                y=plot_df["cosine"],
                mode="markers+text",
                text=plot_df["word_ref"],
                textposition="top center",
                name=vowel_pair,
                legendgroup=vowel_pair,
                marker={"size": 5, "color": colors[vowel_pair], "opacity": 0.65},
                customdata=plot_df[["word_ref", "vowel_pair", "first_vowel", "second_vowel"]],
                hovertemplate=(
                    "word_ref=%{customdata[0]}<br>"
                    "vowel_pair=%{customdata[1]}<br>"
                    "first_vowel=%{customdata[2]}<br>"
                    "second_vowel=%{customdata[3]}<br>"
                    "spectrogram_type=%{x}<br>"
                    "cosine=%{y}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        word_trace_indices.append(len(fig.data) - 1)
        fig.add_trace(
            go.Scatter(
                x=vowel_pair_mean_df["spectrogram_type"],
                y=vowel_pair_mean_df["euclidean"],
                mode="lines+markers",
                name=f"{vowel_pair} mean",
                legendgroup=vowel_pair,
                line={"color": colors[vowel_pair], "width": 3},
                marker={"size": 7, "color": colors[vowel_pair]},
                hovertemplate=(
                    f"vowel_pair={vowel_pair}<br>"
                    "spectrogram_type=%{x}<br>"
                    "mean euclidean=%{y}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=vowel_pair_mean_df["spectrogram_type"],
                y=vowel_pair_mean_df["cosine"],
                mode="lines+markers",
                name=f"{vowel_pair} mean",
                legendgroup=vowel_pair,
                line={"color": colors[vowel_pair], "width": 3},
                marker={"size": 7, "color": colors[vowel_pair]},
                hovertemplate=(
                    f"vowel_pair={vowel_pair}<br>"
                    "spectrogram_type=%{x}<br>"
                    "mean cosine=%{y}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title="Within-word spectrogram vowel relation",
        legend={"groupclick": "togglegroup"},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "buttons": [
                    {
                        "label": "show text",
                        "method": "restyle",
                        "args": [{"mode": "markers+text"}, word_trace_indices],
                    },
                    {
                        "label": "hide text",
                        "method": "restyle",
                        "args": [{"mode": "markers"}, word_trace_indices],
                    },
                ],
                "x": 0,
                "y": 1.12,
            }
        ],
    )
    fig.update_xaxes(title_text="spectrogram type", row=1, col=1)
    fig.update_yaxes(title_text="distance", row=1, col=1)
    fig.update_xaxes(title_text="spectrogram type", row=1, col=2)
    fig.update_yaxes(title_text="similarity", row=1, col=2)
    fig.write_html(embed_plot)
