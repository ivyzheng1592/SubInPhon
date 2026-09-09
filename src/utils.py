import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from nooverlap import push_text_free
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import torch


EMBED_NEW_IDX = [
    "m", "n", "ŋ", "p", "t", "k", "b", "d", "g", "f", "s", "θ", "ʃ", "v", "z", "ð", "ʒ", "h",
    "i", "e", "u", "o", "ɪ", "ɛ", "ʊ", "ɔ",
]
FOCUS_EMBED_NEW_IDX = ["i", "e", "u", "o", "ɪ", "ɛ", "ʊ", "ɔ"]

# Hand-picked embedding palette assembled from Plotly built-in continuous scales.
# The first 9 colors come from `px.colors.sequential.matter`, the next 9 from
# `px.colors.diverging.delta`, and the final 8 from a reordered subset of
# `px.colors.diverging.Spectral` (the ColorBrewer Spectral palette).
COLOR_PALETTE = [
    "rgb(138, 29, 99)", "rgb(107, 24, 93)", "rgb(76, 21, 80)",
    "rgb(250, 205, 145)", "rgb(246, 173, 119)", "rgb(240, 142, 98)",
    "rgb(216, 80, 83)", "rgb(195, 56, 90)", "rgb(168, 40, 96)",
    "rgb(18, 78, 43)", "rgb(34, 120, 36)", "rgb(115, 152, 5)", "rgb(195, 182, 59)",
    "rgb(140, 193, 186)", "rgb(60, 154, 171)", "rgb(30, 110, 161)", "rgb(38, 62, 144)",
    "rgb(239, 226, 156)",
    "rgb(158,1,66)", "rgb(213,62,79)", "rgb(94,79,162)", "rgb(50,136,189)",
    "rgb(244,109,67)", "rgb(253,174,97)", "rgb(102,194,165)", "rgb(171,221,164)",
]

# Focused vowel palette taken from a reordered 8-color subset of Plotly's
# `px.colors.diverging.Spectral`, which is based on the ColorBrewer Spectral palette.
FOCUS_COLOR_PALETTE = [
    "rgb(158,1,66)", "rgb(213,62,79)", "rgb(94,79,162)", "rgb(50,136,189)",
    "rgb(244,109,67)", "rgb(253,174,97)", "rgb(102,194,165)", "rgb(171,221,164)",
]


def _rgb_to_hex(color: str) -> str:
    channels = color.removeprefix("rgb(").removesuffix(")").split(",")
    return "#{:02x}{:02x}{:02x}".format(*(int(channel.strip()) for channel in channels))


def _build_color_map(labels: Sequence[Any], palette: Sequence[str]) -> Dict[Any, str]:
    return {label: palette[i % len(palette)] for i, label in enumerate(labels)}


def _build_legend_handles(labels: Sequence[str], colors: Mapping[str, str], marker_size: float) -> List[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=colors[label],
            markeredgecolor=colors[label],
            markersize=marker_size,
            alpha=1.0,
            label=label,
        )
        for label in labels
    ]


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

    # Plot train, valid, and test curves.
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex="all", figsize=(6, 6))
    ax1.plot(train_data["epoch"], train_data["loss"], label="train")
    ax1.plot(valid_data["epoch"], valid_data["loss"], label="valid")
    ax1.plot(test_data["epoch"], test_data["loss"], label="test")
    ax1.legend()
    ax1.set_title("Loss")
    ax1.set_ylim(-0.1, 3.1)

    ax2.plot(train_data["epoch"], train_data["acc"], label="train")
    ax2.plot(valid_data["epoch"], valid_data["acc"], label="valid")
    ax2.plot(test_data["epoch"], test_data["acc"], label="test")
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

    # Plot reconstruction loss, prediction loss, and prediction accuracy.
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex="all", figsize=(6, 8))
    ax1.plot(train_data["epoch"], train_data["rec_loss"], label="train")
    ax1.plot(valid_data["epoch"], valid_data["rec_loss"], label="valid")
    ax1.plot(test_data["epoch"], test_data["rec_loss"], label="test")
    ax1.legend()
    ax1.set_title("Reconstruction Loss")
    ax1.set_ylim(-0.5, 20.5)

    ax2.plot(train_data["epoch"], train_data["pred_loss"], label="train")
    ax2.plot(valid_data["epoch"], valid_data["pred_loss"], label="valid")
    ax2.plot(test_data["epoch"], test_data["pred_loss"], label="test")
    ax2.legend()
    ax2.set_title("Prediction Loss")
    ax2.set_ylim(-0.1, 3.1)

    ax3.plot(train_data["epoch"], train_data["pred_acc"], label="train")
    ax3.plot(valid_data["epoch"], valid_data["pred_acc"], label="valid")
    ax3.plot(test_data["epoch"], test_data["pred_acc"], label="test")
    ax3.legend()
    ax3.set_title("Prediction Acc")
    ax3.set_ylim(-0.05, 1.05)

    plt.savefig(acc_plot)
    plt.close()


# Attention plots

# Plot text attention weights.
def plot_txt_att(ur: Sequence[str], sr: Sequence[str], attention: torch.Tensor, att_plot: str) -> None:
    attention = attention.cpu().numpy()

    plt.rcParams.update({"font.size": 5})
    fig, ax = plt.subplots(1, 1, figsize=(3, 2.25))
    im = ax.matshow(attention, cmap="bone")
    ax.set_xticks(ticks=np.arange(len(ur)), labels=ur)
    ax.set_yticks(ticks=np.arange(len(sr)), labels=sr)
    ax.tick_params(axis="both", labelsize=5)
    colorbar = fig.colorbar(im)
    colorbar.ax.tick_params(labelsize=5)
    plt.savefig(att_plot, dpi=300)
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
    plt.savefig(txt_att_plot, dpi=300)
    plt.close()

    # Plot source spectrogram, target spectrogram, and audio attention.
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    axs[0, 0].axis("off")
    axs[0, 1].imshow(ur_aud, origin="lower", aspect="auto")
    axs[1, 0].imshow(sr_aud, origin="lower", aspect="auto")
    axs[1, 1].imshow(aud_attention, origin="lower", aspect="auto", cmap="bone")
    plt.tight_layout()
    plt.savefig(aud_att_plot, dpi=300)
    plt.close()


# Embedding plots

# Plot one static 3D embedding snapshot.
def plot_embed(
    embed_store: Mapping[str, Any],
    focus_list: Sequence[str],
    embed_plot: str,
    plot_option: str = "focus",
) -> None:
    embed_df = pd.DataFrame.from_dict(embed_store, orient="index")
    focus_embed_df = embed_df[embed_df.index.isin(focus_list)].reindex(FOCUS_EMBED_NEW_IDX).dropna(how="all")
    all_embed_df = embed_df.reindex(EMBED_NEW_IDX).dropna(how="all")

    focus_pca = PCA(n_components=3)
    focus_reduced_data = focus_pca.fit_transform(focus_embed_df)
    focus_reduced_df = pd.DataFrame(data=focus_reduced_data, columns=["pc1", "pc2", "pc3"])
    focus_reduced_df["phoneme"] = focus_embed_df.index
    focus_embed_colors = {
        phoneme: _rgb_to_hex(color)
        for phoneme, color in _build_color_map(FOCUS_EMBED_NEW_IDX, FOCUS_COLOR_PALETTE).items()
    }

    plt.rcParams.update({"font.size": 5})

    if plot_option == "all":
        all_pca = PCA(n_components=3)
        all_reduced_data = all_pca.fit_transform(all_embed_df)
        all_reduced_df = pd.DataFrame(data=all_reduced_data, columns=["pc1", "pc2", "pc3"])
        all_reduced_df["phoneme"] = all_embed_df.index
        all_embed_colors = {
            phoneme: _rgb_to_hex(color)
            for phoneme, color in _build_color_map(EMBED_NEW_IDX, COLOR_PALETTE).items()
        }

        fig = plt.figure(figsize=(6, 2.5))
        all_ax = fig.add_subplot(121, projection="3d")
        focus_ax = fig.add_subplot(122, projection="3d")
        all_ax.set_position([0.00, 0.23, 0.35, 0.75])
        focus_ax.set_position([0.42, 0.23, 0.35, 0.75])
        fig.text(0.175, 0.03, "(a) Phoneme embedding", ha="center")
        fig.text(0.595, 0.03, "(b) Vowel embedding", ha="center")

        for i in focus_reduced_df.index:
            focus_ax.scatter(
                xs=focus_reduced_df.loc[i, "pc1"],
                ys=focus_reduced_df.loc[i, "pc2"],
                zs=focus_reduced_df.loc[i, "pc3"],
                s=5,
                color=focus_embed_colors[focus_reduced_df.loc[i, "phoneme"]],
                label=focus_reduced_df.loc[i, "phoneme"],
            )
            focus_ax.text(
                x=focus_reduced_df.loc[i, "pc1"],
                y=focus_reduced_df.loc[i, "pc2"],
                z=focus_reduced_df.loc[i, "pc3"],
                s=focus_reduced_df.loc[i, "phoneme"],
                ha="left",
                va="bottom",
            )
        focus_ax.set_xlabel("pc1")
        focus_ax.set_ylabel("pc2")
        focus_ax.set_zlabel("pc3")
        push_text_free(fig, focus_ax)

        for i in all_reduced_df.index:
            all_ax.scatter(
                xs=all_reduced_df.loc[i, "pc1"],
                ys=all_reduced_df.loc[i, "pc2"],
                zs=all_reduced_df.loc[i, "pc3"],
                s=5,
                color=all_embed_colors[all_reduced_df.loc[i, "phoneme"]],
                label=all_reduced_df.loc[i, "phoneme"],
            )
            all_ax.text(
                x=all_reduced_df.loc[i, "pc1"],
                y=all_reduced_df.loc[i, "pc2"],
                z=all_reduced_df.loc[i, "pc3"],
                s=all_reduced_df.loc[i, "phoneme"],
                ha="left",
                va="bottom",
            )
        all_ax.set_xlabel("pc1")
        all_ax.set_ylabel("pc2")
        all_ax.set_zlabel("pc3")
        push_text_free(fig, all_ax)

        legend_handles = _build_legend_handles(EMBED_NEW_IDX, all_embed_colors, marker_size=3)
        fig.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(0.84, 0.5),
            ncol=2,
            frameon=False,
            fontsize=5,
            handletextpad=0.2,
            columnspacing=0.5,
        )
    else:
        fig = plt.figure(figsize=(3, 2.5))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_position([0.0, 0.25, 0.7, 0.7])
        for i in focus_reduced_df.index:
            ax.scatter(
                xs=focus_reduced_df.loc[i, "pc1"],
                ys=focus_reduced_df.loc[i, "pc2"],
                zs=focus_reduced_df.loc[i, "pc3"],
                s=5,
                color=focus_embed_colors[focus_reduced_df.loc[i, "phoneme"]],
                label=focus_reduced_df.loc[i, "phoneme"],
            )
            ax.text(
                x=focus_reduced_df.loc[i, "pc1"],
                y=focus_reduced_df.loc[i, "pc2"],
                z=focus_reduced_df.loc[i, "pc3"],
                s=focus_reduced_df.loc[i, "phoneme"],
                ha="left",
                va="bottom",
            )
        ax.set_xlabel("pc1")
        ax.set_ylabel("pc2")
        ax.set_zlabel("pc3")
        legend_handles = _build_legend_handles(FOCUS_EMBED_NEW_IDX, focus_embed_colors, marker_size=3)
        fig.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(0.85, 0.6),
            ncol=1,
            frameon=False,
            fontsize=5,
            handletextpad=0.2,
            columnspacing=0.5,
        )
        push_text_free(fig, ax)

    plt.savefig(embed_plot, dpi=300)
    plt.close()


# Plot animated 3D embedding trajectories across epochs.
def plot_embed_updated(
    embed_store: Mapping[str, Mapping[str, Any]],
    focus_list: Sequence[str],
    embed_plot: str,
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
        r"^(?P<language>.+?)"
        r"(?:_(?P<property>[^_]+))?_(?P<modality>txt|fea|aud)_(?P<directionality>[^_]+)"
        r"_(?P<condition>[^_]+)_run(?P<run_num>\d+)"
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

    phoneme = pd.CategoricalDtype(categories=EMBED_NEW_IDX, ordered=True)
    vowel = pd.CategoricalDtype(categories=FOCUS_EMBED_NEW_IDX, ordered=True)
    combined_df["phoneme"] = combined_df["phoneme"].astype(phoneme)
    focus_combined_df["phoneme"] = focus_combined_df["phoneme"].astype(vowel)

    # Sort by phoneme and epoch before building animation frames.
    combined_df = combined_df.sort_values(by=["phoneme", "epoch"], ascending=[True, True])
    focus_combined_df = focus_combined_df.sort_values(by=["phoneme", "epoch"], ascending=[True, True])

    phoneme_labels = EMBED_NEW_IDX
    vowel_labels = FOCUS_EMBED_NEW_IDX
    phoneme_colors = _build_color_map(phoneme_labels, COLOR_PALETTE)
    vowel_colors = _build_color_map(vowel_labels, FOCUS_COLOR_PALETTE)
    epochs = sorted(combined_df["epoch"].unique())

    # Build one 3D scatter trace from the selected dataframe rows.
    def build_trace(
        plot_df: pd.DataFrame,
        label: str,
        color: str,
        visible: Optional[bool] = None,
        showlegend: Optional[bool] = None,
    ) -> go.Scatter3d:
        trace_kwargs = {
            "x": plot_df["pc1"],
            "y": plot_df["pc2"],
            "z": plot_df["pc3"],
            "mode": "markers",
            "name": label,
            "legendgroup": label,
            "marker": {"size": 4, "color": color},
            "text": plot_df["phoneme"].astype(str),
            "hovertemplate": (
                "phoneme=%{text}<br>"
                "pc1=%{x}<br>pc2=%{y}<br>pc3=%{z}<extra></extra>"
            ),
        }
        if visible is not None:
            trace_kwargs["visible"] = visible
        if showlegend is not None:
            trace_kwargs["showlegend"] = showlegend

        return go.Scatter3d(
            **trace_kwargs,
        )

    initial_epoch = epochs[0]
    initial_phoneme_df = combined_df[combined_df["epoch"] == initial_epoch]
    initial_vowel_df = focus_combined_df[focus_combined_df["epoch"] == initial_epoch]

    traces = []
    for phoneme_label in phoneme_labels:
        plot_df = initial_phoneme_df[initial_phoneme_df["phoneme"] == phoneme_label]
        traces.append(
            build_trace(
                plot_df,
                phoneme_label,
                phoneme_colors[phoneme_label],
                visible=True,
                showlegend=True,
            )
        )
    for vowel_label in vowel_labels:
        plot_df = initial_vowel_df[initial_vowel_df["phoneme"] == vowel_label]
        traces.append(
            build_trace(
                plot_df,
                vowel_label,
                vowel_colors[vowel_label],
                visible=False,
                showlegend=False,
            )
        )

    phoneme_trace_count = len(phoneme_labels)
    vowel_trace_count = len(vowel_labels)

    # Build one animation frame per epoch with phoneme traces followed by vowel traces.
    frames = []
    for epoch in epochs:
        epoch_phoneme_df = combined_df[combined_df["epoch"] == epoch]
        epoch_vowel_df = focus_combined_df[focus_combined_df["epoch"] == epoch]
        frame_traces = []
        for phoneme_label in phoneme_labels:
            plot_df = epoch_phoneme_df[epoch_phoneme_df["phoneme"] == phoneme_label]
            frame_traces.append(build_trace(plot_df, phoneme_label, phoneme_colors[phoneme_label]))
        for vowel_label in vowel_labels:
            plot_df = epoch_vowel_df[epoch_vowel_df["phoneme"] == vowel_label]
            frame_traces.append(build_trace(plot_df, vowel_label, vowel_colors[vowel_label]))
        frames.append(
            go.Frame(
                name=str(epoch),
                data=frame_traces,
                traces=list(range(phoneme_trace_count + vowel_trace_count)),
            )
        )

    visibility_phoneme = [True] * phoneme_trace_count + [False] * vowel_trace_count
    visibility_vowel = [False] * phoneme_trace_count + [True] * vowel_trace_count

    slider_steps = [
        {
            "label": str(epoch),
            "method": "animate",
            "args": [
                [str(epoch)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        }
        for epoch in epochs
    ]

    fig = go.Figure(data=traces, frames=frames)
    fig.update_layout(
        title="Phoneme embedding",
        scene={
            "xaxis_title": "pc1",
            "yaxis_title": "pc2",
            "zaxis_title": "pc3",
        },
        legend={"itemsizing": "constant"},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "buttons": [
                    {
                        "label": "phoneme",
                        "method": "update",
                        "args": [
                            {"visible": visibility_phoneme, "showlegend": visibility_phoneme},
                            {"title": "Phoneme embedding"},
                        ],
                    },
                    {
                        "label": "vowel",
                        "method": "update",
                        "args": [
                            {"visible": visibility_vowel, "showlegend": visibility_vowel},
                            {"title": "Vowel embedding"},
                        ],
                    },
                    {
                        "label": "play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": 500, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
                "x": 0,
                "y": 1.15,
            },
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "epoch="},
                "pad": {"t": 40},
                "steps": slider_steps,
            },
        ],
    )
    fig.write_html(embed_plot)


# Plot one static 3D audio vowel embedding snapshot for predicted spectrograms.
def plot_aud_embed(
    aud_embed_store: Mapping[str, Sequence[Any]],
    embed_plot: str,
    embed_plot_pc2: Optional[str] = None,
) -> int:
    combined_df = pd.DataFrame(aud_embed_store)

    # Remove any item that has a placeholder or mismatched vowel row so only complete source/target/pred triplets remain.
    invalid_item_indices = combined_df.loc[combined_df["vowel_label"].isin([None, False]), "item_index"].unique()
    combined_df = combined_df[~combined_df["item_index"].isin(invalid_item_indices)].copy()
    if len(combined_df) == 0:
        return 0

    # Keep only predicted spectrogram embeddings for the static plot.
    pred_df = combined_df[combined_df["spectrogram_type"] == "pred"].copy()
    if len(pred_df) == 0:
        return 0

    # Assign IPA vowel order for plotting and labeling.
    vowel_dtype = pd.CategoricalDtype(categories=FOCUS_EMBED_NEW_IDX, ordered=True)
    pred_df["vowel_label"] = pred_df["vowel_label"].astype(vowel_dtype)
    pred_df = pred_df.sort_values(["vowel_label", "word_ref", "vowel_index"])

    # Reduce predicted spectrogram embeddings in one shared PCA space.
    feature_cols = [col for col in combined_df.columns if col.startswith("mel_")]
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(pred_df[feature_cols])
    reduced_df = pd.DataFrame(data=reduced_data, columns=["pc1", "pc2", "pc3"])
    reduced_df["word_ref"] = pred_df["word_ref"].to_numpy()
    reduced_df["vowel_index"] = pred_df["vowel_index"].to_numpy()
    reduced_df["vowel_label"] = pred_df["vowel_label"].to_numpy()

    colors = {
        vowel_label: _rgb_to_hex(color)
        for vowel_label, color in _build_color_map(FOCUS_EMBED_NEW_IDX, FOCUS_COLOR_PALETTE).items()
    }

    plt.rcParams.update({"font.size": 5})
    fig = plt.figure(figsize=(3, 2.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([0.0, 0.25, 0.7, 0.7])
    for vowel_label in FOCUS_EMBED_NEW_IDX:
        plot_df = reduced_df[reduced_df["vowel_label"] == vowel_label]
        if len(plot_df) == 0:
            continue
        ax.scatter(
            xs=plot_df["pc1"],
            ys=plot_df["pc2"],
            zs=plot_df["pc3"],
            s=0.1,
            color=colors[vowel_label],
            alpha=0.5,
            label=vowel_label,
        )
    ax.set_xlabel("pc1")
    ax.set_ylabel("pc2")
    ax.set_zlabel("pc3")
    legend_handles = _build_legend_handles(FOCUS_EMBED_NEW_IDX, colors, marker_size=3)
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.85, 0.6),
        ncol=1,
        frameon=False,
        fontsize=5,
        handletextpad=0.2,
        columnspacing=0.5,
    )
    push_text_free(fig, ax)

    plt.savefig(embed_plot, dpi=300)
    if embed_plot_pc2 is not None:
        # View from the pc1 side so pc2 reads horizontally in the second static plot.
        ax.view_init(elev=30, azim=0)
        plt.savefig(embed_plot_pc2, dpi=300)
    plt.close()
    return int(pred_df["item_index"].nunique())


# Plot interactive audio vowel embeddings for source, target, and predicted spectrograms.
def plot_aud_embed_updated(
    aud_embed_store: Mapping[str, Sequence[Any]],
    embed_plot: str,
) -> int:
    combined_df = pd.DataFrame(aud_embed_store)
    spectrogram_types = ["source", "target", "pred"]

    # Remove any item that has a placeholder or mismatched vowel row so only complete source/target/pred triplets remain.
    invalid_item_indices = combined_df.loc[combined_df["vowel_label"].isin([None, False]), "item_index"].unique()
    combined_df = combined_df[~combined_df["item_index"].isin(invalid_item_indices)].copy()
    if len(combined_df) == 0:
        return 0

    # Assign IPA vowel order for sorting.
    vowel_dtype = pd.CategoricalDtype(categories=FOCUS_EMBED_NEW_IDX, ordered=True)
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

    plotted_vowels = [
        vowel_label
        for vowel_label in FOCUS_EMBED_NEW_IDX
        if vowel_label in set(reduced_df["vowel_label"])
    ]
    colors = {
        vowel_label: _rgb_to_hex(color)
        for vowel_label, color in _build_color_map(FOCUS_EMBED_NEW_IDX, FOCUS_COLOR_PALETTE).items()
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
                    mode="markers",
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
        ],
    )
    fig.write_html(embed_plot)
    return int(combined_df["item_index"].nunique())


# Plot within-word distance and similarity between first and second vowel embeddings.
def plot_aud_vowel_distance(
    vowel_distance_store: Mapping[str, Sequence[Any]],
    embed_plot: str,
) -> int:
    spectrogram_types = ["source", "target", "pred"]
    spectrogram_dtype = pd.CategoricalDtype(categories=spectrogram_types, ordered=True)

    # Convert the recorded vowel-distance store into a dataframe for plotting.
    pair_df = pd.DataFrame(vowel_distance_store)
    if len(pair_df) == 0:
        return 0

    # Keep source, target, and predicted rows in a fixed display order.
    pair_df["spectrogram_type"] = pair_df["spectrogram_type"].astype(spectrogram_dtype)
    pair_df = pair_df.sort_values(["item_index", "spectrogram_type"])

    # Build one combined source/target/pred vowel-pair label per item so the three rows share a color group.
    combined_vowel_pair_by_item = {}
    for item_index in sorted(pair_df["item_index"].unique()):
        item_df = pair_df[pair_df["item_index"] == item_index].sort_values("spectrogram_type")
        combined_vowel_pair_by_item[item_index] = ", ".join(item_df["vowel_pair"].tolist())
    pair_df["combined_vowel_pair"] = pair_df["item_index"].map(combined_vowel_pair_by_item)

    # Use the combined source/target/pred vowel-pair label as the plot color and legend group.
    plotted_combined_vowel_pairs = sorted(pair_df["combined_vowel_pair"].unique())
    colors = {
        combined_vowel_pair: _rgb_to_hex(color)
        for combined_vowel_pair, color in _build_color_map(plotted_combined_vowel_pairs, COLOR_PALETTE).items()
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

    # Plot individual words as points grouped by the combined item-level vowel-pair label.
    for combined_vowel_pair in plotted_combined_vowel_pairs:
        plot_df = pair_df[pair_df["combined_vowel_pair"] == combined_vowel_pair].sort_values(
            ["spectrogram_type", "word_ref"]
        )
        combined_vowel_pair_mean_df = plot_df.groupby("spectrogram_type", observed=True)[["euclidean", "cosine"]].mean()
        combined_vowel_pair_mean_df = combined_vowel_pair_mean_df.reindex(spectrogram_types).reset_index()

        fig.add_trace(
            go.Scatter(
                x=plot_df["spectrogram_type"],
                y=plot_df["euclidean"],
                mode="markers",
                name=combined_vowel_pair,
                legendgroup=combined_vowel_pair,
                marker={"size": 5, "color": colors[combined_vowel_pair], "opacity": 0.65},
                customdata=plot_df[["word_ref", "combined_vowel_pair", "vowel_pair", "first_vowel", "second_vowel"]],
                hovertemplate=(
                    "word_ref=%{customdata[0]}<br>"
                    "combined_vowel_pair=%{customdata[1]}<br>"
                    "vowel_pair=%{customdata[2]}<br>"
                    "first_vowel=%{customdata[3]}<br>"
                    "second_vowel=%{customdata[4]}<br>"
                    "spectrogram_type=%{x}<br>"
                    "euclidean=%{y}<extra></extra>"
                ),
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df["spectrogram_type"],
                y=plot_df["cosine"],
                mode="markers",
                name=combined_vowel_pair,
                legendgroup=combined_vowel_pair,
                marker={"size": 5, "color": colors[combined_vowel_pair], "opacity": 0.65},
                customdata=plot_df[["word_ref", "combined_vowel_pair", "vowel_pair", "first_vowel", "second_vowel"]],
                hovertemplate=(
                    "word_ref=%{customdata[0]}<br>"
                    "combined_vowel_pair=%{customdata[1]}<br>"
                    "vowel_pair=%{customdata[2]}<br>"
                    "first_vowel=%{customdata[3]}<br>"
                    "second_vowel=%{customdata[4]}<br>"
                    "spectrogram_type=%{x}<br>"
                    "cosine=%{y}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=combined_vowel_pair_mean_df["spectrogram_type"],
                y=combined_vowel_pair_mean_df["euclidean"],
                mode="lines+markers",
                name=f"{combined_vowel_pair} mean",
                legendgroup=combined_vowel_pair,
                line={"color": colors[combined_vowel_pair], "width": 3},
                marker={"size": 7, "color": colors[combined_vowel_pair]},
                hovertemplate=(
                    f"combined_vowel_pair={combined_vowel_pair}<br>"
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
                x=combined_vowel_pair_mean_df["spectrogram_type"],
                y=combined_vowel_pair_mean_df["cosine"],
                mode="lines+markers",
                name=f"{combined_vowel_pair} mean",
                legendgroup=combined_vowel_pair,
                line={"color": colors[combined_vowel_pair], "width": 3},
                marker={"size": 7, "color": colors[combined_vowel_pair]},
                hovertemplate=(
                    f"combined_vowel_pair={combined_vowel_pair}<br>"
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
    )
    fig.update_xaxes(
        title_text="spectrogram type",
        categoryorder="array",
        categoryarray=spectrogram_types,
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="distance", row=1, col=1)
    fig.update_xaxes(
        title_text="spectrogram type",
        categoryorder="array",
        categoryarray=spectrogram_types,
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="similarity", row=1, col=2)
    fig.write_html(embed_plot)
    return int(pair_df["item_index"].nunique())
