# created 2025/11/10
# A class that handles data recording of multiple runs in dictionaries
# Dictionary data are saved to file using utils in AudioTrainer

import os
import re
from typing import Any, List, Tuple
import numpy as np
import pandas as pd
import torch
import hyper_params as hp
from ipa_transformation import txt_ipa_to_arpabet
from text_recorder import TextRecorder


class AudioRecorder(TextRecorder):
    def __init__(
        self,
        dataset: Any,
        trial_num: str,
        language: Any,
        modality: str,
        directionality: str,
        condition: str,
        run_num: int,
    ) -> None:
        super().__init__(dataset, trial_num, language, modality, directionality, condition, run_num)

        # Create the audio accuracy store.
        self.acc_store = {
            'trial_num': [], 'language': [], 'modality': [],
            'directionality': [], 'property': [], 'condition': [], 
            'run_num': [], 'epoch': [], 'record_type': [],
            'rec_loss': [], 'pred_loss': [], 'pred_acc': []
        }

        # Create the audio embedding store.
        self.aud_embed_store = {
            "spectrogram_type": [],
            "item_index": [],
            "word_ref": [],
            "vowel_index": [],
            "vowel_label": [],
        }
        for mel_idx in range(dataset.n_mels):
            self.aud_embed_store[f"mel_{mel_idx}"] = []
        self.aud_vowel_relation_store = {
            "spectrogram_type": [],
            "item_index": [],
            "word_ref": [],
            "first_vowel": [],
            "second_vowel": [],
            "vowel_pair": [],
            "euclidean": [],
            "cosine": [],
        }

        # Build the audio embedding result directory.
        self.audio_embed_dir = os.path.join(
            "results",
            trial_num + "_" + self.lang_name + "_" + modality,
            self.result_root + "_aud_embed_plots",
        )
        os.makedirs(self.audio_embed_dir, exist_ok=True)

        # Build the audio embedding files and plots.
        self.aud_embed_file = os.path.join(self.audio_embed_dir, self.run_root + "_aud_embed.csv")
        self.aud_vowel_relation_file = os.path.join(self.audio_embed_dir, self.run_root + "_aud_vowel_relation.csv")
        self.pred_embed_plot = os.path.join(self.audio_embed_dir, self.run_root + "_pred_embed.png")
        self.aud_embed_plot = os.path.join(self.audio_embed_dir, self.run_root + "_aud_embed.html")
        self.aud_vowel_relation_plot = os.path.join(
            self.audio_embed_dir,
            self.run_root + "_aud_vowel_relation.html",
        )

    # a function that records accuracy rates into a dictionary
    # the function is called at each training/evaluation epoch
    def record_acc(
        self,
        epoch: int,
        record_type: str,
        rec_loss: float,
        pred_loss: float,
        pred_acc: float,
    ) -> None:
        # add current accuracy data to the accuracy data storage
        self._append_base_fields(self.acc_store, epoch, record_type)
        self.acc_store['rec_loss'].append(rec_loss)
        self.acc_store['pred_loss'].append(pred_loss)
        self.acc_store['pred_acc'].append(pred_acc)

    # Read the two vowel intervals from the TextGrid phones tier.
    def read_vowel_intervals(
        self,
        textgrid_file: str,
    ) -> List[Tuple[float, float, str]]:
        vowel_labels = list(self.language.focus.keys())
        # TextGrid phone labels are ARPABET and may carry stress digits, while
        # the language inventory is IPA. Build a stressless ARPABET -> IPA map
        # for matching intervals, but keep IPA as the stored vowel label.
        vowel_label_by_arpabet = {
            re.sub(r"\d", "", txt_ipa_to_arpabet(vowel_label)): vowel_label
            for vowel_label in vowel_labels
        }
        current_tier = None
        phones_tier = None
        current_interval = {}
        in_interval = False

        # Read the TextGrid file line by line and scan through its interval tiers until the phones tier is found.
        # Each elif handles a different kind of TextGrid line on the next loop iteration.
        with open(textgrid_file, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()

                if line.startswith("class = ") and '"IntervalTier"' in line:
                    if current_tier is not None and current_tier["name"] == "phones":
                        phones_tier = current_tier
                    current_tier = {"name": "", "intervals": []}
                    current_interval = {}
                    in_interval = False

                # Store the name of the tier currently being scanned.
                elif current_tier is not None and line.startswith("name = "):
                    current_tier["name"] = line.split("=", 1)[1].strip().strip('"')

                # Start collecting one interval inside the current tier block.
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

        # Find the interval tier named "phones".
        if current_tier is not None and current_tier["name"] == "phones":
            phones_tier = current_tier
        if phones_tier is None:
            raise RuntimeError(f"Could not find phones tier in {textgrid_file}")

        # Keep intervals whose ARPABET label matches a focus vowel, ignoring stress numbers.
        # Store the corresponding IPA label so downstream CSVs and plots remain linguistically readable.
        vowel_intervals = []
        for start_time, end_time, textgrid_label in phones_tier["intervals"]:
            arpabet_label = re.sub(r"\d", "", textgrid_label)
            if arpabet_label in vowel_label_by_arpabet:
                vowel_intervals.append(
                    (start_time, end_time, vowel_label_by_arpabet[arpabet_label])
                )

        return vowel_intervals

    # Convert one time interval into mel-frame boundaries.
    def time_to_mel_frame(
        self,
        start_time: float,
        end_time: float,
        max_frames: int,
    ) -> Tuple[int, int]:
        start_frame = int(round(start_time * hp.sample_rate / 256)) + 1
        end_frame = int(round(end_time * hp.sample_rate / 256)) + 1
        start_frame = max(0, min(start_frame, max_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, max_frames))
        return start_frame, end_frame

    # Read the TextGrid intervals and add one row per vowel token to the audio embedding store.
    def record_audio_embedding(
        self,
        embedding_type: str,
        spectrogram: torch.Tensor,
        textgrid_file: str,
        word_ref: str,
        item_index: int,
    ) -> None:
        # Read the vowel intervals from the TextGrid file.
        intervals = self.read_vowel_intervals(textgrid_file)
        
        # Add placeholder rows if the TextGrid does not provide exactly two vowel intervals.
        if len(intervals) != 2:
            for vowel_index in range(2):
                self.aud_embed_store["spectrogram_type"].append(embedding_type)
                self.aud_embed_store["item_index"].append(item_index)
                self.aud_embed_store["word_ref"].append(word_ref)
                self.aud_embed_store["vowel_index"].append(vowel_index)
                self.aud_embed_store["vowel_label"].append("NA")
                for mel_idx in range(self.dataset.n_mels):
                    self.aud_embed_store[f"mel_{mel_idx}"].append("NA")
            return
        
        max_frames = spectrogram.shape[1]
        # Otherwise, convert each vowel interval into one mean mel embedding row.
        for vowel_index, (start_time, end_time, vowel_label) in enumerate(intervals):
            start_frame, end_frame = self.time_to_mel_frame(start_time, end_time, max_frames)
            vowel_slice = spectrogram[:, start_frame:end_frame]
            vowel_embed = vowel_slice.mean(dim=1).cpu().tolist()

            # Append the vowel metadata and mel values to the selected store.
            self.aud_embed_store["spectrogram_type"].append(embedding_type)
            self.aud_embed_store["item_index"].append(item_index)
            self.aud_embed_store["word_ref"].append(word_ref)
            self.aud_embed_store["vowel_index"].append(vowel_index)
            self.aud_embed_store["vowel_label"].append(vowel_label)
            for mel_idx, value in enumerate(vowel_embed):
                self.aud_embed_store[f"mel_{mel_idx}"].append(value)

    # Calculate within-word vowel distances from the recorded audio embedding store.
    def record_vowel_relation(self) -> None:
        embed_df = pd.DataFrame(self.aud_embed_store)

        # Drop placeholder rows and stop early if no real vowel embeddings were recorded.
        embed_df = embed_df[embed_df["vowel_label"] != "NA"].copy()
        if len(embed_df) == 0:
            return

        spectrogram_types = ["source", "target", "pred"]
        feature_cols = [col for col in embed_df.columns if col.startswith("mel_")]

        # Go through all spectrogram types of all evaluated word items and record one relation row per type.
        for item_index in sorted(embed_df["item_index"].unique()):
            for spectrogram_type in spectrogram_types:
                word_df = embed_df[
                    (embed_df["item_index"] == item_index)
                    & (embed_df["spectrogram_type"] == spectrogram_type)
                ].sort_values("vowel_index")
                if len(word_df) < 2:
                    # Add a placeholder row when this spectrogram type does not provide a full vowel pair.
                    self.aud_vowel_relation_store["spectrogram_type"].append(spectrogram_type)
                    self.aud_vowel_relation_store["item_index"].append(item_index)
                    self.aud_vowel_relation_store["word_ref"].append("NA")
                    self.aud_vowel_relation_store["first_vowel"].append("NA")
                    self.aud_vowel_relation_store["second_vowel"].append("NA")
                    self.aud_vowel_relation_store["vowel_pair"].append("NA")
                    self.aud_vowel_relation_store["euclidean"].append("NA")
                    self.aud_vowel_relation_store["cosine"].append("NA")
                    continue

                # Pull out the first and second vowel embeddings for this word-level pair.
                first_vowel = word_df.iloc[0]
                second_vowel = word_df.iloc[1]
                first_embed = first_vowel[feature_cols].astype(float).to_numpy()
                second_embed = second_vowel[feature_cols].astype(float).to_numpy()

                # Measure how similar the two vowels are in the mel-embedding space.
                euclidean = np.linalg.norm(first_embed - second_embed)
                cosine = np.dot(first_embed, second_embed) / (
                    np.linalg.norm(first_embed) * np.linalg.norm(second_embed)
                )

                # Append the metadata and distance measures for this pair to the relation store.
                self.aud_vowel_relation_store["spectrogram_type"].append(spectrogram_type)
                self.aud_vowel_relation_store["item_index"].append(item_index)
                self.aud_vowel_relation_store["word_ref"].append(first_vowel["word_ref"])
                self.aud_vowel_relation_store["first_vowel"].append(first_vowel["vowel_label"])
                self.aud_vowel_relation_store["second_vowel"].append(second_vowel["vowel_label"])
                self.aud_vowel_relation_store["vowel_pair"].append(
                    f"{first_vowel['vowel_label']}_{second_vowel['vowel_label']}"
                )
                self.aud_vowel_relation_store["euclidean"].append(euclidean)
                self.aud_vowel_relation_store["cosine"].append(cosine)
