# created 2025/11/10
# A class that handles data recording of multiple runs in dictionaries
# Dictionary data are saved to file using utils in AudioTrainer

import os
from typing import Any
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

        # result storages
        self.acc_store = {
            'trial_num': [], 'language': [], 'modality': [],
            'directionality': [], 'condition': [], 'run_num': [], 'epoch': [], 'record_type': [],
            'rec_loss': [], 'pred_loss': [], 'pred_acc': []
        }

        self.audio_embed_dir = os.path.join(
            "Results",
            trial_num + "_" + self.lang_name + "_" + modality,
            self.lang_name + "_" + modality + "_audio_embed_plots",
        )
        os.makedirs(self.audio_embed_dir, exist_ok=True)
        base_name = (
            self.lang_name + "_" + modality + "_" + self.directionality + "_" +
            self.condition + "_run" + str(self.run_num)
        )
        self.source_audio_embed_file = os.path.join(self.audio_embed_dir, base_name + "_source_audio_embedding.csv")
        self.target_audio_embed_file = os.path.join(self.audio_embed_dir, base_name + "_target_audio_embedding.csv")
        self.pred_audio_embed_file = os.path.join(self.audio_embed_dir, base_name + "_predicted_audio_embedding.csv")
        self.source_audio_embed_plot = os.path.join(self.audio_embed_dir, base_name + "_source_audio_embedding.png")
        self.target_audio_embed_plot = os.path.join(self.audio_embed_dir, base_name + "_target_audio_embedding.png")
        self.pred_audio_embed_plot = os.path.join(self.audio_embed_dir, base_name + "_predicted_audio_embedding.png")


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
