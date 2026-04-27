# created 2025/06/04
# A class that handles data recording of multiple runs in dictionaries
# Dictionary data are saved to file using utils in TextTrainer

import os
from typing import Any, Dict, List, Tuple
import hyper_params as hp


class TextRecorder:
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
        # Store the core run metadata used across all recorder outputs.
        self.dataset = dataset
        self.trial_num = trial_num
        self.language = language
        self.lang_name = hp.lang_name
        self.modality = modality
        self.directionality = directionality
        self.property = hp.property
        self.condition = condition
        self.run_num = run_num

        # Store the dataset alphabets used for tensor/string conversion.
        self.ur_alphabet = self.dataset.ur_alphabet
        self.sr_alphabet = self.dataset.sr_alphabet

        # Build the shared filename roots for this run.
        self.result_root = "_".join(part for part in [self.lang_name, self.modality] if part)
        self.run_root = "_".join(
            part for part in [
                self.result_root,
                self.directionality,
                self.property,
                self.condition,
                "run" + str(self.run_num),
            ]
            if part
        )

        # Create the accuracy recorder store.
        self.acc_store = {
            'trial_num': [], 'language': [], 'modality': [],
            'directionality': [], 'property': [], 'condition': [], 
            'run_num': [], 'epoch': [], 'record_type': [],
            'loss': [], 'acc': []
        }

        # Create the prediction recorder store.
        self.pred_store = {
            'trial_num': [], 'language': [], 'modality': [],
            'directionality': [], 'property': [], 'condition': [], 
            'run_num': [], 'epoch': [], 'record_type': [],
            'ur': [], 'sr': [], 'pred_sr': [],
            'v1_error': [], 'v2_error': [],
            'sr_v1': [], 'sr_v2': [],
            'pred_sr_v1': [], 'pred_sr_v2': [],
        }
        if hp.gen_eval:
            self.pred_store.update({
                'v3_error': [],
                'sr_v3': [],
                'pred_sr_v3': [],
            })
        if hp.pred_log != "vowel_only_error":
            self.pred_store.update({
                'o1_error': [], 'o2_error': [],
                'sr_o1': [], 'sr_o2': [],
                'pred_sr_o1': [], 'pred_sr_o2': [],
                'c1_error': [], 'c2_error': [],
                'sr_c1': [], 'sr_c2': [],
                'pred_sr_c1': [], 'pred_sr_c2': [],
            })
            if hp.gen_eval:
                self.pred_store.update({
                    'o3_error': [],
                    'sr_o3': [],
                    'pred_sr_o3': [],
                    'c3_error': [],
                    'sr_c3': [],
                    'pred_sr_c3': [],
                })

        # Build the result files and directories for this run.
        self.acc_file = os.path.join("results", trial_num + "_" + self.lang_name + "_" + modality,
                                     self.result_root + "_acc.csv")
        self.pred_file = os.path.join("results", trial_num + "_" + self.lang_name + "_" + modality,
                                      self.result_root + "_pred.csv")

        self.acc_plot_dir = os.path.join("results", trial_num + "_" + self.lang_name + "_" + modality,
                                         self.result_root + "_acc_plots")
        os.makedirs(self.acc_plot_dir, exist_ok=True)
        self.acc_plot = os.path.join(self.acc_plot_dir, self.run_root + "_acc_plot.png")

        self.model_dir = os.path.join("results", trial_num + "_" + self.lang_name + "_" + modality,
                                      self.result_root + "_model_files",
                                      self.run_root + "_model_files")
        os.makedirs(self.model_dir, exist_ok=True)

        self.att_plot_dir = os.path.join("results", trial_num + "_" + self.lang_name + "_" + modality,
                                         self.result_root + "_att_plots",
                                         self.run_root + "_att_plots")
        os.makedirs(self.att_plot_dir, exist_ok=True)

        self.embed_plot_dir = os.path.join("results", trial_num + "_" + self.lang_name + "_" + modality,
                                           self.result_root + "_embed_plots")
        os.makedirs(self.embed_plot_dir, exist_ok=True)
        self.embed_plot = os.path.join(self.embed_plot_dir, self.run_root + "_embedding.html")
        self.focus_embed_plot = os.path.join(self.embed_plot_dir, self.run_root + "_focus_embedding.html")

    # Add the shared run metadata fields to one recorder store.
    def _append_base_fields(self, store: Dict[str, List[Any]], epoch: int, record_type: str) -> None:
        store['trial_num'].append(self.trial_num)
        store['language'].append(self.lang_name)
        store['modality'].append(self.modality)
        store['directionality'].append(self.directionality)
        store['property'].append(self.property)
        store['condition'].append(self.condition)
        store['run_num'].append(self.run_num)
        store['epoch'].append(epoch)
        store['record_type'].append(record_type)

    # a function that converts one pair of ur, sr, and pred_sr tensor to list
    def tensor2list(
        self,
        ur: Any,
        sr: Any,
        pred_sr: Any,
    ) -> Tuple[List[str], List[str], List[str]]:
        # convert tensor to vector
        ur_vector = ur.cpu().numpy()
        sr_vector = sr.cpu().numpy()
        pred_sr_vector = pred_sr.cpu().numpy()

        # convert vector to word list and string
        ur_list, _ = self.ur_alphabet.vec2word(ur_vector)
        sr_list, _ = self.sr_alphabet.vec2word(sr_vector)
        pred_sr_list, _ = self.sr_alphabet.vec2word(pred_sr_vector)

        return ur_list, sr_list, pred_sr_list

    # a function that converts one pair of ur, sr, and pred_sr tensor to string
    def tensor2string(self, ur: Any, sr: Any, pred_sr: Any) -> Tuple[str, str, str]:
        # convert tensor to vector
        ur_vector = ur.cpu().numpy()
        sr_vector = sr.cpu().numpy()
        pred_sr_vector = pred_sr.cpu().numpy()

        # convert vector to word list and string
        _, ur_string = self.ur_alphabet.vec2word(ur_vector)
        _, sr_string = self.sr_alphabet.vec2word(sr_vector)
        _, pred_sr_string = self.sr_alphabet.vec2word(pred_sr_vector)

        return ur_string, sr_string, pred_sr_string

    # a function that converts one pair of ur, sr, and pred_sr tensor to syllables
    def tensor2syll(self, ur: Any, sr: Any, pred_sr: Any) -> Tuple[List[List[Any]], List[List[Any]], List[List[Any]]]:
        # convert tensor to word list
        ur_list, sr_list, pred_sr_list = self.tensor2list(ur, sr, pred_sr)

        # decompose word list into structured syllables
        ur_sylls = self.language.decompose_stimuli(ur_list)
        sr_sylls = self.language.decompose_stimuli(sr_list)
        pred_sr_sylls = self.language.decompose_stimuli(pred_sr_list)

        return ur_sylls, sr_sylls, pred_sr_sylls

    # a function that records accuracy rates into a dictionary
    # the function is called at each training/evaluation epoch
    def record_acc(self, epoch: int, record_type: str, loss: float, acc: float) -> None:
        # add current accuracy data to the accuracy data storage
        self._append_base_fields(self.acc_store, epoch, record_type)
        self.acc_store['loss'].append(loss)
        self.acc_store['acc'].append(acc)

    # a function that records the prediction information line into a dictionary and the prediction correctness in a list
    # the function is called at each training/evaluation epoch
    def record_pred(
        self,
        epoch: int,
        record_type: str,
        src: List[Any],
        trg: List[Any],
        pred_trg: List[Any],
    ) -> float:

        epoch_correct = []

        # for every batch in an epoch
        for i in range(len(src)):
            # for individual pairs in a batch
            for j in range(hp.batch_size):
                ur = src[i][:, j]
                sr = trg[i][:, j]
                pred_sr = pred_trg[i][:, j]

                # convert the pair
                ur_string, sr_string, pred_sr_string = self.tensor2string(ur, sr, pred_sr)
                ur_sylls, sr_sylls, pred_sr_sylls = self.tensor2syll(ur, sr, pred_sr)

                # prediction log mode:
                # - "vowel_only_error": skip if any consonant mismatch
                # - "consonant_vowel_error": keep consonant and vowel errors
                # - "all_correct_syll": keep all predictions with valid syllable structure

                # record prediction correctness
                if sr_string == pred_sr_string:
                    epoch_correct.append(1)
                    if hp.pred_log != "all_correct_syll":
                        continue
                else:
                    epoch_correct.append(0)

                # Skip this recording if the prediction cannot be parsed into legal syllables.
                if pred_sr_sylls == [False, False, False]:
                    continue

                syll_count = len(sr_sylls)

                sr_o = [s[0] for s in sr_sylls]
                sr_v = [s[1] for s in sr_sylls]
                sr_c = [s[2] for s in sr_sylls]
                pred_o = [s[0] for s in pred_sr_sylls]
                pred_v = [s[1] for s in pred_sr_sylls]
                pred_c = [s[2] for s in pred_sr_sylls]

                # in vowel-only mode, skip if any consonant mismatch
                if hp.pred_log == "vowel_only_error":
                    if any(sr_o[i] != pred_o[i] or sr_c[i] != pred_c[i] for i in range(syll_count)):
                        continue

                # compute per-syllable error flags
                v_errors = [1 if sr_v[i] != pred_v[i] else 0 for i in range(syll_count)]
                if hp.pred_log != "vowel_only_error":
                    o_errors = [1 if sr_o[i] != pred_o[i] else 0 for i in range(syll_count)]
                    c_errors = [1 if sr_c[i] != pred_c[i] else 0 for i in range(syll_count)]

                self._append_base_fields(self.pred_store, epoch, record_type)
                self.pred_store['ur'].append(ur_string)
                self.pred_store['sr'].append(sr_string)
                self.pred_store['pred_sr'].append(pred_sr_string)

                # record consonant errors only when enabled
                if hp.pred_log != "vowel_only_error":
                    self.pred_store['o1_error'].append(o_errors[0])
                    self.pred_store['o2_error'].append(o_errors[1])
                    self.pred_store['sr_o1'].append(sr_o[0])
                    self.pred_store['sr_o2'].append(sr_o[1])
                    self.pred_store['pred_sr_o1'].append(pred_o[0])
                    self.pred_store['pred_sr_o2'].append(pred_o[1])
                    self.pred_store['c1_error'].append(c_errors[0])
                    self.pred_store['c2_error'].append(c_errors[1])
                    self.pred_store['sr_c1'].append(sr_c[0])
                    self.pred_store['sr_c2'].append(sr_c[1])
                    self.pred_store['pred_sr_c1'].append(pred_c[0])
                    self.pred_store['pred_sr_c2'].append(pred_c[1])
                self.pred_store['v1_error'].append(v_errors[0])
                self.pred_store['v2_error'].append(v_errors[1])
                self.pred_store['sr_v1'].append(sr_v[0])
                self.pred_store['sr_v2'].append(sr_v[1])
                self.pred_store['pred_sr_v1'].append(pred_v[0])
                self.pred_store['pred_sr_v2'].append(pred_v[1])

                # record third-syllable fields only for 3-syllable outputs
                if syll_count == 3:
                    if hp.pred_log != "vowel_only_error":
                        self.pred_store['o3_error'].append(o_errors[2])
                        self.pred_store['sr_o3'].append(sr_o[2])
                        self.pred_store['pred_sr_o3'].append(pred_o[2])
                        self.pred_store['c3_error'].append(c_errors[2])
                        self.pred_store['sr_c3'].append(sr_c[2])
                        self.pred_store['pred_sr_c3'].append(pred_c[2])
                    self.pred_store['v3_error'].append(v_errors[2])
                    self.pred_store['sr_v3'].append(sr_v[2])
                    self.pred_store['pred_sr_v3'].append(pred_v[2])
                elif 'v3_error' in self.pred_store:
                    if hp.pred_log != "vowel_only_error":
                        self.pred_store['o3_error'].append("")
                        self.pred_store['sr_o3'].append("")
                        self.pred_store['pred_sr_o3'].append("")
                        self.pred_store['c3_error'].append("")
                        self.pred_store['sr_c3'].append("")
                        self.pred_store['pred_sr_c3'].append("")
                    self.pred_store['v3_error'].append("")
                    self.pred_store['sr_v3'].append("")
                    self.pred_store['pred_sr_v3'].append("")

        if len(epoch_correct) == 0:
            raise RuntimeError(f"No predictions recorded for epoch {epoch} ({record_type})")

        epoch_acc = sum(epoch_correct) / len(epoch_correct)  # calculate epoch accuracy rate
        return epoch_acc
