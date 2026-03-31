# created 2025/06/04
# A class that handles data recording of multiple runs in dictionaries
# Dictionary data are saved to file using utils in TextRun

import os
import hyper_params as hp


class TextRecorder:
    def __init__(self, dataset, trial_num, language, modality, condition, run_num):
        self.dataset = dataset
        self.trial_num = trial_num
        self.language = language
        self.lang_name = language.lang_name
        self.modality = modality
        self.condition = condition
        self.run_num = run_num
        self.pred_log = hp.pred_log
        if self.pred_log not in ["vowel_only_error", "consonant_vowel_error", "all_correct_syll"]:
            raise RuntimeError(f"Invalid pred_log: {self.pred_log}")

        self.ur_alphabet = self.dataset.ur_alphabet
        self.sr_alphabet = self.dataset.sr_alphabet

        # result storages
        self.acc_store = {
            'trial_num': [], 'language': [], 'modality': [],
            'condition': [], 'run_num': [], 'epoch': [], 'record_type': [],
            'loss': [], 'acc': []
        }

        self.pred_store = {
            'trial_num': [], 'language': [], 'modality': [],
            'condition': [], 'run_num': [], 'epoch': [], 'record_type': [],
            'ur': [], 'sr': [], 'pred_sr': [],
            'v1_error': [], 'v2_error': [],
            'sr_v1': [], 'sr_v2': [], 'pred_sr_v1': [], 'pred_sr_v2': []
        }
        if self.pred_log != "vowel_only_error":
            self.pred_store.update({
                'o1_error': [], 'o2_error': [],
                'sr_o1': [], 'sr_o2': [], 'pred_sr_o1': [], 'pred_sr_o2': [],
                'c1_error': [], 'c2_error': [],
                'sr_c1': [], 'sr_c2': [], 'pred_sr_c1': [], 'pred_sr_c2': [],
            })

        # results files and directories
        self.acc_file = os.path.join("Results", trial_num + "_" + self.lang_name + "_" + modality,
                                     self.lang_name + "_" + modality + "_acc.csv")
        self.pred_file = os.path.join("Results", trial_num + "_" + self.lang_name + "_" + modality,
                                      self.lang_name + "_" + modality + "_pred.csv")

        self.acc_plot_dir = os.path.join("Results", trial_num + "_" + self.lang_name + "_" + modality,
                                         self.lang_name + "_" + modality + "_acc_plots")
        os.makedirs(self.acc_plot_dir, exist_ok=True)
        self.acc_plot = os.path.join(self.acc_plot_dir,
                                     self.lang_name + "_" + modality + "_" + self.condition +
                                     "_run" + str(self.run_num) + "_acc_plot.png")

        self.model_dir = os.path.join("Results", trial_num + "_" + self.lang_name + "_" + modality,
                                      self.lang_name + "_" + modality + "_model_files",
                                      self.lang_name + "_" + modality + "_" + self.condition +
                                      "_run" + str(self.run_num) + "_model_files")
        os.makedirs(self.model_dir, exist_ok=True)

        self.att_plot_dir = os.path.join("Results", trial_num + "_" + self.lang_name + "_" + modality,
                                         self.lang_name + "_" + modality + "_att_plots",
                                         self.lang_name + "_" + modality + "_" + self.condition +
                                         "_run" + str(self.run_num) + "_att_plots")
        os.makedirs(self.att_plot_dir, exist_ok=True)

        self.embed_plot_dir = os.path.join("Results", trial_num + "_" + self.lang_name + "_" + modality,
                                           self.lang_name + "_" + modality + "_embed_plots")
        os.makedirs(self.embed_plot_dir, exist_ok=True)
        self.embed_plot = os.path.join(self.embed_plot_dir,
                                       self.lang_name + "_" + modality + "_" + self.condition +
                                       "_run" + str(self.run_num) + "_embedding.html")
        self.focus_embed_plot = os.path.join(self.embed_plot_dir,
                                             self.lang_name + "_" + modality + "_" + self.condition +
                                             "_run" + str(self.run_num) + "_focus_embedding.html")

    def _append_base_fields(self, store, epoch, record_type):
        store['trial_num'].append(self.trial_num)
        store['language'].append(self.lang_name)
        store['modality'].append(self.modality)
        store['condition'].append(self.condition)
        store['run_num'].append(self.run_num)
        store['epoch'].append(epoch)
        store['record_type'].append(record_type)

    # a function that converts one pair of ur, sr, and pred_sr tensor to list
    def tensor2list(self, ur, sr, pred_sr):
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
    def tensor2string(self, ur, sr, pred_sr):
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
    def tensor2syll(self, ur, sr, pred_sr):
        # convert tensor to word list
        ur_list, sr_list, pred_sr_list = self.tensor2list(ur, sr, pred_sr)

        # decompose word list into structured syllables
        ur_sylls = self.language.decompose_stimuli(ur_list)
        sr_sylls = self.language.decompose_stimuli(sr_list)
        pred_sr_sylls = self.language.decompose_stimuli(pred_sr_list)

        return ur_sylls, sr_sylls, pred_sr_sylls

    # a function that records accuracy rates into a dictionary
    # the function is called at each training/evaluation epoch
    def record_acc(self, epoch, record_type, loss, acc):
        # add current accuracy data to the accuracy data storage
        self._append_base_fields(self.acc_store, epoch, record_type)
        self.acc_store['loss'].append(loss)
        self.acc_store['acc'].append(acc)

    # a function that records the prediction information line into a dictionary and the prediction correctness in a list
    # the function is called at each training/evaluation epoch
    def record_pred(self, epoch, record_type, src, trg, pred_trg):

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
                    if self.pred_log != "all_correct_syll":
                        continue
                else:
                    epoch_correct.append(0)

                # skip this recording if the prediction has wrong syllable structure
                if False in pred_sr_sylls[0] or False in pred_sr_sylls[1]:
                    continue

                # compare the actual and predicted target surface form
                # assume no error and change error from 0 to 1
                if self.pred_log != "vowel_only_error":
                    o1_error = 0
                    o2_error = 0
                    c1_error = 0
                    c2_error = 0
                v1_error = 0
                v2_error = 0

                sr_o1 = sr_sylls[0][0]
                sr_o2 = sr_sylls[1][0]
                sr_v1 = sr_sylls[0][1]
                sr_v2 = sr_sylls[1][1]
                sr_c1 = sr_sylls[0][2]
                sr_c2 = sr_sylls[1][2]

                pred_sr_o1 = pred_sr_sylls[0][0]
                pred_sr_o2 = pred_sr_sylls[1][0]
                pred_sr_v1 = pred_sr_sylls[0][1]
                pred_sr_v2 = pred_sr_sylls[1][1]
                pred_sr_c1 = pred_sr_sylls[0][2]
                pred_sr_c2 = pred_sr_sylls[1][2]

                if self.pred_log != "vowel_only_error":
                    if sr_o1 != pred_sr_o1:
                        o1_error = 1
                    if sr_o2 != pred_sr_o2:
                        o2_error = 1
                    if sr_c1 != pred_sr_c1:
                        c1_error = 1
                    if sr_c2 != pred_sr_c2:
                        c2_error = 1
                elif self.pred_log == "vowel_only_error" and (
                    sr_o1 != pred_sr_o1 or sr_o2 != pred_sr_o2 or
                    sr_c1 != pred_sr_c1 or sr_c2 != pred_sr_c2
                ):
                    continue
                if sr_v1 != pred_sr_v1:
                    v1_error = 1
                if sr_v2 != pred_sr_v2:
                    v2_error = 1

                self._append_base_fields(self.pred_store, epoch, record_type)
                self.pred_store['ur'].append(ur_string)
                self.pred_store['sr'].append(sr_string)
                self.pred_store['pred_sr'].append(pred_sr_string)

                if self.pred_log != "vowel_only_error":
                    self.pred_store['o1_error'].append(o1_error)
                    self.pred_store['o2_error'].append(o2_error)
                    self.pred_store['sr_o1'].append(sr_o1)
                    self.pred_store['sr_o2'].append(sr_o2)
                    self.pred_store['pred_sr_o1'].append(pred_sr_o1)
                    self.pred_store['pred_sr_o2'].append(pred_sr_o2)
                    self.pred_store['c1_error'].append(c1_error)
                    self.pred_store['c2_error'].append(c2_error)
                    self.pred_store['sr_c1'].append(sr_c1)
                    self.pred_store['sr_c2'].append(sr_c2)
                    self.pred_store['pred_sr_c1'].append(pred_sr_c1)
                    self.pred_store['pred_sr_c2'].append(pred_sr_c2)
                self.pred_store['v1_error'].append(v1_error)
                self.pred_store['v2_error'].append(v2_error)
                self.pred_store['sr_v1'].append(sr_v1)
                self.pred_store['sr_v2'].append(sr_v2)
                self.pred_store['pred_sr_v1'].append(pred_sr_v1)
                self.pred_store['pred_sr_v2'].append(pred_sr_v2)

        if len(epoch_correct) == 0:
            raise RuntimeError(f"No predictions recorded for epoch {epoch} ({record_type})")

        epoch_acc = sum(epoch_correct) / len(epoch_correct)  # calculate epoch accuracy rate
        return epoch_acc
