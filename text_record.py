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
            'v1_error': [], 'v2_error': [], #'v3_error': [],
            'sr_v1': [], 'sr_v2': [], #'sr_v3': [],
            'pred_sr_v1': [], 'pred_sr_v2': [], #'pred_sr_v3': [],
            #'o1_error': [], 'o2_error': [], 'o3_error': [],
            #'sr_o1': [], 'sr_o2': [], 'sr_o3': [],
            #'pred_sr_o1': [], 'pred_sr_o2': [], 'pred_sr_o3': [],
            #'c1_error': [], 'c2_error': [], 'c3_error': [],
            #'sr_c1': [], 'sr_c2': [], 'sr_c3': [],
            #'pred_sr_c1': [], 'pred_sr_c2': [], 'pred_sr_c3': []
        }

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
                if hp.pred_log != "all_correct_syll":
                    continue
                else:
                    epoch_correct.append(0)

                # skip this recording if the prediction has wrong syllable structure
                if any(False in syll for syll in pred_sr_sylls):
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

                # append only if the column exists (you may comment out columns)
                def _append_if(key, value):
                    if key in self.pred_store:
                        self.pred_store[key].append(value)

                self._append_base_fields(self.pred_store, epoch, record_type)
                self.pred_store['ur'].append(ur_string)
                self.pred_store['sr'].append(sr_string)
                self.pred_store['pred_sr'].append(pred_sr_string)

                # record consonant errors only when enabled
                if hp.pred_log != "vowel_only_error":
                    _append_if('o1_error', o_errors[0])
                    _append_if('o2_error', o_errors[1])
                    _append_if('sr_o1', sr_o[0])
                    _append_if('sr_o2', sr_o[1])
                    _append_if('pred_sr_o1', pred_o[0])
                    _append_if('pred_sr_o2', pred_o[1])
                    _append_if('c1_error', c_errors[0])
                    _append_if('c2_error', c_errors[1])
                    _append_if('sr_c1', sr_c[0])
                    _append_if('sr_c2', sr_c[1])
                    _append_if('pred_sr_c1', pred_c[0])
                    _append_if('pred_sr_c2', pred_c[1])
                _append_if('v1_error', v_errors[0])
                _append_if('v2_error', v_errors[1])
                _append_if('sr_v1', sr_v[0])
                _append_if('sr_v2', sr_v[1])
                _append_if('pred_sr_v1', pred_v[0])
                _append_if('pred_sr_v2', pred_v[1])

                # record third-syllable fields only for 3-syllable outputs
                if syll_count == 3:
                    if hp.pred_log != "vowel_only_error":
                        _append_if('o3_error', o_errors[2])
                        _append_if('sr_o3', sr_o[2])
                        _append_if('pred_sr_o3', pred_o[2])
                        _append_if('c3_error', c_errors[2])
                        _append_if('sr_c3', sr_c[2])
                        _append_if('pred_sr_c3', pred_c[2])
                    _append_if('v3_error', v_errors[2])
                    _append_if('sr_v3', sr_v[2])
                    _append_if('pred_sr_v3', pred_v[2])

        if len(epoch_correct) == 0:
            raise RuntimeError(f"No predictions recorded for epoch {epoch} ({record_type})")

        epoch_acc = sum(epoch_correct) / len(epoch_correct)  # calculate epoch accuracy rate
        return epoch_acc
