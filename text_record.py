# created 2025/06/04
# A class that handles data recording of multiple runs in dictionaries
# Dictionary data are saved to file using utils in TextRun

import os
import torch
import hyper_params as hp


class TextRecorder:
    def __init__(self, dataset, trial_num, language, condition, run_num):

        self.dataset = dataset
        self.trial_num = trial_num
        self.language = language
        self.lang_name = language.lang_name
        self.condition = condition
        self.run_num = run_num

        self.ur_alphabet = self.dataset.ur_alphabet
        self.sr_alphabet = self.dataset.sr_alphabet

        # result storages
        self.acc_store = {
            'trial_num': [], 'language': [], 'condition': [], 'run_num': [], 'epoch': [],
            'record_type': [], 'loss': [], 'acc': []
        }

        self.pred_store = {
            'trial_num': [], 'language': [], 'condition': [],
            'run_num': [], 'epoch': [], 'record_type': [],
            'ur': [], 'sr': [], 'pred_sr': [],
            #'o1_error': [], 'o2_error': [],
            #'sr_o1': [], 'sr_o2': [], 'pred_sr_o1': [], 'pred_sr_o2': [],
            #'c1_error': [], 'c2_error': [],
            #'sr_c1': [], 'sr_c2': [], 'pred_sr_c1': [], 'pred_sr_c2': [],
            'v1_error': [], 'v2_error': [], 'ur_v1': [], 'ur_v2': [],
            'sr_v1': [], 'sr_v2': [], 'pred_sr_v1': [], 'pred_sr_v2': []
        }

        self.att_store = {
            'ur': [], 'pred_sr': [], 'self/self': [1] * hp.batch_size,
            'v1/v2': [0] * hp.batch_size, 'v2/v1': [0] * hp.batch_size,
            'v/c': [0] * hp.batch_size, 'c/v': [0] * hp.batch_size
        }

        self.ur_embed_phone_store = {key: [] for key in self.language.phone}
        self.ur_embed_vowel_store = {key: [] for key in self.language.vowel}
        self.sr_embed_phone_store = {key: [] for key in self.language.phone}
        self.sr_embed_vowel_store = {key: [] for key in self.language.vowel}

        # results files and directories
        self.acc_file = os.path.join("Results", trial_num + "_" + self.lang_name,
                                     self.lang_name + "_acc.csv")
        self.pred_file = os.path.join("Results", trial_num + "_" + self.lang_name,
                                      self.lang_name + "_pred.csv")

        self.acc_plot_dir = os.path.join("Results", trial_num + "_" + self.lang_name,
                                         self.lang_name + "_acc_plots")
        os.makedirs(self.acc_plot_dir, exist_ok=True)
        self.acc_plot = os.path.join(self.acc_plot_dir,
                                     self.lang_name + "_" + self.condition +
                                     "_run" + str(self.run_num) + "_acc_plot.png")

        self.model_dir = os.path.join("Results", trial_num + "_" + self.lang_name,
                                      self.lang_name + "_model_files",
                                      self.lang_name + "_" + self.condition +
                                      "_run" + str(self.run_num) + "_model_files")
        os.makedirs(self.model_dir, exist_ok=True)

        self.att_plot_dir = os.path.join("Results", trial_num + "_" + self.lang_name,
                                         self.lang_name + "_att_plots",
                                         self.lang_name + "_" + self.condition +
                                         "_run" + str(self.run_num) + "_att_plots")
        os.makedirs(self.att_plot_dir, exist_ok=True)
        self.att_file = os.path.join(self.att_plot_dir,
                                     self.lang_name + "_" + self.condition +
                                     "_run" + str(self.run_num) + "_att_type.csv")

        self.embed_plot_dir = os.path.join("Results", trial_num + "_" + self.lang_name,
                                           self.lang_name + "_embed_plots",
                                           self.lang_name + "_" + self.condition +
                                           "_run" + str(self.run_num) + "_embed_plots")
        os.makedirs(self.embed_plot_dir, exist_ok=True)

    # a function that converts one pair of ur, sr, and pred_sr tensor to list
    def tensor2list(self, ur, sr, pred_sr):
        # convert tensor to vector
        ur_vector = ur.cpu().numpy()
        sr_vector = sr.cpu().numpy()
        pred_sr_vector = pred_sr.cpu().numpy()

        # convert vector to word list and string
        ur_list, _ = self.dataset.ur_alphabet.vec2word(ur_vector)
        sr_list, _ = self.dataset.sr_alphabet.vec2word(sr_vector)
        pred_sr_list, _ = self.dataset.sr_alphabet.vec2word(pred_sr_vector)

        return ur_list, sr_list, pred_sr_list

    # a function that converts one pair of ur, sr, and pred_sr tensor to string
    def tensor2string(self, ur, sr, pred_sr):
        # convert tensor to vector
        ur_vector = ur.cpu().numpy()
        sr_vector = sr.cpu().numpy()
        pred_sr_vector = pred_sr.cpu().numpy()

        # convert vector to word list and string
        _, ur_string = self.dataset.ur_alphabet.vec2word(ur_vector)
        _, sr_string = self.dataset.sr_alphabet.vec2word(sr_vector)
        _, pred_sr_string = self.dataset.sr_alphabet.vec2word(pred_sr_vector)

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
        self.acc_store['trial_num'].append(self.trial_num)
        self.acc_store['language'].append(self.lang_name)
        self.acc_store['condition'].append(self.condition)
        self.acc_store['run_num'].append(self.run_num)
        self.acc_store['epoch'].append(epoch)
        self.acc_store['record_type'].append(record_type)
        self.acc_store['loss'].append(loss)
        self.acc_store['acc'].append(acc)

    # a function that records the prediction information line into a dictionary and the prediction correctness in a list
    # the function is called at each training/evaluation batch
    def record_pred(self, epoch, record_type, src, trg, pred_trg):

        batch_correct = []

        # for individual pairs in a batch
        for i in range(hp.batch_size):
            ur = src[:, i]
            sr = trg[:, i]
            pred_sr = pred_trg[:, i]

            # convert the pair
            ur_string, sr_string, pred_sr_string = self.tensor2string(ur, sr, pred_sr)
            ur_sylls, sr_sylls, pred_sr_sylls = self.tensor2syll(ur, sr, pred_sr)

            # record prediction correctness
            # skip prediction recording if the prediction is correct
            if sr_string == pred_sr_string:
                batch_correct.insert(i, 1)
                continue
            else:
                batch_correct.insert(i, 0)

            # skip this recording if the prediction has wrong syllable structure
            if False in pred_sr_sylls[0] or False in pred_sr_sylls[1]:
                continue

            # compare the actual and predicted target surface form
            # assume no error and change error from 0 to 1
            #o1_error = 0
            #o2_error = 0
            #c1_error = 0
            #c2_error = 0
            v1_error = 0
            v2_error = 0

            ur_v1 = ur_sylls[0][1]
            ur_v2 = ur_sylls[1][1]

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

            # skip this recording if the prediction has wrong consonant
            if sr_o1 != pred_sr_o1 or sr_o2 != pred_sr_o2 or \
                sr_c1 != pred_sr_c1 or sr_c2 != pred_sr_c2:
                continue
            #if sr_o1 != pred_sr_o1:
                #o1_error = 1
            #if sr_o2 != pred_sr_o2:
                #o2_error = 1
            #if sr_c1 != pred_sr_c1:
                #c1_error = 1
            #if sr_c2 != pred_sr_c2:
                #c2_error = 1
            if sr_v1 != pred_sr_v1:
                v1_error = 1
            if sr_v2 != pred_sr_v2:
                v2_error = 1

            self.pred_store['trial_num'].append(self.trial_num)
            self.pred_store['language'].append(self.lang_name)
            self.pred_store['condition'].append(self.condition)
            self.pred_store['run_num'].append(self.run_num)
            self.pred_store['epoch'].append(epoch)
            self.pred_store['record_type'].append(record_type)

            self.pred_store['ur'].append(ur_string)
            self.pred_store['sr'].append(sr_string)
            self.pred_store['pred_sr'].append(pred_sr_string)

            #self.pred_store['o1_error'].append(o1_error)
            #self.pred_store['o2_error'].append(o2_error)
            #self.pred_store['sr_o1'].append(sr_o1)
            #self.pred_store['sr_o2'].append(sr_o2)
            #self.pred_store['pred_sr_o1'].append(pred_sr_o1)
            #self.pred_store['pred_sr_o2'].append(pred_sr_o2)
            #self.pred_store['c1_error'].append(c1_error)
            #self.pred_store['c2_error'].append(c2_error)
            #self.pred_store['sr_c1'].append(sr_c1)
            #self.pred_store['sr_c2'].append(sr_c2)
            #self.pred_store['pred_sr_c1'].append(pred_sr_c1)
            #self.pred_store['pred_sr_c2'].append(pred_sr_c2)
            self.pred_store['v1_error'].append(v1_error)
            self.pred_store['v2_error'].append(v2_error)
            self.pred_store['ur_v1'].append(ur_v1)
            self.pred_store['ur_v2'].append(ur_v2)
            self.pred_store['sr_v1'].append(sr_v1)
            self.pred_store['sr_v2'].append(sr_v2)
            self.pred_store['pred_sr_v1'].append(pred_sr_v1)
            self.pred_store['pred_sr_v2'].append(pred_sr_v2)

        return batch_correct

    # a function that records and plots attention from each predicted sr to ur in a batch
    # the function is called in evaluate attention
    def record_att(self, i, pred_sr_sylls, word_att, ur_string, pred_sr_string):

        # get the largest attention value for individual token in the predicted sr
        max_att = torch.argmax(word_att, dim=1).tolist()
        # max_att = [trg_len]

        # for individual token in the predicted sr
        # check the position of the largest attention value for each token
        # assume attention is self/self and change from 1 to 0 if not
        if pred_sr_sylls[0][0] is None and pred_sr_sylls[1][2] is None:  # <SOS>VCV<EOS>
            if max_att[1] in [3]:
                self.att_store['v1/v2'][i] = 1
                self.att_store['self/self'][i] = 0
            if max_att[3] in [1]:
                self.att_store['v2/v1'][i] = 1
                self.att_store['self/self'][i] = 0
            if max_att[1] in [2, 4] or max_att[3] in [2, 4]:
                self.att_store['v/c'][i] = 1
                self.att_store['self/self'][i] = 0
            if max_att[2] in [1, 3]:
                self.att_store['c/v'][i] = 1
                self.att_store['self/self'][i] = 0
        if pred_sr_sylls[0][0] is None and pred_sr_sylls[1][2] is not None:  # <SOS>VCVC<EOS>
            if max_att[1] in [3]:
                self.att_store['v1/v2'][i] = 1
                self.att_store['self/self'][i] = 0
            if max_att[3] in [1]:
                self.att_store['v2/v1'][i] = 1
                self.att_store['self/self'][i] = 0
            if max_att[1] in [2, 4] or max_att[3] in [2, 4]:
                self.att_store['v/c'][i] = 1
                self.att_store['self/self'][i] = 0
            if max_att[2] in [1, 3] or max_att[4] in [1, 3]:
                self.att_store['c/v'][i] = 1
                self.att_store['self/self'][i] = 0
        if pred_sr_sylls[0][0] is not None and pred_sr_sylls[1][2] is None:  # <SOS>CVCV<EOS>
            if max_att[2] in [4]:
                self.att_store['v1/v2'][i] = 1
                self.att_store['self/self'][i] = 0
            if max_att[4] in [2]:
                self.att_store['v2/v1'][i] = 1
                self.att_store['self/self'][i] = 0
            if max_att[2] in [1, 3, 5] or max_att[4] in [1, 3, 5]:
                self.att_store['v/c'][i] = 1
                self.att_store['self/self'][i] = 0
            if max_att[1] in [2, 4] or max_att[3] in [2, 4]:
                self.att_store['c/v'][i] = 1
                self.att_store['self/self'][i] = 0
        if pred_sr_sylls[0][0] is not None and pred_sr_sylls[1][2] is not None:  # <SOS>CVCVC<EOS>
            if max_att[2] in [4]:
                self.att_store['v1/v2'][i] = 1
                self.att_store['self/self'][i] = 0
            if max_att[4] in [2]:
                self.att_store['v2/v1'][i] = 1
                self.att_store['self/self'][i] = 0
            if max_att[2] in [1, 3, 5] or max_att[4] in [1, 3, 5]:
                self.att_store['v/c'][i] = 1
                self.att_store['self/self'][i] = 0
            if max_att[1] in [2, 4] or max_att[3] in [2, 4] or max_att[5] in [2, 4]:
                self.att_store['c/v'][i] = 1
                self.att_store['self/self'][i] = 0

        # append ur and pred sr of the current word
        self.att_store['ur'].append(ur_string)
        self.att_store['pred_sr'].append(pred_sr_string)

    # a function that records source and target embedding
    # the function is called in evaluate embedding
    def record_embed(self, ur_embed, sr_embed):
        ur_embed = ur_embed.detach().numpy()
        sr_embed = sr_embed.detach().numpy()
        # ur_embed = [input_dim, embedding_dim]
        # sr_embed = [input_dim, embedding_dim]

        # for individual embedding in ur_embed and sr_embed
        input_dim = ur_embed.shape[0]
        for i in range(input_dim):
            ur_char = self.dataset.ur_alphabet.idx2char[i]
            sr_char = self.dataset.sr_alphabet.idx2char[i]

            if ur_char in self.ur_embed_phone_store:
                self.ur_embed_phone_store[ur_char] = ur_embed[i]
            if ur_char in self.ur_embed_vowel_store:
                self.ur_embed_vowel_store[ur_char] = ur_embed[i]
            if sr_char in self.sr_embed_phone_store:
                self.sr_embed_phone_store[sr_char] = sr_embed[i]
            if sr_char in self.sr_embed_vowel_store:
                self.sr_embed_vowel_store[sr_char] = sr_embed[i]
