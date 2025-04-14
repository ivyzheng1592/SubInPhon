# created 2025/03/07 structure based on Ben Trevett tutorial
# updated 2025/04/01 upgraded to class
# A script that defines training and evaluation at each epoch of each run

import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
import hyper_params as hp
import utils
import Dataset.stimuli_generator as EVH


class TextRun:
    def __init__(self, seq2seq, dataset, trial_num, datatype, language, condition, run_num):

        # condition hyperparameters
        self.seq2seq = seq2seq
        self.dataset = dataset
        self.trial_num = trial_num
        self.datatype = datatype
        self.language = language
        self.condition = condition
        self.run_num = run_num

        self.ur_alphabet = self.dataset.ur_alphabet
        self.sr_alphabet = self.dataset.sr_alphabet

        # results storages: dictionary
        # we convert the dictionary to pandas dataframe and save to file
        self.acc_store = {
            'trial_num': [], 'datatype': [], 'language': [], 'condition': [], 'run_num': [],
            'epoch': [], 'record_type': [], 'loss': [], 'acc': []
        }

        self.pred_store = {
            'trial_num': [], 'datatype': [], 'language': [], 'condition': [], 'run_num': [],
            'epoch': [], 'record_type': [], 'ur': [], 'sr': [], 'pred_sr': [],
            'o1_error': [], 'o2_error': [], 'c1_error': [], 'c2_error': [],
            'sr_o1': [], 'sr_o2': [], 'pred_sr_o1': [], 'pred_sr_o2': [],
            'sr_c1': [], 'sr_c2': [], 'pred_sr_c1': [], 'pred_sr_c2': [],
            'v1_error': [], 'v2_error': [], 'ur_v1': [], 'ur_v2': [],
            'sr_v1': [], 'sr_v2': [], 'pred_sr_v1': [], 'pred_sr_v2': []
        }

        EVH_phone = (EVH.onset_ae + EVH.coda_ae +
                    list(EVH.vowel_front_ae_txt.keys()) +
                    list(EVH.vowel_back_ae_txt.keys()))
        EVH_vowel = (list(EVH.vowel_front_ae_txt.keys()) +
                    list(EVH.vowel_back_ae_txt.keys()))
        self.ur_embed_phone_store = {key: [] for key in EVH_phone}
        self.ur_embed_vowel_store = {key: [] for key in EVH_vowel}
        self.sr_embed_phone_store = {key: [] for key in EVH_phone}
        self.sr_embed_vowel_store = {key: [] for key in EVH_vowel}

        self.att_store = {
            'ur': [], 'pred_sr': [], 'self/self': [1] * hp.batch_size,
            'v1/v2': [0] * hp.batch_size, 'v2/v1': [0] * hp.batch_size,
            'v/c': [0] * hp.batch_size, 'c/v': [0] * hp.batch_size
        }

        # results files
        self.acc_file = os.path.join("Results", trial_num + "_" + datatype,
                                     language + "_" + condition +
                                     "_run" + str(run_num) + "_acc.csv")
        self.pred_file = os.path.join("Results", trial_num + "_" + datatype,
                                      language + "_" + condition +
                                      "_run" + str(run_num) + "_pred.csv")
        self.acc_plot = os.path.join("Results", trial_num + "_" + datatype,
                                     language + "_" + condition +
                                     "_run" + str(run_num) + "_acc_plot.png")

        self.model_dir = os.path.join("Results", trial_num + "_" + datatype,
                                      language + "_" + condition +
                                      "_run" + str(run_num) + "_model_files")
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.embed_plot_dir = os.path.join("Results", trial_num + "_" + datatype,
                                           language + "_" + condition +
                                           "_run" + str(run_num) + "_embed_plots")
        if not os.path.exists(self.embed_plot_dir):
            os.mkdir(self.embed_plot_dir)

        self.att_plot_dir = os.path.join("Results", trial_num + "_" + datatype,
                                         language + "_" + condition +
                                         "_run" + str(run_num) + "_att_plots")
        if not os.path.exists(self.att_plot_dir):
            os.mkdir(self.att_plot_dir)
        self.att_file = os.path.join(self.att_plot_dir,
                                     language + "_" + condition +
                                     "_run" + str(run_num) + "_att_type.csv")

        # model weight initialization
        for name, param in self.seq2seq.named_parameters():
            if "weight" in name:  # weights
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:  # biases
                nn.init.constant_(param.data, 0)
        #model_parameters = sum(p.numel() for p in self.seq2seq.parameters() if p.requires_grad)

        # optimizer and loss function
        self.optimizer = torch.optim.Adam(self.seq2seq.parameters(), lr=hp.learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=hp.special_tokens.index(hp.pad_token))

    # a function that completes one repetition of training
    def train(self, train_dataloader, valid_dataloader):

        # at each epoch, display progress bar
        for epoch in tqdm.tqdm(range(hp.n_epochs)):
            # update loss for each batch
            train_loss, train_acc = self.train_one_epoch(epoch, "train",
                                                         train_dataloader, hp.teacher_forcing_ratio)
            valid_loss, valid_acc = self.evaluate_one_epoch(epoch, "valid", valid_dataloader)
            print(f"Epoch {epoch} Train Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f} "
                  f"| Train Acc: {train_acc:7.3f}")
            print(f"Epoch {epoch} Valid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f} "
                  f"| Valid Acc: {valid_acc:7.3f}")

            # record the accuracy value
            self.record_acc(epoch, "train", train_loss, train_acc)
            self.record_acc(epoch, "valid", valid_loss, valid_acc)

            # save the model
            model_file = os.path.join(self.model_dir,
                                      self.language + "_" + self.condition +
                                      "_run" + str(self.run_num) +
                                      "_epoch" + str(epoch) + "_seq2seq.pth")
            torch.save(self.seq2seq.state_dict(), model_file)
            print(f"Epoch {epoch} model trained and stored at {model_file}")

        # save loss, accuracy, and predicted results
        utils.save_to_file(self.acc_store, self.acc_file)
        utils.save_to_file(self.pred_store, self.pred_file)
        utils.plot_acc(self.acc_file, self.acc_plot)
        print(f"Run {self.run_num} training loss, accuracy, and predicted results are saved")

    # a function that completes one repetition of evaluation at the end of training
    def test(self, test_dataloader):
        # load the model
        model_file = os.path.join(self.model_dir,
                                  self.language + "_" + self.condition +
                                  "_run" + str(self.run_num) +
                                  "_epoch" + str(hp.n_epochs-1) + "_seq2seq.pth")
        self.seq2seq.load_state_dict(torch.load(model_file))

        # check loss for the test dataset
        test_loss, test_acc = self.evaluate_one_epoch(hp.n_epochs, "test", test_dataloader)
        print(f"Run {self.run_num} Test Loss: {test_loss:7.3f} | Test PPL: {np.exp(test_loss):7.3f} "
              f"| Test Acc: {test_acc:7.3f}")

        # record the accuracy value
        self.record_acc(hp.n_epochs, "test", test_loss, test_acc)

        # save loss, accuracy, and predicted results
        utils.save_to_file(self.acc_store, self.acc_file)
        utils.save_to_file(self.pred_store, self.pred_file)
        print(f"Run {self.run_num} testing loss, accuracy, and predicted results are saved")

    # a function that manages training at one epoch
    def train_one_epoch(self, epoch, record_type, data_loader, teacher_forcing_ratio):
        self.seq2seq.train()  # enable dropout in training
        epoch_loss = 0
        epoch_acc = 0

        # training in one batch
        for i, (src, trg) in enumerate(data_loader):
            # src = [src_len, batch_size]
            # trg = [trg_len, batch_size]

            self.optimizer.zero_grad()  # reset gradient at each iteration to 0
            output, pred, _, _, _ = self.seq2seq(src, trg, teacher_forcing_ratio)
            # output = [trg_len, batch_size, output_dim]
            # pred = [trg_len, batch_size]

            # record predictions in dictionary format
            trg_strings, pred_strings, batch_preds = self.transform_one_batch(src, trg, pred)
            self.record_pred(epoch, record_type, batch_preds)

            # calculate total number of correct predictions in a batch
            batch_correct = [1 if trg_strings[i] == pred_strings[i] else 0
                             for i in range(hp.batch_size)]
            batch_acc = sum(batch_correct) / len(batch_correct)  # calculate batch accuracy rate
            epoch_acc += batch_acc  # add to epoch accuracy rate

            # remove the <SOS> token from output and target and reshape for loss calculation
            output_dim = output.shape[2]
            output = output[1:].view(-1, output_dim)
            # output = [(trg_len - 1) * batch_size, output_dim]
            trg = trg[1:].view(-1)
            # trg = [(trg_len - 1) * batch_size]

            batch_loss = self.criterion(output, trg)  # calculate batch loss
            epoch_loss += batch_loss.item()  # add to epoch loss
            batch_loss.backward()  # backpropagate loss
            # nn.utils.clip_grad_norm_(model.parameters(), clip)
            # clip the gradients to prevent exploding, uncomment if necessary
            self.optimizer.step()  # update the weights

        # average loss and accuracy over all batches
        epoch_loss = epoch_loss / len(data_loader)
        epoch_acc = epoch_acc / len(data_loader)

        return epoch_loss, epoch_acc

    # a function that manages evaluation at one epoch
    def evaluate_one_epoch(self, epoch, record_type, data_loader):
        self.seq2seq.eval()  # disable dropout in evaluation
        epoch_loss = 0
        epoch_acc = 0

        # evaluation in one batch
        with torch.no_grad():  # disable gradient tracking
            for i, (src, trg) in enumerate(data_loader):
                # src = [src_len, batch_size]
                # trg = [trg_len, batch_size]

                output, pred, _, _, _ = self.seq2seq(src, trg, 0)  # turn off teacher forcing
                # output = [trg_len, batch_size, output_dim]
                # pred = [trg_len, batch_size]

                # record predictions in dictionary format
                trg_strings, pred_strings, batch_preds = self.transform_one_batch(src, trg, pred)
                self.record_pred(epoch, record_type, batch_preds)

                # calculate total number of correct predictions in a batch
                batch_correct = [1 if trg_strings[i] == pred_strings[i] else 0
                                 for i in range(hp.batch_size)]
                batch_acc = sum(batch_correct) / len(batch_correct)  # calculate batch accuracy rate
                epoch_acc += batch_acc  # add to epoch accuracy rate

                # remove the <SOS> token from output and target and reshape for loss calculation
                output_dim = output.shape[2]
                output = output[1:].view(-1, output_dim)
                # output = [(trg_len - 1) * batch_size, output_dim]
                trg = trg[1:].view(-1)
                # trg = [(trg_len - 1) * batch_size]

                batch_loss = self.criterion(output, trg)  # calculate batch loss
                epoch_loss += batch_loss.item()  # add to epoch loss

        # average loss and accuracy over all batches
        epoch_loss = epoch_loss / len(data_loader)
        epoch_acc = epoch_acc / len(data_loader)

        return epoch_loss, epoch_acc

    # a function that manages evaluation of one random batch
    def evaluate_one_batch(self, test_dataloader, eval_epoch = hp.n_epochs-1, eval_type="both"):
        # get one random batch of test data
        dataiter = iter(test_dataloader)
        src, trg = next(dataiter)
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]

        # load the model
        model_file = os.path.join(self.model_dir,
                                  self.language + "_" + self.condition +
                                  "_run" + str(self.run_num) +
                                  "_epoch" + str(eval_epoch) + "_seq2seq.pth")
        self.seq2seq.load_state_dict(torch.load(model_file))
        self.seq2seq.eval()  # disable dropout in evaluation

        with torch.no_grad():  # disable gradient tracking
            # get decoder embedding and predicted attention weights
            _, pred, src_embed, trg_embed, att = self.seq2seq(src, trg, 0)  # turn off teacher forcing
            # pred = [trg_len, batch_size]
            # src_embed = [src_len, batch_size, embedding_dim]
            # trg_embed = [trg_len, batch_size, embedding_dim]
            # att = [trg_len, batch_size, src_len]

            # for individual pairs in a batch
            for i in range(hp.batch_size):
                ur = src[:, i]
                sr = trg[:, i]
                pred_sr = pred[:, i]

                # transform the pair
                (ur_list, ur_string,
                 sr_list, sr_string,
                 pred_sr_list, pred_sr_string) = self.transform_one_pair(ur, sr, pred_sr)

                # retrieve attention weights
                # notice that trg_len and src_len have changed because padding was removed
                src_len = len(ur_list)
                trg_len = len(sr_list)
                word_att = att[:trg_len, i, :src_len]
                # word_att = [trg_len, src_len]

                if eval_type == "both" or eval_type == "attention":
                    # plot attention
                    att_plot = os.path.join(self.att_plot_dir,
                                            self.language + "_" + self.condition +
                                            "_run" + str(self.run_num) +
                                            "_epoch" + str(eval_epoch) + "_" +
                                            ur_string + "_" + sr_string + ".png")
                    utils.plot_att(ur_list, pred_sr_list, word_att, att_plot)

                    # decompose word list into structured syllables
                    # a list of two lists, each in the shape of [C, V, C]
                    pred_sr_sylls = EVH.decompose_stimuli(EVH.onset_ae, EVH.coda_ae, EVH.vowel_front_ae_txt,
                                                          EVH.vowel_back_ae_txt, pred_sr_list)

                    # record attention type of the predicted sr
                    self.record_att(i, pred_sr_sylls, word_att, ur_string, sr_string)

                if eval_type == "both" or eval_type == "embedding":
                    # record embedding of the ur and predicted sr
                    self.record_embed(i, ur_list, pred_sr_list, src_embed, trg_embed)

            if eval_type == "both" or eval_type == "attention":
                # save attention recording to file
                utils.save_to_file(self.att_store, self.att_file)
                print(f"Run {self.run_num} attention plots and types are saved for investigation")

            if eval_type == "both" or eval_type == "embedding":
                # plot embedding for both all phones and only vowels
                ur_embed_phone_plot = os.path.join(self.embed_plot_dir,
                                                   self.language + "_" + self.condition +
                                                   "_run" + str(self.run_num) +
                                                   "_epoch" + str(eval_epoch) + "_ur_phoneme.png")
                ur_embed_phone_file = os.path.join(self.embed_plot_dir,
                                                   self.language + "_" + self.condition +
                                                   "_run" + str(self.run_num) +
                                                   "_epoch" + str(eval_epoch) + "_ur_phoneme.csv")
                ur_embed_vowel_plot = os.path.join(self.embed_plot_dir,
                                                   self.language + "_" + self.condition +
                                                   "_run" + str(self.run_num) +
                                                   "_epoch" + str(eval_epoch) + "_ur_vowel.png")
                ur_embed_vowel_file = os.path.join(self.embed_plot_dir,
                                                   self.language + "_" + self.condition +
                                                   "_run" + str(self.run_num) +
                                                   "_epoch" + str(eval_epoch) + "_ur_vowel.csv")
                sr_embed_phone_plot = os.path.join(self.embed_plot_dir,
                                                   self.language + "_" + self.condition +
                                                   "_run" + str(self.run_num) +
                                                   "_epoch" + str(eval_epoch) + "_sr_phoneme.png")
                sr_embed_phone_file = os.path.join(self.embed_plot_dir,
                                                   self.language + "_" + self.condition +
                                                   "_run" + str(self.run_num) +
                                                   "_epoch" + str(eval_epoch) + "_sr_phoneme.csv")
                sr_embed_vowel_plot = os.path.join(self.embed_plot_dir,
                                                   self.language + "_" + self.condition +
                                                   "_run" + str(self.run_num) +
                                                   "_epoch" + str(eval_epoch) + "_sr_vowel.png")
                sr_embed_vowel_file = os.path.join(self.embed_plot_dir,
                                                   self.language + "_" + self.condition +
                                                   "_run" + str(self.run_num) +
                                                   "_epoch" + str(eval_epoch) + "_sr_vowel.csv")

                try:
                    # Code that might raise a runtime error
                    utils.plot_embed(self.ur_embed_phone_store, ur_embed_phone_file, ur_embed_phone_plot,
                                     "phoneme embedding")
                    utils.plot_embed(self.ur_embed_vowel_store, ur_embed_vowel_file, ur_embed_vowel_plot,
                                     "vowel embedding")
                    utils.plot_embed(self.sr_embed_phone_store, sr_embed_phone_file, sr_embed_phone_plot,
                                     "phoneme embedding")
                    utils.plot_embed(self.sr_embed_vowel_store, sr_embed_vowel_file, sr_embed_vowel_plot,
                                     "vowel embedding")
                except Exception as e:
                    print(f"The error {e} occurred in run {self.run_num}. Continue running ...")
                    self.evaluate_one_batch(test_dataloader, eval_epoch=eval_epoch, eval_type="embedding")
                print(f"Run {self.run_num} embedding plots and files are saved for investigation")

    # a function that transforms one pair of ur, sr, and pred_sr tensor to list and string
    def transform_one_pair(self, ur, sr, pred_sr):
        # convert tensor to vector
        ur_vector = [int(x) for x in ur.tolist()]
        sr_vector = [int(x) for x in sr.tolist()]
        pred_sr_vector = [int(x) for x in pred_sr.tolist()]

        # convert vector to word list and string
        ur_list, ur_string = self.dataset.ur_alphabet.vec2word(ur_vector)
        sr_list, sr_string = self.dataset.sr_alphabet.vec2word(sr_vector)
        pred_sr_list, pred_sr_string = self.dataset.sr_alphabet.vec2word(pred_sr_vector)

        return ur_list, ur_string, sr_list, sr_string, pred_sr_list, pred_sr_string

    # a function that transforms one batch of src, trg, and pred_trg to list and string
    # and one line that contains all prediction information we want to record
    def transform_one_batch(self, src, trg, pred_trg):
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]

        pred_lines = {
            'ur': [], 'sr': [], 'pred_sr': [],
            'o1_error': [], 'o2_error': [], 'c1_error': [], 'c2_error': [],
            'sr_o1': [], 'sr_o2': [], 'pred_sr_o1': [], 'pred_sr_o2': [],
            'sr_c1': [], 'sr_c2': [], 'pred_sr_c1': [], 'pred_sr_c2': [],
            'v1_error': [], 'v2_error': [], 'ur_v1': [], 'ur_v2': [],
            'sr_v1': [], 'sr_v2': [], 'pred_sr_v1': [], 'pred_sr_v2': []
        }
        trg_strings = []
        pred_strings = []

        # for individual pairs in a batch
        for i in range(hp.batch_size):
            ur = src[:, i]
            sr = trg[:, i]
            pred_sr = pred_trg[:, i]

            # transform the pair
            (ur_list, ur_string,
             sr_list, sr_string,
             pred_sr_list, pred_sr_string) = self.transform_one_pair(ur, sr, pred_sr)

            trg_strings.insert(i, sr_string)
            pred_strings.insert(i, pred_sr_string)

            # skip this recording if the prediction is correct
            if sr_string == pred_sr_string:
                continue

            # decompose word list into structured syllables
            # a list of two lists, each in the shape of [C, V, C]
            ur_sylls = EVH.decompose_stimuli(EVH.onset_ae, EVH.coda_ae, EVH.vowel_front_ae_txt,
                                             EVH.vowel_back_ae_txt, ur_list)
            sr_sylls = EVH.decompose_stimuli(EVH.onset_ae, EVH.coda_ae, EVH.vowel_front_ae_txt,
                                             EVH.vowel_back_ae_txt, sr_list)
            pred_sr_sylls = EVH.decompose_stimuli(EVH.onset_ae, EVH.coda_ae, EVH.vowel_front_ae_txt,
                                             EVH.vowel_back_ae_txt, pred_sr_list)

            # skip this recording if the prediction has wrong syllable structure
            if False in pred_sr_sylls[0] or False in pred_sr_sylls[1]:
                continue

            # compare the actual and predicted target surface form
            # assume no error and change error from 0 to 1
            o1_error = 0
            o2_error = 0
            c1_error = 0
            c2_error = 0
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
            #if sr_o1 != pred_sr_o1 or sr_o2 != pred_sr_o2 or \
                    #sr_c1 != pred_sr_c1 or sr_c2 != pred_sr_c2:
                #continue
            if sr_o1 != pred_sr_o1:
                o1_error = 1
            if sr_o2 != pred_sr_o2:
                o2_error = 1
            if sr_c1 != pred_sr_c1:
                c1_error = 1
            if sr_c2 != pred_sr_c2:
                c2_error = 1
            if sr_v1 != pred_sr_v1:
                v1_error = 1
            if sr_v2 != pred_sr_v2:
                v2_error = 1

            pred_lines['ur'].append(ur_string)
            pred_lines['sr'].append(sr_string)
            pred_lines['pred_sr'].append(pred_sr_string)
            pred_lines['o1_error'].append(o1_error)
            pred_lines['o2_error'].append(o2_error)
            pred_lines['c1_error'].append(c1_error)
            pred_lines['c2_error'].append(c2_error)
            pred_lines['sr_o1'].append(sr_o1)
            pred_lines['sr_o2'].append(sr_o2)
            pred_lines['pred_sr_o1'].append(pred_sr_o1)
            pred_lines['pred_sr_o2'].append(pred_sr_o2)
            pred_lines['sr_c1'].append(sr_c1)
            pred_lines['sr_c2'].append(sr_c2)
            pred_lines['pred_sr_c1'].append(pred_sr_c1)
            pred_lines['pred_sr_c2'].append(pred_sr_c2)
            pred_lines['v1_error'].append(v1_error)
            pred_lines['v2_error'].append(v2_error)
            pred_lines['ur_v1'].append(ur_v1)
            pred_lines['ur_v2'].append(ur_v2)
            pred_lines['sr_v1'].append(sr_v1)
            pred_lines['sr_v2'].append(sr_v2)
            pred_lines['pred_sr_v1'].append(pred_sr_v1)
            pred_lines['pred_sr_v2'].append(pred_sr_v2)

        return trg_strings, pred_strings, pred_lines

    # a function that records accuracy rates into a dictionary
    # the function is called at each training/evaluation epoch
    def record_acc(self, epoch, record_type, loss, acc):
        # add current accuracy data to the accuracy data storage
        self.acc_store['trial_num'].append(self.trial_num)
        self.acc_store['datatype'].append(self.datatype)
        self.acc_store['language'].append(self.language)
        self.acc_store['condition'].append(self.condition)
        self.acc_store['run_num'].append(self.run_num)
        self.acc_store['epoch'].append(epoch)
        self.acc_store['record_type'].append(record_type)
        self.acc_store['loss'].append(loss)
        self.acc_store['acc'].append(acc)

    # a function that records the prediction information line into a dictionary
    # the function is called at each training/evaluation batch
    def record_pred(self, epoch, record_type, preds):

        # the size of recorded predictions
        # should be len(data_loader) * batch_size
        pred_size = len(preds['ur'])

        # add current prediction data to the prediction data storage
        self.pred_store['trial_num'].extend([self.trial_num] * pred_size)
        self.pred_store['datatype'].extend([self.datatype] * pred_size)
        self.pred_store['language'].extend([self.language] * pred_size)
        self.pred_store['condition'].extend([self.condition] * pred_size)
        self.pred_store['run_num'].extend([self.run_num] * pred_size)
        self.pred_store['epoch'].extend([epoch] * pred_size)
        self.pred_store['record_type'].extend([record_type] * pred_size)
        self.pred_store['ur'].extend(preds['ur'])
        self.pred_store['sr'].extend(preds['sr'])
        self.pred_store['pred_sr'].extend(preds['pred_sr'])
        self.pred_store['o1_error'].extend(preds['o1_error'])
        self.pred_store['o2_error'].extend(preds['o2_error'])
        self.pred_store['c1_error'].extend(preds['c1_error'])
        self.pred_store['c2_error'].extend(preds['c2_error'])
        self.pred_store['sr_o1'].extend(preds['sr_o1'])
        self.pred_store['sr_o2'].extend(preds['sr_o2'])
        self.pred_store['pred_sr_o1'].extend(preds['pred_sr_o1'])
        self.pred_store['pred_sr_o2'].extend(preds['pred_sr_o2'])
        self.pred_store['sr_c1'].extend(preds['sr_c1'])
        self.pred_store['sr_c2'].extend(preds['sr_c2'])
        self.pred_store['pred_sr_c1'].extend(preds['pred_sr_c1'])
        self.pred_store['pred_sr_c2'].extend(preds['pred_sr_c2'])
        self.pred_store['v1_error'].extend(preds['v1_error'])
        self.pred_store['v2_error'].extend(preds['v2_error'])
        self.pred_store['ur_v1'].extend(preds['ur_v1'])
        self.pred_store['ur_v2'].extend(preds['ur_v2'])
        self.pred_store['sr_v1'].extend(preds['sr_v1'])
        self.pred_store['sr_v2'].extend(preds['sr_v2'])
        self.pred_store['pred_sr_v1'].extend(preds['pred_sr_v1'])
        self.pred_store['pred_sr_v2'].extend(preds['pred_sr_v2'])

    # a function that records the attention from each predicted sr to ur in a batch
    # the function is called in evaluate one batch
    def record_att(self, i, pred_sr_sylls, word_att, ur_string, pred_sr_string):
        # get the largest attention value for individual token in the predicted sr
        max_att = torch.argmax(word_att, dim=1).tolist()
        # max_att = [trg_len]

        # for individual token in the predicted sr
        # check the position of the largest attention value for each token
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

    # a function that records the embedding of each ur and sr in a batch
    # the function is called in evaluate one batch
    def record_embed(self, i, ur_list, pred_sr_list, src_embed, trg_embed):
        # for individual token in the ur
        # if the token embedding has not been recorded
        # record its embedding values
        for j, token in enumerate(ur_list):
            if token in self.ur_embed_phone_store and not self.ur_embed_phone_store[token]:
                src_token_embed = src_embed[j, i, :]
                # token_embed = [embedding_dim]
                src_token_embed = src_token_embed.tolist()
                self.ur_embed_phone_store[token] = src_token_embed
            if token in self.ur_embed_vowel_store and not self.ur_embed_vowel_store[token]:
                self.ur_embed_vowel_store[token] = src_token_embed

        # for individual token in the predicted sr
        for j, token in enumerate(pred_sr_list):
            if token in self.sr_embed_phone_store and not self.sr_embed_phone_store[token]:
                trg_token_embed = trg_embed[j, i, :]
                # token_embed = [embedding_dim]
                trg_token_embed = trg_token_embed.tolist()
                self.sr_embed_phone_store[token] = trg_token_embed
            if token in self.sr_embed_vowel_store and not self.sr_embed_vowel_store[token]:
                self.sr_embed_vowel_store[token] = trg_token_embed