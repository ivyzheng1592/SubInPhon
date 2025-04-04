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

        # results storages: dictionary
        # we convert the dictionary to pandas dataframe and save to file
        self.acc_store = {
            'trial_num': [], 'datatype': [], 'language': [], 'condition': [], 'run_num': [],
            'epoch': [], 'record_type': [], 'loss': [], 'acc': []
        }
        self.pred_store = {
            'trial_num': [], 'datatype': [], 'language': [], 'condition': [], 'run_num': [],
            'epoch': [], 'record_type': [], 'ur': [], 'sr': [], 'pred_sr': [],
            'c_error': [], 'v1_error': [], 'v2_error': [],
            'ur_v1': [], 'ur_v2': [], 'sr_v1': [], 'sr_v2': [], 'pred_sr_v1': [], 'pred_sr_v2': []
        }

        # results files
        self.acc_file = os.path.join("Results", trial_num, datatype,
                                     language + "_" + condition +
                                     "_run" + str(run_num) + "_acc.csv")
        self.pred_file = os.path.join("Results", trial_num, datatype,
                                      language + "_" + condition +
                                      "_run" + str(run_num) + "_pred.csv")
        self.model_file = os.path.join("Results", trial_num, datatype,
                                       language + "_" + condition +
                                       "_run" + str(run_num) + "_seq2seq.pth")
        self.acc_plot = os.path.join("Results", trial_num, datatype,
                                     language + "_" + condition +
                                     "_run" + str(run_num) + "_acc_plot.png")
        self.att_plot_dir = os.path.join("Results", trial_num, datatype,
                                         language + "_" + condition +
                                         "_run" + str(run_num) + "_attention_plots")
        if not os.path.exists(self.att_plot_dir):
            os.mkdir(self.att_plot_dir)

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
            train_loss, train_acc, train_preds = self.train_one_epoch(train_dataloader, hp.teacher_forcing_ratio)
            valid_loss, valid_acc, valid_preds = self.evaluate_one_epoch(valid_dataloader)
            print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f} "
                  f"| Train Acc: {train_acc:7.3f}")
            print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f} "
                  f"| Valid Acc: {valid_acc:7.3f}")

            # record the accuracy value
            self.record_acc(epoch, "train", train_loss, train_acc)
            self.record_acc(epoch, "valid", valid_loss, valid_acc)

            # record the model predicted results
            self.record_pred(epoch, "train", train_preds)
            self.record_pred(epoch, "valid", valid_preds)

            # save the model
            torch.save(self.seq2seq.state_dict(), self.model_file)
            print(f"Model trained and stored at {self.model_file}")

        # save loss, accuracy, and predicted results
        utils.save_to_file(self.acc_store, self.acc_file)
        utils.save_to_file(self.pred_store, self.pred_file)
        utils.plot_acc(self.acc_file, self.acc_plot)
        print(f"Loss, accuracy, and predicted results are saved")

    # a function that completes one repetition of evaluation at the end of training
    def test(self, test_dataloader):
        # load the model
        self.seq2seq.load_state_dict(torch.load(self.model_file))

        # check loss for the test dataset
        test_loss, test_acc, test_preds = self.evaluate_one_epoch(test_dataloader)
        print(f"\tTest Loss: {test_loss:7.3f} | Test PPL: {np.exp(test_loss):7.3f} "
              f"| Test Acc: {test_acc:7.3f}")

        # record the accuracy value and the model predicted results
        self.record_acc(hp.n_epochs, "test", test_loss, test_acc)
        self.record_pred(hp.n_epochs, "test", test_preds)

        # save loss, accuracy, and predicted results
        utils.save_to_file(self.acc_store, self.acc_file)
        utils.save_to_file(self.pred_store, self.pred_file)
        print(f"Loss, accuracy, and predicted results are saved")

    # a function that manages training at one epoch
    def train_one_epoch(self, data_loader, teacher_forcing_ratio):
        self.seq2seq.train()  # enable dropout in training
        epoch_loss = 0
        epoch_acc = 0
        epoch_preds = {}

        # training in one batch
        for i, (src, trg) in enumerate(data_loader):
            # src = [src_len, batch_size]
            # trg = [trg_len, batch_size]

            self.optimizer.zero_grad()  # reset gradient at each iteration to 0
            output, pred, _ = self.seq2seq(src, trg, teacher_forcing_ratio)
            # output = [trg_len, batch_size, output_dim]
            # pred = [trg_len, batch_size]

            # record predictions in dictionary format
            batch_preds = self.one_pred_line(src, trg, pred)
            if epoch_preds:  # if epoch_preds is not empty
                for key in epoch_preds.keys():
                    epoch_preds[key].extend(batch_preds[key])
            else:  # if epoch_preds is empty
                epoch_preds.update(batch_preds)

            batch_size = trg.shape[1]
            # calculate total number of correct predictions in a batch
            batch_correct = torch.all(torch.eq(pred, trg), dim=0).sum()
            batch_acc = batch_correct.item() / batch_size  # calculate batch accuracy rate
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

        return epoch_loss, epoch_acc, epoch_preds

    # a function that manages evaluation at one epoch
    def evaluate_one_epoch(self, data_loader):
        self.seq2seq.eval()  # disable dropout in evaluation
        epoch_loss = 0
        epoch_acc = 0
        epoch_preds = {}

        # evaluation in one batch
        with torch.no_grad():  # disable gradient tracking
            for i, (src, trg) in enumerate(tqdm.tqdm(data_loader)):
                # src = [src_len, batch_size]
                # trg = [trg_len, batch_size]

                output, pred, _ = self.seq2seq(src, trg, 0)  # turn off teacher forcing
                # output = [trg_len, batch_size, output_dim]
                # pred = [trg_len, batch_size]

                # record predictions in dictionary format
                batch_preds = self.one_pred_line(src, trg, pred)
                if epoch_preds:  # if epoch_preds is not empty
                    for key in epoch_preds.keys():
                        epoch_preds[key].extend(batch_preds[key])
                else:  # if epoch_preds is empty
                    epoch_preds.update(batch_preds)

                batch_size = trg.shape[1]
                # calculate total number of correct predictions in a batch
                batch_correct = torch.all(torch.eq(pred, trg), dim=0).sum()
                batch_acc = batch_correct.item() / batch_size  # calculate batch accuracy rate
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

        return epoch_loss, epoch_acc, epoch_preds

    # a function that manages evaluation of one random batch
    def evaluate_one_batch(self, test_dataloader, dataset):
        # get one random batch of test data
        dataiter = iter(test_dataloader)
        src, trg = next(dataiter)
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]

        # load the model
        self.seq2seq.load_state_dict(torch.load(self.model_file))
        self.seq2seq.eval()  # disable dropout in evaluation
        with torch.no_grad():  # disable gradient tracking
            _, pred, att = self.seq2seq(src, trg, 0)  # turn off teacher forcing
            # pred = [trg_len, batch_size]
            # att = [trg_len, batch_size, src_len]
            batch_preds = self.one_pred_line(src, trg, pred)

            # for individual items in a batch
            for i in range(hp.batch_size):
                ur_tensor = src[:, i]
                sr_tensor = trg[:, i]
                pred_sr_tensor = pred[:, i]
                att_tensor = att[:, i, :]
                # att_tensor = [trg_len, src_len]

                # convert tensor to vector
                ur_vector = [int(x) for x in ur_tensor.tolist()]
                sr_vector = [int(x) for x in sr_tensor.tolist()]
                pred_sr_vector = [int(x) for x in pred_sr_tensor.tolist()]

                # convert vector to word list and string
                ur_list, ur_string = self.dataset.ur_alphabet.vec2word(ur_vector)
                sr_list, sr_string = self.dataset.sr_alphabet.vec2word(sr_vector)
                pred_sr_list, pred_sr_string = self.dataset.sr_alphabet.vec2word(pred_sr_vector)

                # plot attention
                att_plot = os.path.join(self.att_plot_dir,
                                        ur_string + "_" + sr_string + ".png")
                utils.plot_att(ur_list, pred_sr_list, att_tensor, att_plot)

    def one_pred_line(self, src, trg, pred_trg):
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]

        pred_lines = {
            'ur': [], 'sr': [], 'pred_sr': [], 'c_error': [], 'v1_error': [], 'v2_error': [],
            'ur_v1': [], 'ur_v2': [], 'sr_v1': [], 'sr_v2': [], 'pred_sr_v1': [], 'pred_sr_v2': []
        }

        # for individual items in a batch
        for i in range(hp.batch_size):
            ur_tensor = src[:, i]
            sr_tensor = trg[:, i]
            pred_sr_tensor = pred_trg[:, i]

            # convert tensor to vector
            ur_vector = [int(x) for x in ur_tensor.tolist()]
            sr_vector = [int(x) for x in sr_tensor.tolist()]
            pred_sr_vector = [int(x) for x in pred_sr_tensor.tolist()]

            # convert vector to word list and string
            ur_list, ur_string = self.dataset.ur_alphabet.vec2word(ur_vector)
            sr_list, sr_string = self.dataset.sr_alphabet.vec2word(sr_vector)
            pred_sr_list, pred_sr_string = self.dataset.sr_alphabet.vec2word(pred_sr_vector)

            # decompose word list into structured syllables
            # a list of two lists, each in the shape of [C, V, C]
            ur_sylls = EVH.decompose_stimuli(EVH.onset_ae, EVH.coda_ae, EVH.vowel_front_ae, EVH.vowel_back_ae,
                                             EVH.syll_struct_ae, ur_list)
            sr_sylls = EVH.decompose_stimuli(EVH.onset_ae, EVH.coda_ae, EVH.vowel_front_ae, EVH.vowel_back_ae,
                                             EVH.syll_struct_ae, sr_list)
            pred_sr_sylls = EVH.decompose_stimuli(EVH.onset_ae, EVH.coda_ae, EVH.vowel_front_ae, EVH.vowel_back_ae,
                                                  EVH.syll_struct_ae, pred_sr_list)

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

            # compare the actual and predicted target surface form
            # assume no error and change error from 0 to 1
            c_error = 0
            v1_error = 0
            v2_error = 0
            if sr_o1 != pred_sr_o1 or sr_o2 != pred_sr_o2 or \
                    sr_c1 != pred_sr_c1 or sr_c2 != pred_sr_c2:
                c_error = 1
            if sr_v1 != pred_sr_v1:
                v1_error = 1
            if sr_v2 != pred_sr_v2:
                v2_error = 1

            pred_lines['ur'].append(ur_string)
            pred_lines['sr'].append(sr_string)
            pred_lines['pred_sr'].append(pred_sr_string)
            pred_lines['c_error'].append(c_error)
            pred_lines['v1_error'].append(v1_error)
            pred_lines['v2_error'].append(v2_error)
            pred_lines['ur_v1'].append(ur_v1)
            pred_lines['ur_v2'].append(ur_v2)
            pred_lines['sr_v1'].append(sr_v1)
            pred_lines['sr_v2'].append(sr_v2)
            pred_lines['pred_sr_v1'].append(pred_sr_v1)
            pred_lines['pred_sr_v2'].append(pred_sr_v2)

        return pred_lines

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
        self.pred_store['c_error'].extend(preds['c_error'])
        self.pred_store['v1_error'].extend(preds['v1_error'])
        self.pred_store['v2_error'].extend(preds['v2_error'])
        self.pred_store['ur_v1'].extend(preds['ur_v1'])
        self.pred_store['ur_v2'].extend(preds['ur_v2'])
        self.pred_store['sr_v1'].extend(preds['sr_v1'])
        self.pred_store['sr_v2'].extend(preds['sr_v2'])
        self.pred_store['pred_sr_v1'].extend(preds['pred_sr_v1'])
        self.pred_store['pred_sr_v2'].extend(preds['pred_sr_v2'])