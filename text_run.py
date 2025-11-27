# created 2025/03/07 structure based on Ben Trevett tutorial
# updated 2025/04/01 upgraded to class
# A class that defines training and evaluation at each epoch of each run
# Data is recorded into dictionary in TextRecorder

import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
import utils
import hyper_params as hp


class TextRun:
    def __init__(self, seq2seq, recorder):
        # condition hyperparameters
        self.seq2seq = seq2seq
        self.recorder = recorder

        # optimizer and loss function
        self.optimizer = torch.optim.Adam(self.seq2seq.parameters(), lr=hp.learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=hp.special_tokens.index(hp.pad_token))

    # a function that completes one repetition of training
    def train(self, train_dataloader, valid_dataloader): \
        # save untrained model
        model_file = os.path.join(self.recorder.model_dir,
                                  self.recorder.lang_name + "_" + self.recorder.property + "_" +
                                  self.recorder.modality + "_" + self.recorder.condition +
                                  "_run" + str(self.recorder.run_num) + "_epoch-1_seq2seq.pth")

        torch.save(self.seq2seq.state_dict(), model_file)
        print(f"Untrained model stored at {model_file}")

        # at each epoch, display progress bar
        for epoch in tqdm.tqdm(range(hp.n_epochs)):
            # update loss
            train_loss, train_src, train_trg, train_pred = self.train_one_epoch(train_dataloader,
                                                                                hp.teacher_forcing_ratio)
            valid_loss, valid_src, valid_trg, valid_pred = self.evaluate_one_epoch(valid_dataloader)

            # record predictions and prediction correctness
            train_acc = self.recorder.record_pred(epoch, "train", train_src, train_trg, train_pred)
            valid_acc = self.recorder.record_pred(epoch, "valid", valid_src, valid_trg, valid_pred)
            # record accuracy
            self.recorder.record_acc(epoch, "train", train_loss, train_acc)
            self.recorder.record_acc(epoch, "valid", valid_loss, valid_acc)

            print(f"Epoch {epoch} Train Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f} "
                  f"| Train Acc: {train_acc:7.3f}")
            print(f"Epoch {epoch} Valid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f} "
                  f"| Valid Acc: {valid_acc:7.3f}")

            # save model every other save_epochs
            if epoch % hp.save_epochs == 0 or epoch == hp.n_epochs-1:
                model_file = os.path.join(self.recorder.model_dir,
                                          self.recorder.lang_name + "_" + self.recorder.property + "_" +
                                          self.recorder.modality + "_" + self.recorder.condition +
                                          "_run" + str(self.recorder.run_num) + "_epoch" + str(epoch) +
                                          "_seq2seq.pth")
                torch.save(self.seq2seq.state_dict(), model_file)
                print(f"Epoch {epoch} model trained and stored at {model_file}")

        # plot accuracy at the end of training
        utils.plot_txt_acc(self.recorder.acc_store, self.recorder.acc_plot)
        print(f"Run {self.recorder.run_num} training loss, accuracy, and predicted results are saved")

    # a function that completes one repetition of evaluation at the end of training
    def test(self, test_dataloader):
        # load model
        model_file = os.path.join(self.recorder.model_dir,
                                  self.recorder.lang_name + "_" + self.recorder.property + "_" +
                                  self.recorder.modality + "_" + self.recorder.condition +
                                  "_run" + str(self.recorder.run_num) + "_epoch" + str(hp.n_epochs-1) +
                                  "_seq2seq.pth")
        self.seq2seq.load_state_dict(torch.load(model_file))

        # check loss for test dataset
        test_loss, test_srcs, test_trgs, test_preds = self.evaluate_one_epoch(test_dataloader)

        # record predictions and prediction correctness
        test_acc = self.recorder.record_pred(hp.n_epochs, "test", test_srcs, test_trgs, test_preds)
        # record accuracy
        self.recorder.record_acc(hp.n_epochs, "test", test_loss, test_acc)

        print(f"Test Loss: {test_loss:7.3f} | Test PPL: {np.exp(test_loss):7.3f} "
              f"| Test Acc: {test_acc:7.3f}")

        # save accuracy recording to file at the end of testing
        utils.save_to_file(self.recorder.acc_store, self.recorder.acc_file)
        # save prediction recording to file at the end of testing
        utils.save_to_file(self.recorder.pred_store, self.recorder.pred_file)
        print(f"Run {self.recorder.run_num} testing loss, accuracy, and predicted results are saved")

    # a function that manages training at one epoch
    def train_one_epoch(self, data_loader, teacher_forcing_ratio):
        self.seq2seq.train()  # enable dropout in training
        epoch_loss = 0

        # storing predictions
        srcs = []
        trgs = []
        preds = []

        # training in one batch
        for i, (src, trg) in enumerate(data_loader):
            # src = [src_len, batch_size]
            # trg = [trg_len, batch_size]

            self.optimizer.zero_grad()  # reset gradient at each iteration to 0
            output, pred, _ = self.seq2seq(src, trg, teacher_forcing_ratio)
            # output = [trg_len, batch_size, output_dim]
            # pred = [trg_len, batch_size]

            # record predictions
            srcs.append(src)
            trgs.append(trg)
            preds.append(pred)

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

        # average loss over all batches
        epoch_loss = epoch_loss / len(data_loader)
        return epoch_loss, srcs, trgs, preds

    # a function that manages evaluation at one epoch
    def evaluate_one_epoch(self, data_loader):
        self.seq2seq.eval()  # disable dropout in evaluation
        epoch_loss = 0

        # storing predictions
        srcs = []
        trgs = []
        preds = []

        # evaluation in one batch
        with torch.no_grad():  # disable gradient tracking
            for i, (src, trg) in enumerate(data_loader):
                # src = [src_len, batch_size]
                # trg = [trg_len, batch_size]

                output, pred, _ = self.seq2seq(src, trg, 0)  # turn off teacher forcing
                # output = [trg_len, batch_size, output_dim]
                # pred = [trg_len, batch_size]

                # record predictions
                srcs.append(src)
                trgs.append(trg)
                preds.append(pred)

                # remove the <SOS> token from output and target and reshape for loss calculation
                output_dim = output.shape[2]
                output = output[1:].view(-1, output_dim)
                # output = [(trg_len - 1) * batch_size, output_dim]
                trg = trg[1:].view(-1)
                # trg = [(trg_len - 1) * batch_size]

                batch_loss = self.criterion(output, trg)  # calculate batch loss
                epoch_loss += batch_loss.item()  # add to epoch loss

        # average loss over all batches
        epoch_loss = epoch_loss / len(data_loader)
        return epoch_loss, srcs, trgs, preds

    # a function that manages evaluation of one random batch
    def evaluate_attention(self, test_dataloader, eval_epoch=hp.n_epochs-1):
        # get one random batch of test data
        dataiter = iter(test_dataloader)
        src, trg = next(dataiter)
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]

        # load model
        model_file = os.path.join(self.recorder.model_dir,
                                  self.recorder.lang_name + "_" + self.recorder.property + "_" +
                                  self.recorder.modality + "_" + self.recorder.condition +
                                  "_run" + str(self.recorder.run_num) + "_epoch" + str(eval_epoch) +
                                  "_seq2seq.pth")
        self.seq2seq.load_state_dict(torch.load(model_file))
        self.seq2seq.eval()  # disable dropout in evaluation

        with torch.no_grad():  # disable gradient tracking
            # get predicted sr and attention weights
            _, pred, att = self.seq2seq(src, trg, 0)  # turn off teacher forcing
            # pred = [trg_len, batch_size]
            # att = [trg_len, batch_size, src_len]

            for i in range(hp.batch_size):
                ur = src[:, i]
                sr = trg[:, i]
                pred_sr = pred[:, i]

                # convert predictions
                ur_string, _, pred_sr_string = self.recorder.tensor2string(ur, sr, pred_sr)
                ur_list, _, pred_sr_list = self.recorder.tensor2list(ur, sr, pred_sr)

                # retrieve attention weights
                # notice that trg_len and src_len have changed because padding was removed
                src_len = len(ur_list)
                trg_len = len(pred_sr_list)
                word_att = att[:trg_len, i, :src_len]
                # word_att = [trg_len, src_len]

                # plot attention
                att_plot = os.path.join(self.recorder.att_plot_dir,
                                        self.recorder.lang_name + "_" + self.recorder.property + "_" +
                                        self.recorder.modality + "_" + self.recorder.condition +
                                        "_run" + str(self.recorder.run_num) + "_epoch" + str(eval_epoch) + "_" +
                                        ur_string + "_" + pred_sr_string + ".png")
                utils.plot_txt_att(ur_list, pred_sr_list, word_att, att_plot)
            print(f"Run {self.recorder.run_num} attention plots are saved for investigation")

    def evaluate_embedding(self):
        # a dictionary of dictionaries to store all embeddings
        phone_spaces = {}
        # select focus group
        focus = list(self.recorder.language.focus.keys())

        for file_name in os.listdir(self.recorder.model_dir):
            # load model
            model_file = os.path.join(self.recorder.model_dir, file_name)
            self.seq2seq.load_state_dict(torch.load(model_file))

            # retrieve target embedding
            embed = self.seq2seq.decoder.embedding.weight

            # retrieve embedding of all phonemes and focus group
            phone_space = self.recorder.dataset.sr_alphabet.embed2fea(embed)
            phone_spaces[file_name] = phone_space

            embed_file = os.path.join(self.recorder.embed_plot_dir,
                                      file_name.replace("_seq2seq.pth", "_embedding.csv"))
            embed_plot = os.path.join(self.recorder.embed_plot_dir,
                                      file_name.replace("_seq2seq.pth", "_embedding.png"))
            # plot embedding
            utils.plot_embed(phone_space, focus, embed_plot)
            # save embedding recording to file
            utils.save_to_file(phone_space, embed_file)

        # plot embedding
        utils.plot_embed_updated(phone_spaces, focus, self.recorder.embed_plot, self.recorder.focus_embed_plot)
        print(f"Run {self.recorder.run_num} embedding plots and files are saved for investigation")