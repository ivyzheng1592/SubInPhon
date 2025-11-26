# created 2025/11/10
# A class that defines training and evaluation at each epoch of each run
# Data is recorded into dictionary in AudioRecorder

import os
import tqdm
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import utils
import hyper_params as hp


class AudioRun:
    def __init__(self, seq2seq, recorder):

        # condition hyperparameters
        self.seq2seq = seq2seq
        self.recorder = recorder

        # save untrained model
        model_file = os.path.join(self.recorder.model_dir,
                                  self.recorder.lang_name + "_" + self.recorder.condition +
                                  "_run" + str(self.recorder.run_num) + "_epoch-1_seq2seq.pth")
        torch.save(self.seq2seq.state_dict(), model_file)
        print(f"Untrained model stored at {model_file}")

        # optimizer
        self.optimizer = torch.optim.Adam(self.seq2seq.parameters(), lr=hp.learning_rate)

    # a function for loss calculation
    def get_loss(self, output, spec, trg_txt, trg_aud):

        # remove the <SOS> token from output and target and reshape for loss calculation
        txt_dim = output.shape[2]
        output = output[1:].view(-1, txt_dim)
        # output = [(trg_len - 1) * batch_size, txt_dim]
        trg_txt = trg_txt[1:].view(-1)
        # trg = [(trg_len - 1) * batch_size]

        # detect padded 0s from target spectrogram for loss calculation
        weight = torch.where(torch.eq(trg_aud, 0), 0.0, 1.0)

        rec_loss = F.l1_loss(spec, trg_aud, reduction='mean', weight=weight)
        pred_loss = F.cross_entropy(output, trg_txt, ignore_index=hp.special_tokens.index(hp.pad_token))

        return rec_loss, pred_loss

    # a function that completes one repetition of training
    def train(self, train_dataloader, valid_dataloader):

        # at each epoch, display progress bar
        for epoch in tqdm.tqdm(range(hp.n_epochs)):
            # update loss
            train_rec_loss, train_pred_loss, train_src, train_trg, train_pred = (
                self.train_one_epoch(train_dataloader, hp.text_teacher_forcing, hp.audio_teacher_forcing))
            valid_rec_loss, valid_pred_loss, valid_src, valid_trg, valid_pred = (
                self.evaluate_one_epoch(valid_dataloader))

            # record predictions and prediction correctness
            train_acc = self.recorder.record_pred(epoch, "train", train_src, train_trg, train_pred)
            valid_acc = self.recorder.record_pred(epoch, "valid", valid_src, valid_trg, valid_pred)
            # record accuracy
            self.recorder.record_acc(epoch, "train", train_rec_loss, train_pred_loss, train_acc)
            self.recorder.record_acc(epoch, "valid", valid_rec_loss, valid_pred_loss, valid_acc)

            print(f"Epoch {epoch} Train Reconstruction Task Loss: {train_rec_loss:7.3f} "
                  f"| Train Prediction Task Loss: {train_pred_loss:7.3f} "
                  f"| Train Prediction Acc: {train_acc:7.3f}")
            print(f"Epoch {epoch} Valid Reconstruction Task Loss: {valid_rec_loss:7.3f} "
                  f"| Valid Prediction Task Loss: {valid_pred_loss:7.3f} "
                  f"| Valid Prediction Acc: {valid_acc:7.3f}")

            # save model every other save_epochs
            if epoch % hp.save_epochs == 0 or epoch == hp.n_epochs-1:
                model_file = os.path.join(self.recorder.model_dir,
                                          self.recorder.lang_name + "_" + self.recorder.condition +
                                          "_run" + str(self.recorder.run_num) + "_epoch" + str(epoch) +
                                          "_seq2seq.pth")
                torch.save(self.seq2seq.state_dict(), model_file)
                print(f"Epoch {epoch} model trained and stored at {model_file}")

        # plot accuracy at the end of training
        utils.plot_aud_acc(self.recorder.acc_store, self.recorder.acc_plot)
        print(f"Run {self.recorder.run_num} training loss, accuracy, and predicted results are saved")

    # a function that completes one repetition of evaluation at the end of training
    def test(self, test_dataloader):

        # load model
        model_file = os.path.join(self.recorder.model_dir,
                                  self.recorder.lang_name + "_" + self.recorder.condition +
                                  "_run" + str(self.recorder.run_num) + "_epoch" + str(hp.n_epochs-1) +
                                  "_seq2seq.pth")
        self.seq2seq.load_state_dict(torch.load(model_file))

        # check loss for test dataset
        test_rec_loss, test_pred_loss, test_srcs, test_trgs, test_preds = (
            self.evaluate_one_epoch(test_dataloader))

        # record predictions and prediction correctness
        test_acc = self.recorder.record_pred(hp.n_epochs, "test", test_srcs, test_trgs, test_preds)
        # record accuracy
        self.recorder.record_acc(hp.n_epochs, "test", test_rec_loss, test_pred_loss, test_acc)

        print(f"Test Reconstruction Task Loss: {test_rec_loss:7.3f} "
              f"| Test Prediction Task Loss: {test_pred_loss:7.3f} "
              f"| Test Prediction Acc: {test_acc:7.3f}")

        # save accuracy recording to file at the end of testing
        utils.save_to_file(self.recorder.acc_store, self.recorder.acc_file)
        # save prediction recording to file at the end of testing
        utils.save_to_file(self.recorder.pred_store, self.recorder.pred_file)
        print(f"Run {self.recorder.run_num} testing loss, accuracy, and predicted results are saved")

    # a function that manages training at one epoch
    def train_one_epoch(self, data_loader, txt_teacher_forcing, aud_teacher_forcing):
        self.seq2seq.train()  # enable dropout in training
        epoch_pred_loss = 0
        epoch_rec_loss = 0

        # storing text predictions
        src_txts = []
        trg_txts = []
        pred_txts = []

        # training in one batch
        for i, input in enumerate(data_loader):
            src_txt, src_aud, trg_txt, trg_aud = input
            # src_txt = [txt_src_len, batch_size]
            # src_aud = [batch_size, n_channels, freq, aud_src_len]
            # trg_txt = [txt_trg_len, batch_size]
            # trg_aud = [batch_size, n_channels, freq, aud_trg_len]

            self.optimizer.zero_grad()  # reset gradient at each iteration to 0
            output, pred, spec, _, _ = self.seq2seq(input, txt_teacher_forcing, aud_teacher_forcing)
            # output = [txt_trg_len, batch_size, txt_output_dim]
            # pred = [txt_trg_len, batch_size]
            # spec = [aud_trg_len, batch_size, aud_output_dim]

            # record predictions
            src_txts.append(src_txt)
            trg_txts.append(trg_txt)
            pred_txts.append(pred)

            rec_loss, pred_loss = self.get_loss(output, spec, trg_txt, trg_aud)
            batch_loss = rec_loss + pred_loss  # calculate batch loss
            epoch_rec_loss += rec_loss.item()
            epoch_pred_loss += pred_loss.item()  # add to epoch loss
            batch_loss.backward()  # backpropagate loss
            clip_grad_norm_(self.seq2seq.parameters(), max_norm=1.0)
            # clip the gradients to prevent exploding, uncomment if necessary
            self.optimizer.step()  # update the weights

        # average loss over all batches
        epoch_rec_loss = epoch_rec_loss / len(data_loader)
        epoch_pred_loss = epoch_pred_loss / len(data_loader)
        return epoch_rec_loss, epoch_pred_loss, src_txts, trg_txts, pred_txts

    # a function that manages evaluation at one epoch
    def evaluate_one_epoch(self, data_loader):
        self.seq2seq.eval()  # disable dropout in evaluation
        epoch_rec_loss = 0
        epoch_pred_loss = 0

        # storing text predictions
        src_txts = []
        trg_txts = []
        pred_txts = []

        # evaluation in one batch
        with torch.no_grad():  # disable gradient tracking
            for i, input in enumerate(data_loader):
                src_txt, src_aud, trg_txt, trg_aud = input
                # src_txt = [txt_src_len, batch_size]
                # src_aud = [batch_size, n_channels, freq, aud_src_len]
                # trg_txt = [txt_trg_len, batch_size]
                # trg_aud = [batch_size, n_channels, freq, aud_trg_len]

                output, pred, spec, _, _ = self.seq2seq(input, 0, 0)  # turn off teacher forcing
                # output = [txt_trg_len, batch_size, txt_output_dim]
                # pred = [txt_trg_len, batch_size]
                # spec = [aud_trg_len, batch_size, aud_output_dim]

                # record predictions
                src_txts.append(src_txt)
                trg_txts.append(trg_txt)
                pred_txts.append(pred)

                rec_loss, pred_loss = self.get_loss(output, spec, trg_txt, trg_aud)  # calculate batch loss
                epoch_rec_loss += rec_loss.item()
                epoch_pred_loss += pred_loss.item()  # add to epoch loss

        # average loss and accuracy over all batches
        epoch_rec_loss = epoch_rec_loss / len(data_loader)
        epoch_pred_loss = epoch_pred_loss / len(data_loader)
        return epoch_rec_loss, epoch_pred_loss, src_txts, trg_txts, pred_txts

    # a function that manages evaluation of one random batch
    def evaluate_attention(self, test_dataloader, eval_epoch=hp.n_epochs-1):
        # get one random batch of test data
        dataiter = iter(test_dataloader)
        input = next(dataiter)

        # load model
        model_file = os.path.join(self.recorder.model_dir,
                                  self.recorder.lang_name + "_" + self.recorder.condition +
                                  "_run" + str(self.recorder.run_num) + "_epoch" + str(eval_epoch) +
                                  "_seq2seq.pth")
        self.seq2seq.load_state_dict(torch.load(model_file))
        self.seq2seq.eval()  # disable dropout in evaluation

        with torch.no_grad():  # disable gradient tracking
            # get predicted sr and attention weights
            _, pred, spec, txt_atts, aud_atts = self.seq2seq(input, 0, 0)  # turn off teacher forcing
            # pred = [txt_trg_len, batch_size]
            # spec = [batch_size, 1, aud_output_dim, aud_trg_len]
            # txt_atts = [txt_trg_len, batch_size, aud_src_len]
            # aud_atts = [aud_trg_len, batch_size, aud_src_len]

            src_txt, src_aud, trg_txt, trg_aud = input
            # src_txt = [txt_src_len, batch_size]
            # src_aud = [batch_size, n_channels, freq, aud_src_len]
            # trg_txt = [txt_trg_len, batch_size]
            # trg_aud = [batch_size, n_channels, freq, aud_trg_len]

            for i in range(hp.batch_size):
                ur_txt = src_txt[:, i]
                sr_txt = trg_txt[:, i]
                pred_sr_txt = pred[:, i]

                # convert predictions
                ur_string, _, pred_sr_string = self.recorder.tensor2string(ur_txt, sr_txt, pred_sr_txt)
                _, _, pred_sr_list = self.recorder.tensor2list(ur_txt, sr_txt, pred_sr_txt)

                # retrieve spectrogram and attention weights
                ur_spec = src_aud[i, :, :, :][0]
                # ur_spec = [n_freq, dur]
                pred_sr_spec = spec[i, :, :, :][0]
                # pred_sr_spec = [n_freq, dur]
                txt_att = txt_atts[:, i, :]
                # txt_att = [txt_trg_len, aud_src_len]
                aud_att = aud_atts[:, i, :]
                # aud_att = [aud_trg_len, aud_src_len]

                # plot attention
                att_plot = os.path.join(self.recorder.att_plot_dir,
                                        self.recorder.lang_name + "_" + self.recorder.condition +
                                        "_run" + str(self.recorder.run_num) + "_epoch" + str(eval_epoch) + "_" +
                                        ur_string + "_" + pred_sr_string + ".png")
                utils.plot_aud_att(ur_spec, pred_sr_list, pred_sr_spec, txt_att, aud_att, att_plot)
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