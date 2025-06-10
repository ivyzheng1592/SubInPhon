# created 2025/03/07 structure based on Ben Trevett tutorial
# updated 2025/04/01 upgraded to class
# A script that defines training and evaluation at each epoch of each run

import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
import utils
import hyper_params as hp


class TextRun:
    def __init__(self, seq2seq, recorder, run_num):

        # condition hyperparameters
        self.seq2seq = seq2seq
        self.recorder = recorder
        self.run_num = run_num

        self.model_dir = os.path.join("Results", recorder.trial_num + "_" + recorder.lang_name,
                                      recorder.lang_name + "_" + recorder.condition +
                                      "_run" + str(self.run_num) + "_model_files")
        os.makedirs(self.model_dir, exist_ok=True)

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

            # record accuracy
            self.recorder.record_acc(self.run_num, epoch, "train", train_loss, train_acc)
            self.recorder.record_acc(self.run_num, epoch, "valid", valid_loss, valid_acc)

            # save model every other save_epochs
            if epoch % hp.save_epochs or epoch == hp.n_epochs-1:
                model_file = os.path.join(self.model_dir,
                                          self.recorder.lang_name + "_" + self.recorder.condition +
                                          "_run" + str(self.run_num) + "_epoch" + str(epoch) +
                                          "_seq2seq.pth")
                torch.save(self.seq2seq.state_dict(), model_file)
                print(f"Epoch {epoch} model trained and stored at {model_file}")

        # save accuracy recording to file
        utils.save_to_file(self.recorder.acc_store, self.recorder.acc_file)
        # plot accuracy at the end of training
        acc_plot = os.path.join(self.recorder.acc_plot_dir,
                                self.recorder.lang_name + "_" + self.recorder.condition +
                                "_run" + self.run_num + "_acc_plot.png")
        utils.plot_acc(self.recorder.acc_store, acc_plot)
        # empty accuracy recording
        for key, value in self.recorder.acc_store:
            value.clear()

        # save prediction recording to file
        utils.save_to_file(self.recorder.pred_store, self.recorder.pred_file)
        # empty prediction recording
        for key, value in self.recorder.pred_store:
            value.clear()

        print(f"Run {self.run_num} training loss, accuracy, and predicted results are saved")

    # a function that completes one repetition of evaluation at the end of training
    def test(self, test_dataloader):
        # load model
        model_file = os.path.join(self.model_dir,
                                  self.recorder.lang_name + "_" + self.recorder.condition +
                                  "_run" + str(self.run_num) + "_epoch" + str(hp.n_epochs-1) +
                                  "_seq2seq.pth")
        self.seq2seq.load_state_dict(torch.load(model_file))

        # check loss for test dataset
        test_loss, test_acc = self.evaluate_one_epoch(hp.n_epochs, "test", test_dataloader)
        print(f"Test Loss: {test_loss:7.3f} | Test PPL: {np.exp(test_loss):7.3f} "
              f"| Test Acc: {test_acc:7.3f}")

        # record accuracy
        self.recorder.record_acc(self.run_num, hp.n_epochs, "test", test_loss, test_acc)

        # save accuracy recording to file
        utils.save_to_file(self.recorder.acc_store, self.recorder.acc_file)
        # empty accuracy recording
        for key, value in self.recorder.acc_store:
            value.clear()

        # save prediction recording to file
        utils.save_to_file(self.recorder.pred_store, self.recorder.pred_file)
        # empty prediction recording
        for key, value in self.recorder.pred_store:
            value.clear()

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
            output, pred, src_embed, trg_embed, att = self.seq2seq(src, trg, teacher_forcing_ratio)
            # output = [trg_len, batch_size, output_dim]
            # pred = [trg_len, batch_size]

            # record predictions
            self.recorder.record_pred(self.run_num, epoch, record_type, src, trg, pred)

            # convert predictions to string for accuracy calculation
            batch_correct = []
            for j in range(hp.batch_size):
                ur = src[:, j]
                sr = trg[:, j]
                pred_sr = pred[:, j]
                _, sr_string, pred_sr_string = self.recorder.tensor2string(ur, sr, pred_sr)
                if sr_string == pred_sr_string:
                    batch_correct.insert(j, 1)
                else:
                    batch_correct.insert(j, 0)
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

                output, pred, src_embed, trg_embed, att = self.seq2seq(src, trg, 0)  # turn off teacher forcing
                # output = [trg_len, batch_size, output_dim]
                # pred = [trg_len, batch_size]

                # record predictions
                self.recorder.record_pred(self.run_num, epoch, record_type, src, trg, pred)

                # convert predictions to string for accuracy calculation
                batch_correct = []
                for j in range(hp.batch_size):
                    ur = src[:, j]
                    sr = trg[:, j]
                    pred_sr = pred[:, j]
                    _, sr_string, pred_sr_string = self.recorder.tensor2string(ur, sr, pred_sr)
                    if sr_string == pred_sr_string:
                        batch_correct.insert(j, 1)
                    else:
                        batch_correct.insert(j, 0)
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
    def evaluate_one_batch(self, test_dataloader, eval_epoch=hp.n_epochs-1, eval_type="both"):
        # get one random batch of test data
        dataiter = iter(test_dataloader)
        src, trg = next(dataiter)
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]

        # load model
        model_file = os.path.join(self.model_dir,
                                  self.recorder.lang_name + "_" + self.recorder.condition +
                                  "_run" + str(self.run_num) + "_epoch" + str(hp.n_epochs - 1) +
                                  "_seq2seq.pth")
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

                # convert predictions
                ur_string, sr_string, _ = self.recorder.tensor2string(ur, sr, pred_sr)
                ur_list, sr_list, pred_sr_list = self.recorder.tensor2list(ur, sr, pred_sr)
                _, _, pred_sr_sylls = self. recorder.tensor2syll(ur, sr, pred_sr)

                # retrieve attention weights
                # notice that trg_len and src_len have changed because padding was removed
                src_len = len(ur_list)
                trg_len = len(sr_list)
                word_att = att[:trg_len, i, :src_len]
                # word_att = [trg_len, src_len]

                # record attention type and embedding
                if eval_type != "embedding":
                    self.recorder.record_att(i, pred_sr_sylls, word_att, ur_string, sr_string)
                    att_plot = os.path.join(self.recorder.att_plot_dir,
                                            self.recorder.lang_name + "_" + self.recorder.condition +
                                            "_run" + str(self.run_num) + "_epoch" + str(eval_epoch) + "_" +
                                            ur_string + "_" + sr_string + ".png")
                    utils.plot_att(ur_list, pred_sr_list, word_att, att_plot)
                if eval_type != "attention":
                    self.recorder.record_embed(i, ur_list, pred_sr_list, src_embed, trg_embed)

            if eval_type != "embedding":
                # save attention recording to file
                att_file = os.path.join(self.recorder.att_plot_dir,
                                        self.recorder.lang_name + "_" + self.recorder.condition +
                                        "_run" + str(self.run_num) + "_epoch" + str(eval_epoch) +
                                        "_att_type.csv")
                utils.save_to_file(self.recorder.att_store, att_file)
                # empty attention recording
                for key, value in self.recorder.att_store:
                    value.clear()
                print(f"Run {self.run_num} attention plots and types are saved for investigation")

            if eval_type != "attention":
                # plot embedding
                ur_embed_phone_plot = os.path.join(self.recorder.embed_plot_dir,
                                                   self.recorder.lang_name + "_" + self.recorder.condition +
                                                   "_run" + str(self.run_num) + "_epoch" + str(eval_epoch) +
                                                   "_ur_phoneme.png")
                ur_embed_phone_file = os.path.join(self.recorder.embed_plot_dir,
                                                   self.recorder.lang_name + "_" + self.recorder.condition +
                                                   "_run" + str(self.run_num) + "_epoch" + str(eval_epoch) +
                                                   "_ur_phoneme.csv")
                ur_embed_vowel_plot = os.path.join(self.recorder.embed_plot_dir,
                                                   self.recorder.lang_name + "_" + self.recorder.condition +
                                                   "_run" + str(self.run_num) + "_epoch" + str(eval_epoch) +
                                                   "_ur_vowel.png")
                ur_embed_vowel_file = os.path.join(self.recorder.embed_plot_dir,
                                                   self.recorder.lang_name + "_" + self.recorder.condition +
                                                   "_run" + str(self.run_num) + "_epoch" + str(eval_epoch) +
                                                   "_ur_vowel.csv")
                sr_embed_phone_plot = os.path.join(self.recorder.embed_plot_dir,
                                                   self.recorder.lang_name + "_" + self.recorder.condition +
                                                   "_run" + str(self.run_num) + "_epoch" + str(eval_epoch) +
                                                   "_sr_phoneme.png")
                sr_embed_phone_file = os.path.join(self.recorder.embed_plot_dir,
                                                   self.recorder.lang_name + "_" + self.recorder.condition +
                                                   "_run" + str(self.run_num) + "_epoch" + str(eval_epoch) +
                                                   "_sr_phoneme.csv")
                sr_embed_vowel_plot = os.path.join(self.recorder.embed_plot_dir,
                                                   self.recorder.lang_name + "_" + self.recorder.condition +
                                                   "_run" + str(self.run_num) + "_epoch" + str(eval_epoch) +
                                                   "_sr_vowel.png")
                sr_embed_vowel_file = os.path.join(self.recorder.embed_plot_dir,
                                                   self.recorder.lang_name + "_" + self.recorder.condition +
                                                   "_run" + str(self.run_num) + "_epoch" + str(eval_epoch) +
                                                   "_sr_vowel.csv")

                try:
                    # code that might raise a runtime error
                    utils.plot_embed(self.recorder.ur_embed_phone_store,
                                     ur_embed_phone_file, ur_embed_phone_plot,
                                     "phoneme embedding")
                    utils.plot_embed(self.recorder.ur_embed_vowel_store,
                                     ur_embed_vowel_file, ur_embed_vowel_plot,
                                     "vowel embedding")
                    utils.plot_embed(self.recorder.sr_embed_phone_store,
                                     sr_embed_phone_file, sr_embed_phone_plot,
                                     "phoneme embedding")
                    utils.plot_embed(self.recorder.sr_embed_vowel_store,
                                     sr_embed_vowel_file, sr_embed_vowel_plot,
                                     "vowel embedding")
                except Exception as e:
                    print(f"The error {e} occurred in run {self.run_num}. Continue running ...")
                    self.evaluate_one_batch(test_dataloader, eval_epoch=eval_epoch, eval_type="embedding")
                print(f"Run {self.run_num} embedding plots and files are saved for investigation")
