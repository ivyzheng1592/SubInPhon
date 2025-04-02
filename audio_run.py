# created 2025/04/01
# A script that defines training and evaluation at each epoch of each run

import os
import csv
import tqdm
import numpy as np
import torch
import torch.nn as nn
import hyper_params as hp
import utils


class AudioRun:
    def __init__(self, seq2seq, trial_num, language, datatype, condition, run_num):

        # condition hyperparameters
        self.seq2seq = seq2seq
        self.trial_num = trial_num
        self.language = language
        self.datatype = datatype
        self.condition = condition
        self.run_num = run_num

        # results files
        self.acc_file = os.path.join("Results", trial_num,
                                     language + "_" + datatype + "_" + condition +
                                     "_run" + str(run_num) + "_acc.csv")
        self.model_file = os.path.join("Results", trial_num,
                                       language + "_" + datatype + "_" + condition +
                                       "_run" + str(run_num) + "_seq2seq.pth")
        self.acc_plot = os.path.join("Results", trial_num,
                                     language + "_" + datatype + "_" + condition +
                                     "_run" + str(run_num) + "_acc_plot.png")
        self.att_plot_dir = os.path.join("Results", trial_num,
                                         language + "_" + datatype + "_" + condition +
                                         "_run" + str(run_num) + "_attention_plots")

        # model weight initialization
        for name, param in self.seq2seq.named_parameters():
            if "weight" in name:  # weights
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:  # biases
                nn.init.constant_(param.data, 0)
        #model_parameters = sum(p.numel() for p in self.seq2seq.parameters() if p.requires_grad)

        # optimizer and loss function
        self.optimizer = torch.optim.Adam(self.seq2seq.parameters(), lr=hp.learning_rate)
        self.criterion = nn.MSELoss(reduction='mean')

    # a function that completes one repetition of training
    def train(self, train_dataloader, valid_dataloader):

        # at each epoch, display progress bar
        for epoch in tqdm.tqdm(range(hp.n_epochs)):
            # update loss for each batch
            train_loss, train_acc = self.train_one_epoch(train_dataloader, hp.teacher_forcing_ratio)
            valid_loss, valid_acc = self.evaluate_one_epoch(valid_dataloader)
            print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f} "
                  f"| Train Acc: {train_acc:7.3f}")
            print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f} "
                  f"| Valid Acc: {valid_acc:7.3f}")

            # save the accuracy value
            self.record_acc(epoch, "train", train_loss, train_acc)
            self.record_acc(epoch, "valid", valid_loss, valid_acc)
            print(f"Accuracy data saved at {self.acc_file}")

            # save the model
            torch.save(self.seq2seq.state_dict(), self.model_file)
            print(f"Model trained and stored at {self.model_file}")

        # plot loss and accuracy for training and validation dataset
        utils.plot_acc(self.acc_file, self.acc_plot)
        print(f"Accuracy plot save at {self.acc_plot}")

    # a function that completes one repetition of evaluation at the end of training
    def test(self, test_dataloader):
        # load the model
        self.seq2seq.load_state_dict(torch.load(self.model_file))

        # check loss for the test dataset
        test_loss, test_acc = self.evaluate_one_epoch(test_dataloader)
        print(f"\tTest Loss: {test_loss:7.3f} | Test PPL: {np.exp(test_loss):7.3f} "
              f"| Test Acc: {test_acc:7.3f}")
        # save the accuracy value
        self.record_acc(hp.n_epochs, "test", test_loss, test_acc)
        print(f"Accuracy data saved at {self.acc_file}")

    # a function that manages evaluation of one random batch
    def evaluate_one_batch(self, data_loader, dataset):
        # get one random batch of test data
        dataiter = iter(data_loader)
        (src_txt, src_aud), (trg_txt, trg_aud) = next(dataiter)
        # src_txt = [src_len, batch_size]
        # src_aud = [batch_size, n_channels, n_freq, n_samples]
        # trg_txt = [trg_len, batch_size]
        # trg_aud = [batch_size, n_channels, n_freq, n_samples]

        # load the model
        self.seq2seq.load_state_dict(torch.load(self.model_file))
        self.seq2seq.eval()  # disable dropout in evaluation
        with torch.no_grad():  # disable gradient tracking
            pred, att = self.seq2seq(src_aud, trg_aud, 0)  # turn off teacher forcing
            # pred = [trg_len, batch_size, n_channels, output_dim]
            # att = [trg_len, batch_size, src_len]

            # for individual items
            for i in range(hp.batch_size):
                ur_tensor = src_aud[i, :, :, :]
                sr_tensor = trg_aud[i, :, :, :]
                pred_sr_tensor = pred[:, i]
                att_tensor = att[:, i, :]
                # attention = [trg_len, src_len]

                # convert tensor to vector
                ur_vector = [int(x) for x in ur_tensor.tolist()]
                sr_vector = [int(x) for x in sr_tensor.tolist()]
                pred_sr_vector = [int(x) for x in pred_sr_tensor.tolist()]

                # convert vector to word
                ur_word, ur_string = dataset.ur_alphabet.vec2word(ur_vector)
                sr_word, sr_string = dataset.sr_alphabet.vec2word(sr_vector)
                pred_sr_word, pred_sr_string = dataset.sr_alphabet.vec2word(pred_sr_vector)

                # compare the actual and predicted target surface form
                print(f"UR: {ur_string}")
                print(f"Actual SR: {sr_string}")
                print(f"Predicted SR: {pred_sr_string}")

                # plot attention
                att_plot = os.path.join(self.att_plot_dir,
                                        ur_string + "_" + pred_sr_string + ".png")
                utils.plot_att(ur_word, pred_sr_word, att_tensor, att_plot)

    # a function that manages training at one epoch
    def train_one_epoch(self, data_loader, teacher_forcing_ratio):
        self.seq2seq.train()  # enable dropout in training
        epoch_loss = 0
        epoch_mse = 0

        # training in one batch
        for i, ((src_txt, src_aud), (trg_txt, trg_aud)) in enumerate(data_loader):
            # src_txt = [src_len, batch_size]
            # src_aud = [batch_size, n_channels, n_freq, src_len]
            # trg_txt = [trg_len, batch_size]
            # trg_aud = [batch_size, n_channels, n_freq, trg_len]

            self.optimizer.zero_grad()  # reset gradient at each iteration to 0
            output, _ = self.seq2seq(src_aud, trg_aud, teacher_forcing_ratio)
            # output = [batch_size, n_channels, output_dim, trg_len]

            batch_mse = nn.functional.mse_loss(output, trg_aud, reduction='mean')  # calculate batch mse
            epoch_mse += batch_mse.item()  # add to epoch mse

            batch_loss = self.criterion(output, trg_aud)  # calculate batch loss
            epoch_loss += batch_loss.item()  # add to epoch loss
            batch_loss.backward()  # backpropagate loss
            # nn.utils.clip_grad_norm_(model.parameters(), clip)
            # clip the gradients to prevent exploding, uncomment if necessary
            self.optimizer.step()  # update the weights

        # average loss and accuracy over all batches
        average_loss = epoch_loss / len(data_loader)
        average_mse = epoch_mse / len(data_loader)

        return average_loss, average_mse

    # a function that manages evaluation at one epoch
    def evaluate_one_epoch(self, data_loader):
        self.seq2seq.eval()  # disable dropout in evaluation
        epoch_loss = 0
        epoch_mse = 0

        # evaluation in one batch
        with torch.no_grad():  # disable gradient tracking
            for i, ((src_txt, src_aud), (trg_txt, trg_aud)) in enumerate(data_loader):
                # src_txt = [src_len, batch_size]
                # src_aud = [batch_size, n_channels, n_freq, trg_len]
                # trg_txt = [trg_len, batch_size]
                # trg_aud = [batch_size, n_channels, n_freq, trg_len]

                output, _ = self.seq2seq(src_aud, trg_aud, 0)  # turn off teacher forcing
                # output = [batch_size, n_channels, output_dim, trg_len]

                batch_mse = nn.functional.mse_loss(output, trg_aud, reduction='mean')  # calculate batch mse
                epoch_mse += batch_mse.item()  # add to epoch mse

                batch_loss = self.criterion(output, trg_aud)  # calculate batch loss
                epoch_loss += batch_loss.item()  # add to epoch loss
                self.optimizer.step()  # update the weights

        # average loss and accuracy over all batches
        average_loss = epoch_loss / len(data_loader)
        average_mse = epoch_mse / len(data_loader)

        return average_loss, average_mse

    def record_acc(self, epoch, record_type, loss, acc):
        if os.path.exists(self.acc_file):
            # open csv file in append mode
            with open(self.acc_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                data = [self.language, self.datatype, self.condition, self.run_num,
                        epoch, record_type, loss, acc]
                writer.writerow(data)
        else:
            # open csv file in write mode and add header
            with open(self.acc_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                header = ['language', 'datatype', 'condition', 'run',
                          'epoch', 'record_type', 'loss', 'acc']
                writer.writerow(header)
                data = [self.language, self.datatype, self.condition, self.run_num,
                        epoch, record_type, loss, acc]
                writer.writerow(data)