# created 2025/03/03 structure based on Ben Trevett tutorial
# updated 2025/03/07 including model specific implementations
# updated 2025/03/17 added accuracy recording and visualization
# updated 2025/04/01 added src, trg, attention visualization and upgraded to class
# A script that defines training and evaluation at each epoch of each repetition

import os
import csv
import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import hyper_params as hp

class TextRun():
    def __init__(self, seq2seq, trial_num, language, datatype, condition, rep):

        # condition hyperparameters
        self.seq2seq = seq2seq
        self.trial_num = trial_num
        self.language = language
        self.datatype = datatype
        self.condition = condition
        self.rep = rep

        # results files
        self.acc_file = os.path.join("Results", trial_num,
                                     language + "_" + datatype + "_" + condition +
                                     "_run" + str(rep) + "_acc.csv")
        self.model_file = os.path.join("Results", trial_num,
                                       language + "_" + datatype + "_" + condition +
                                       "_run" + str(rep) + "_seq2seq.pth")
        self.acc_plot = os.path.join("Results", trial_num,
                                     language + "_" + datatype + "_" + condition +
                                     "_run" + str(rep) + "_acc_plot.png")

        # model weight initialization
        for name, param in self.seq2seq.named_parameters():
            if "weight" in name:  # weights
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:  # biases
                nn.init.constant_(param.data, 0)
        model_parameters = sum(p.numel() for p in self.seq2seq.parameters() if p.requires_grad)
        print(f"The model has {model_parameters} trainable parameters")

        # optimizer and loss function
        self.optimizer = torch.optim.Adam(self.seq2seq.parameters(), lr=hp.learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=hp.special_tokens.index(hp.pad_token))

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
        self.plot_acc()
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
    def evaluate_one_batch(self, test_dataloader):
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

            # compare the actual and predicted target surface form
            print(f"UR: {src}")
            print(f"Actual SR: {trg}")
            print(f"Predicted SR: {pred}")

            # plot attention
            self.plot_att(src, trg, att[: len(trg) - 1])

    # a function that manages training at one epoch
    def train_one_epoch(self, data_loader, teacher_forcing_ratio):
        self.seq2seq.train()  # enable dropout in training
        epoch_loss = 0
        epoch_acc = 0

        # training in one batch
        for i, (src, trg) in enumerate(data_loader):
            # src = [src_len, batch_size]
            # trg = [trg_len, batch_size]

            self.optimizer.zero_grad()  # reset gradient at each iteration to 0
            output, pred, _ = self.seq2seq(src, trg, teacher_forcing_ratio)
            # output = [trg_length, batch_size, output_dim]
            # pred = [trg_len, batch_size]

            batch_size = trg.shape[1]
            # calculate total number of correct predictions in a batch
            batch_correct = torch.all(torch.eq(pred, trg), dim=0).sum()
            batch_acc = batch_correct.item() / batch_size  # calculate batch accuracy rate
            epoch_acc += batch_acc  # add to epoch accuracy rate

            # remove the <SOS> token from output and target and reshape for loss calculation
            output_dim = output.shape[2]
            output = output[1:].view(-1, output_dim)
            # output = [(trg_length - 1) * batch_size, output_dim]
            trg = trg[1:].view(-1)
            # trg = [(trg_length - 1) * batch_size]

            batch_loss = self.criterion(output, trg)  # calculate batch loss
            epoch_loss += batch_loss.item()  # add to epoch loss
            batch_loss.backward()  # backpropagate loss
            # nn.utils.clip_grad_norm_(model.parameters(), clip)
            # clip the gradients to prevent exploding, uncomment if necessary
            self.optimizer.step()  # update the weights

        # average loss and accuracy over all batches
        average_loss = epoch_loss / len(data_loader)
        average_acc = epoch_acc / len(data_loader)

        return average_loss, average_acc

    # a function that manages evaluation at one epoch
    def evaluate_one_epoch(self, data_loader):
        self.seq2seq.eval()  # disable dropout in evaluation
        epoch_loss = 0
        epoch_acc = 0

        # evaluation in one batch
        with torch.no_grad():  # disable gradient tracking
            for i, (src, trg) in enumerate(data_loader):
                # src = [src_len, batch_size]
                # trg = [trg_len, batch_size]

                output, pred, _ = self.seq2seq(src, trg, 0)  # turn off teacher forcing
                # output = [trg_length, batch_size, output_dim]
                # pred = [trg_len, batch_size]

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
        average_loss = epoch_loss / len(data_loader)
        average_acc = epoch_acc / len(data_loader)

        return average_loss, average_acc

    def record_acc(self, epoch, record_type, loss, acc):
        if os.path.exists(self.acc_file):
            # open csv file in append mode
            with open(self.acc_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                data = [self.language, self.datatype, self.condition, self.rep,
                        epoch, record_type, loss, acc]
                writer.writerow(data)
        else:
            # open csv file in write mode and add header
            with open(self.acc_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                header = ['language', 'datatype', 'condition', 'run', 'epoch',
                          'record_type', 'loss', 'acc']
                writer.writerow(header)
                data = [self.language, self.datatype, self.condition, self.rep,
                        epoch, record_type, loss, acc]
                writer.writerow(data)

    def plot_acc(self):
        acc_data = pd.read_csv(self.acc_file)
        train_data = acc_data[acc_data["test_type"] == "train"]
        valid_data = acc_data[acc_data["test_type"] == "valid"]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all')  # create a 2 * 1 plot
        ax1.plot(train_data["epoch"], train_data["loss"], label="train")
        ax1.plot(valid_data["epoch"], valid_data["loss"], label="valid")
        ax1.legend()
        ax1.set_title("Loss")
        ax2.plot(train_data["epoch"], train_data["acc"], label="train")
        ax2.plot(valid_data["epoch"], valid_data["acc"], label="valid")
        ax2.legend()
        ax2.set_title("Acc")

        plt.savefig(self.acc_plot)
        plt.show()

    def plot_att(self, src, trg, attention):
        fig, ax = plt.subplots(figsize=(10, 10))
        attention = attention.squeeze(1).numpy()
        cax = ax.matshow(attention, cmap="bone")
        ax.set_xticks(ticks=np.arange(len(src)), labels=src, rotation=90, size=15)
        translation = trg[1:]
        ax.set_yticks(ticks=np.arange(len(translation)), labels=translation, size=15)
        plt.show()
        plt.close()