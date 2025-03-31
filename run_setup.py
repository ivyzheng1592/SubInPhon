# created 2025/03/03 structure based on Ben Trevett tutorial
# updated 2025/03/07 including model specific implementations
# updated 2025/03/17 including accuracy recording and visualization
# A script that defines training and evaluation at each epoch of each run

import os
import csv
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# weight initialization
def init_weights(model):
    for name, param in model.named_parameters():
        if "weight" in name:  # weights
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:  # biases
            nn.init.constant_(param.data, 0)


# accuracy recoding
def record_acc(acc_file, language, datatype, condition, run, epoch,
               test_type, loss, acc):
    if os.path.exists(acc_file):
        # open csv file in append mode
        with open(acc_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            data = [language, datatype, condition, run, epoch, test_type, loss, acc]
            writer.writerow(data)
    else:
        # open csv file in write mode and add header
        with open(acc_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['language', 'datatype', 'condition', 'run', 'epoch',
                      'test_type', 'loss', 'acc']
            writer.writerow(header)
            data = [language, datatype, condition, run, epoch, test_type, loss, acc]
            writer.writerow(data)


# learning curve plotting
def plot_acc(acc_file, plot_file):
    acc_data = pd.read_csv(acc_file)
    epoch = acc_data["epoch"]
    train_loss = acc_data["train_loss"]
    train_acc = acc_data["train_acc"]
    valid_loss = acc_data["valid_loss"]
    valid_acc = acc_data["valid_acc"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='all', sharey='row')  # create a 2 * 2 plot
    ax1.plot(epoch, train_loss)
    ax1.set_title("Train loss")
    ax2.plot(epoch, train_acc)
    ax2.set_title("Train acc")
    ax3.plot(epoch, valid_loss)
    ax2.set_title("Valid loss")
    ax4.plot(epoch, valid_acc)
    ax4.set_title("Valid acc")

    plt.savefig(plot_file)
    plt.show()


# attention plotting
def plot_att():
    pass


# a function that manages training at one epoch
def train_one_epoch(model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio):
    model.train()  # enable dropout in training
    epoch_loss = 0
    epoch_acc = 0

    # training in one batch
    for i, (src, trg) in enumerate(data_loader):
        #src = src.to(device)
        #trg = trg.to(device)
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]

        optimizer.zero_grad()  # reset gradient at each iteration to 0
        output, pred = model(src, trg, teacher_forcing_ratio)
        # output = [trg_length, batch_size, output_dim]
        # pred = [trg_len, batch_size]

        batch_size = trg.shape[1]
        batch_correct = torch.all(torch.eq(pred, trg), dim=0).sum()  # calculate total number of correct predictions in a batch
        batch_acc = batch_correct.item() / batch_size # calculate batch accuracy rate
        epoch_acc += batch_acc  # add to epoch accuracy rate

        # remove the <SOS> token from output and target and reshape for loss calculation
        output_dim = output.shape[2]
        output = output[1:].view(-1, output_dim)
        # output = [(trg_length - 1) * batch_size, output_dim]
        trg = trg[1:].view(-1)
        # trg = [(trg_length - 1) * batch_size]

        batch_loss = criterion(output, trg)  # calculate batch loss
        epoch_loss += batch_loss.item()  # add to epoch loss
        batch_loss.backward()  # backpropagate loss
        nn.utils.clip_grad_norm_(model.parameters(), clip) # clip the gradients to prevent exploding
        optimizer.step()  # update the weights

    # average loss and accuracy over all batches
    average_loss = epoch_loss / len(data_loader)
    average_acc = epoch_acc / len(data_loader)

    return average_loss, average_acc


# a function that manages evaluation at one epoch
def evaluate_one_epoch(model, data_loader, criterion):
    model.eval()  # disable dropout in evaluation
    epoch_loss = 0
    epoch_acc = 0

    # evaluation in one batch
    with torch.no_grad():  # disable gradient tracking
        for i, (src, trg) in enumerate(data_loader):
            #src = src.to(device)
            #trg = trg.to(device)
            # src = [src_len, batch_size]
            # trg = [trg_len, batch_size]

            output, pred = model(src, trg, 0)  # turn off teacher forcing
            # output = [trg_length, batch_size, output_dim]
            # pred = [trg_len, batch_size]

            batch_size = trg.shape[1]
            batch_correct = torch.all(torch.eq(pred, trg), dim=0).sum()  # calculate total number of correct predictions in a batch
            batch_acc = batch_correct.item() / batch_size  # calculate batch accuracy rate
            epoch_acc += batch_acc  # add to epoch accuracy rate

            # remove the <SOS> token from output and target and reshape for loss calculation
            output_dim = output.shape[2]
            output = output[1:].view(-1, output_dim)
            # output = [(trg_len - 1) * batch_size, output_dim]
            trg = trg[1:].view(-1)
            # trg = [(trg_len - 1) * batch_size, output_dim]

            batch_loss = criterion(output, trg)  # calculate batch loss
            epoch_loss += batch_loss.item()  # add to epoch loss

    # average loss and accuracy over all batches
    average_loss = epoch_loss / len(data_loader)
    average_acc = epoch_acc / len(data_loader)

    return average_loss, average_acc