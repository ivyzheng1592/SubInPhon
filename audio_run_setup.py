# created 2025/04/01 incorporating audio input
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

    plt.savefig(plot_file)
    plt.show()


# attention plotting
def plot_att():
    pass


# a function that manages training at one epoch
def train_one_epoch(model, data_loader, optimizer, criterion, teacher_forcing_ratio):
    model.train()  # enable dropout in training
    epoch_loss = 0
    epoch_acc = 0

    # training in one batch
    for i, (src, trg) in enumerate(data_loader):
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
        #nn.utils.clip_grad_norm_(model.parameters(), clip) # clip the gradients to prevent exploding
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


# a function that completes one repetition of training and evaluation of one model
def audio_run(seq2seq, train_dataloader, valid_dataloader, test_dataloader,
              trial_num, language, datatype, condition, rep_num):
    # results files
    acc_file = os.path.join("Results", trial_num,
                            language + "_" + datatype + "_" + condition + "_run" + str(rep_num) + "_acc.csv")
    model_file = os.path.join("Results", trial_num,
                              language + "_" + datatype + "_" + condition + "_run" + str(rep_num) + "_seq2seq.pth")
    acc_plot = os.path.join("Results", trial_num,
                            language + "_" + datatype + "_" + condition + "_run" + str(rep_num) + "_acc_plot.png")

    # model weight initialization
    seq2seq.apply(init_weights)
    model_parameters = sum(p.numel() for p in seq2seq.parameters() if p.requires_grad)
    print(f"The model has {model_parameters} trainable parameters")

    # optimizer and loss function
    optimizer = torch.optim.Adam(seq2seq.parameters(), lr=hp.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=hp.special_tokens.index(hp.pad_token))

    # at each epoch, display progress bar
    for epoch in tqdm.tqdm(range(hp.n_epochs)):
        # update loss for each batch
        train_loss, train_acc = train_one_epoch(seq2seq, train_dataloader, optimizer, criterion,
                                                hp.teacher_forcing_ratio)
        valid_loss, valid_acc = evaluate_one_epoch(seq2seq, valid_dataloader, criterion)
        print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f} | Train Acc: {train_acc:7.3f}")
        print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f} | Valid Acc: {valid_acc:7.3f}")

        # save the accuracy value
        record_acc(acc_file, language, datatype, condition, rep_num, epoch,
                   "train", train_loss, train_acc)
        record_acc(acc_file, language, datatype, condition, rep_num, epoch,
                   "valid", valid_loss, valid_acc)
        print(f"Accuracy data saved at {acc_file}")

        # save the model
        torch.save(seq2seq.state_dict(), model_file)
        print(f"Model trained and stored at {model_file}")

    # plot loss and accuracy for training and validation dataset
    plot_acc(acc_file, acc_plot)
    print(f"Accuracy plot save at {acc_plot}")

    # load the model
    seq2seq.load_state_dict(torch.load(model_file))

    # check loss for the test dataset
    test_loss, test_acc = evaluate_one_epoch(seq2seq, test_dataloader, criterion)
    print(f"\tTest Loss: {test_loss:7.3f} | Test PPL: {np.exp(test_loss):7.3f} | Test Acc: {test_acc:7.3f}")
    # save the accuracy value
    record_acc(acc_file, language, datatype, condition, rep_num, hp.n_epochs,
               "test", test_loss, test_acc)
    print(f"Accuracy data saved at {acc_file}")

    # randomly select 10 datapoints for