# created 2025/03/03 structure based on Ben Trevett tutorial
# updated 2025/03/07 including model specific implementations
# updated 2025/03/17 including accuracy recording and visualization
# A script that defines training and evaluation at each epoch of each run

import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
from torch.nn.init import normal_, constant_
from torch.utils.data import random_split

from audio_dataset_loader import AudioDataset
from text_dataset_loader import TextDataset, get_dataloader
from network import Encoder, Decoder, BahdanauAttention, Seq2Seq
import hyper_params as hp


# weight initialization
def init_weights(model):
    for name, param in model.named_parameters():
        if "weight" in name:  # weights
            normal_(param.data, mean=0, std=0.01)
        else:  # biases
            constant_(param.data, 0)


# accuracy recoding
def record_acc(acc_file, datatype, condition, run, epoch, train_loss, train_acc, valid_loss, valid_acc):
    if os.path.exists(acc_file):
        # open csv file in append mode
        with open(acc_file, mode='a', newline='') as file:
            writer = csv.writer(file)
    else:
        # open csv file in write mode and add header
        with open(acc_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['datatype', 'condition', 'run', 'epoch', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc']
            writer.writerow(header)

    # append acc data
    data = [datatype, condition, run, epoch, train_loss, train_acc, valid_loss, valid_acc]
    writer.writerow(data)


# learning curve plotting
def plot_acc(trial_num, acc_file):
    acc_data = pd.read_csv(acc_file)
    epoch = acc_data["epoch"]
    train_loss = acc_data["train_loss"]
    train_acc = acc_data["train_acc"]
    valid_loss = acc_data["valid_loss"]
    valid_acc = acc_data["valid_acc"]

    plt.figure()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)  # create a 2 * 2 plot
    ax1.plot(epoch, train_loss, label="Train Loss")
    ax2.plot(epoch, train_acc, label="Train Acc")
    ax3.plot(epoch, valid_loss, label="Valid Loss")
    ax4.plot(epoch, valid_acc, label="Valid Acc")

    plot_file = "Results/" + trial_num + "/accuracy_plot.png"
    plt.savefig(plot_file)
    plt.show()


# attention plotting
def plot_attention():
    pass


# a function that manages training at one epoch
def train_one_epoch(model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device):
    model.train()  # enable dropout in training
    epoch_loss = 0
    epoch_acc = 0

    # training in one batch
    for i, (src, trg) in enumerate(data_loader):
        src = src.to(device)
        trg = trg.to(device)
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]

        optimizer.zero_grad()  # reset gradient at each iteration to 0
        output, pred = model(src, trg, teacher_forcing_ratio)
        # output = [trg_length, batch_size, output_dim]
        # pred = [trg_len, batch_size]

        batch_acc = (pred == trg).sum()  # calculate batch accuracy
        epoch_acc += batch_acc.item()  # add to epoch accuracy

        # remove the <SOS> token from output and target and reshape for loss calculation
        output_dim = output.shape[2]
        output = output[1:].view(-1, output_dim)
        # output = [(trg_length - 1) * batch_size, output_dim]
        trg = trg[1:].view(-1)
        # trg = [(trg_length - 1) * batch_size]

        batch_loss = criterion(output, trg)  # calculate batch loss
        epoch_loss += batch_loss.item()  # add to epoch loss
        batch_loss.backward()  # backpropagate loss
        clip_grad_norm_(model.parameters(), clip) # clip the gradients to prevent exploding
        optimizer.step()  # update the weights

    # average loss and accuracy over all batches
    average_loss = epoch_loss / len(data_loader)
    average_acc = epoch_acc / len(data_loader)

    return average_loss, average_acc


# a function that manages evaluation at one epoch
def evaluate_one_epoch(model, data_loader, criterion, device):
    model.eval()  # disable dropout in evaluation
    epoch_loss = 0
    epoch_acc = 0

    # evaluation in one batch
    with torch.no_grad():  # disable gradient tracking
        for i, (src, trg) in enumerate(data_loader):
            src = src.to(device)
            trg = trg.to(device)
            # src = [src_len, batch_size]
            # trg = [trg_len, batch_size]

            output, pred = model(src, trg, 0)  # turn off teacher forcing
            # output = [trg_length, batch_size, output_dim]
            # pred = [trg_len, batch_size]

            batch_acc = (pred == trg).sum()  # calculate batch accuracy
            epoch_acc += batch_acc.item()  # add to epoch accuracy

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


# a function that completes one run of training and evaluation of one model
def run_once(trial_num, run, datatype, condition):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    print(" - Loading dataset and building vocabulary:")
    annotations_file = "Dataset/English_" + datatype + "_" + condition + ".csv"

    if datatype == "txt":
        dataset = TextDataset(annotations_file, hp.special_tokens)
        print(f"The dataset contains {len(dataset)} UR-SR pairs")

        ur_vocab_size = len(dataset.ur_alphabet)
        sr_vocab_size = len(dataset.sr_alphabet)
        print(f"The UR vocabulary size is {ur_vocab_size}")
        print(f"The SR vocabulary size is {sr_vocab_size}")
    elif datatype == "aud":
        audio_dir = "Dataset/audio/English"
        dataset = AudioDataset(annotations_file, audio_dir, hp.sample_rate, hp.n_samples, device=device,
                               wav2mel=True, power2db=True, normalize=True)
        print(f"The dataset contains {len(dataset)} UR-SR pairs")
    else:
        raise ValueError("Invalid datatype input")

    print(" - Splitting dataset:")
    train_data, valid_data, test_data = random_split(dataset, [0.8, 0.1, 0.1])

    print(" - Creating dataloader:")
    train_dataloader = get_dataloader(train_data)
    valid_dataloader = get_dataloader(valid_data)
    test_dataloader = get_dataloader(test_data)

    print(" - Initializing model:")
    # model hyperparameters
    if datatype == "txt":
        encoder_input_dim = ur_vocab_size
        decoder_input_dim = sr_vocab_size
        output_dim = sr_vocab_size
    elif datatype == "aud":
        pass
    else:
        raise ValueError("Invalid datatype input")

    # model initialization
    attention = BahdanauAttention(hp.hidden_dim)
    encoder_net = Encoder(encoder_input_dim, hp.encoder_embedding_dim, hp.hidden_dim,
                          hp.n_layers, hp.encoder_dropout).to(device)
    decoder_net = Decoder(decoder_input_dim, hp.decoder_embedding_dim, hp.hidden_dim, output_dim,
                          hp.n_layers, hp.decoder_dropout, attention).to(device)
    seq2seq = Seq2Seq(encoder_net, decoder_net).to(device)

    # weight initialization
    seq2seq.apply(init_weights)
    model_parameters = sum(p.numel() for p in seq2seq.parameters() if p.requires_grad)
    print(f"The model has {model_parameters} trainable parameters")

    # optimizer and loss function
    optimizer = torch.optim.Adam(seq2seq.parameters(), lr=hp.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=hp.special_tokens.index(hp.pad_token))

    print(" - Training model:")
    # at each epoch, display progress bar
    for epoch in tqdm.tqdm(range(hp.n_epochs)):
        # update loss for each batch
        train_loss, train_acc = train_one_epoch(seq2seq, train_dataloader, optimizer, criterion, hp.clip,
                                                hp.teacher_forcing_ratio, device)
        valid_loss, valid_acc = evaluate_one_epoch(seq2seq, valid_dataloader, criterion, device)
        print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f} | Train Acc: {train_acc:7.3f}")
        print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f} | Train Acc: {valid_acc:7.3f}")

        # save and the accuracy value
        acc_file = "Results/" + trial_num + "/acc.csv"
        record_acc(acc_file, datatype, condition, run, epoch, train_acc, valid_acc)
        print(f"Accuracy data saved at {acc_file}")

        # save the model
        model_file = "Results/" + trial_num + "/English_" + datatype + "_" + condition + "_run" + run + "_seq2seq.pth"
        torch.save(seq2seq.state_dict(), model_file)
        print(f"Model trained and stored at {model_file}")

    # plot the accuracy value
    plot_acc(acc_file)

    print(" - Evaluating model:")
    # load the model
    seq2seq.load_state_dict(torch.load(model_file))

    # check loss for the test dataset
    test_loss, test_acc = evaluate_one_epoch(seq2seq, test_dataloader, criterion, device)
