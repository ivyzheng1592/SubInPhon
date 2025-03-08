# created 2025/03/03 w/ Ben Trevett tutorial

import torch
import torch.nn as nn
import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from text_dataset_loader import TextDataset, get_dataloader
from run_setup import train_fn, evaluate_fn, init_weights
from network import Encoder, Decoder, BahdanauAttention, Seq2Seq
import hyper_params as hp


if __name__ == "__main__":
    print(" - Loading dataset and building vocabulary:")
    annotations_file = "Dataset/English_txt_harmony.csv"
    text_dataset = TextDataset(annotations_file, hp.special_tokens)
    print(f"The dataset contains {len(text_dataset)} UR-SR pairs")

    ur_vocab_size = len(text_dataset.ur_alphabet)
    sr_vocab_size = len(text_dataset.sr_alphabet)
    print(f"The UR vocabulary size is {ur_vocab_size}")
    print(f"The SR vocabulary size is {sr_vocab_size}")

    print(" - Splitting dataset:")
    train_data, valid_data, test_data = random_split(text_dataset, [0.8, 0.1, 0.1])

    print(" - Creating dataloader:")
    train_dataloader = get_dataloader(train_data)
    valid_dataloader = get_dataloader(valid_data)
    test_dataloader = get_dataloader(test_data)

    print(" - Initializing model:")
    # model hyperparameters
    encoder_input_dim = ur_vocab_size
    decoder_input_dim = sr_vocab_size
    output_dim = sr_vocab_size

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

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

    #writer = SummaryWriter(f'runs/Loss_plot')
    #step = 0

    print(" - Training model:")
    # placeholder for the best validation loss
    best_valid_loss = float("inf")

    # at each epoch, display progress bar
    for epoch in tqdm.tqdm(range(hp.n_epochs)):

        # update loss for each batch
        train_loss = train_fn(seq2seq, train_dataloader, optimizer, criterion, hp.clip, hp.teacher_forcing_ratio, device)
        valid_loss = evaluate_fn(seq2seq, valid_dataloader, criterion, device)
        print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
        print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")

        # if the validation loss in this epoch is the best so far
        # update the best validation loss and save the model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(seq2seq.state_dict(), "seq2seq.pth")
            print("Model trained and stored at seq2seq.pth.")

    print(" - Evaluating model:")
    # load the model
    seq2seq.load_state_dict(torch.load("seq2seq.pth"))

    # check loss for the test dataset
    test_loss = evaluate_fn(seq2seq, test_dataloader, criterion, device)