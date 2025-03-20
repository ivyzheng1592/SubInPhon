# created 2025/03/20
# An LSTM encoder-decoder without global attention based on Ben Trevett tutorial and Aladdin Persson tutorial

import torch
import torch.nn as nn
import random


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, embedding_dim)  # map the input vocabulary to a d-dimensional space
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)  # input embedding space, output hidden space
        self.dropout = nn.Dropout(dropout)  # dropout probability, see https://arxiv.org/abs/1207.0580

    def forward(self, input: torch.Tensor):
        # input = [input_len, batch_size]

        embedding = self.dropout(self.embedding(input))
        # embedding = [input_len, batch_size, embedding_dim]

        output, (hidden, cell) = self.rnn(embedding)
        # output = [input_len, batch_size, hidden_dim * n_direction=1]
        # hidden = [n_layers=1 * n_direction=1, batch_size, hidden_dim]
        # cell = [n_layers=1 * n_direction=1, batch_size, hidden_dim]

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, dropout, attention):
        super(Decoder, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        assert (
            self.input_dim == self.output_dim
        ), "Decoder input dimension and output dimension must be the same!"

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)  # input embedding space, output hidden space
        self.fc_out = nn.Linear(hidden_dim, output_dim)  # input hidden space, output predictions
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch_size]
        # hidden = [1, batch_size, hidden_dim]
        # cell = [1, batch_size, hidden_dim]

        input = input.unsqueeze(0)
        # input = [input_len=1, batch_size]

        embedding = self.dropout(self.embedding(input))
        # embedding = [1, batch_size, embedding_dim]

        output, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # outputs = [1, batch_size, hidden_dim]
        # hidden = [1, batch_size, hidden_dim]
        # cell = [1, batch_size, hidden_dim]

        output = self.fc_out(output.squeeze(0))
        # output = [batch_size, output_dim]

        return output, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert (
            self.encoder.hidden_dim == self.decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            self.encoder.n_layers == self.decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio):
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]

        hidden, cell = self.encoder(src)
        # hidden = [1, batch_size, encoder_hidden_dim]
        # cell = [1, batch_size, encoder_hidden_dim]

        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        output_dim = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, output_dim).to(self.device)
        predictions = torch.zeros(trg_len, batch_size).to(self.device)
        # outputs store the probability of all output vocabulary for each input token
        # predictions store the predicted output token for each input token
        # outputs = [trg_len, batch_size, output_dim]
        # predictions = [trg_len, batch_size]

        input = trg[0]  # first input to the decoder is the <SOS> token
        for t in range(1, trg_len):
            # at every time step, insert input token, encoder_states, and previous hidden and cell
            # receive output and new hidden and cell
            # and get the best word predicted by the decoder
            output, hidden, cell, _ = self.decoder(input, hidden, cell)
            # output = [batch_size, output_dim]
            # hidden = [batch_size, hidden_dim]
            # cell = [batch_size, hidden_dim]
            best_guess = output.argmax(1)
            # best_guess = [batch_size]

            # store output and best guess for current time step
            outputs[t] = output
            predictions[t] = best_guess

            # with probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the decoder predicted it to be
            # Teacher Forcing is used so that the model gets used to seeing similar inputs at training and testing time
            input = trg[t] if random.random() < teacher_forcing_ratio else best_guess
            # input = [batch_size]

        return outputs, predictions