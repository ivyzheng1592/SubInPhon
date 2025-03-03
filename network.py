# created 2025/01/07
# updated 2025/03/03
# An LSTM encoder-decoder with global attention based on Ben Trevett tutorial

import random
import torch
import torch.nn as nn
import hyper_params as hp
from torchsummary import summary


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, n_layers, p):
        super(Encoder, self).__init__()
        #self.hidden_dim = hidden_dim
        #self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, embedding_dim)  # map the input vocabulary to a d-dimensional space
        self.rnn = nn.GRU(embedding_dim, encoder_hidden_dim, bidirectional=True)  # input embedding space and output hidden space
        self.fc = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)  # select the better context vector from forward and backward
        self.dropout = nn.Dropout(p)  # dropout with probability p, see https://arxiv.org/abs/1207.0580

    def forward(self, x: torch.Tensor):
        # x = [x_length, batch_size]

        embedding = self.dropout(self.embedding(x))
        # embedding = [x_length, batch_size, embedding_dim]

        encoder_states, (hidden, cell) = self.rnn(embedding)
        # encoder_states = [x_length, batch_size, hidden_dim * n_directions]
        # hidden = [n_layers * n_directions, batch_size, hidden_dim]
        # cell = [n_layers * n_directions, batch_size, hidden_dim]
        # encoder_states come from the top hidden layer
        # hidden and cell are the context vectors

        # Use forward, backward cells and hidden through a linear layer
        # so that it can be input to the decoder which is not bidirectional
        # Also using index slicing ([idx:idx+1]) to keep the dimension
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        # hidden shape: (2, N, hidden_size)
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

        return encoder_states, hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, p):
        # input_size and output_size should be the same
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.dropout = nn.Dropout(p)
        #self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size, num_layers)  # input forward and backward encoder hidden states and context vector

        self.energy = nn.Linear(hidden_size * 3, 1)  # calculate energy state
        self.softmax = nn.Softmax(dim=0)  # calculate attention
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, encoder_states, hidden, cell):
        x = x.unsqueeze(0)
        # x = [batch_size] -> x = [1, batch_size]

        # Not applying Dropout on the embeddings!
        # embedding = self.dropout(self.embedding(x))
        embedding = self.embedding(x)
        # embedding: (1, N, embedding_size)

        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        # hidden_reshaped: (seq_length, N, hidden_size*2)

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        # energy: (seq_length, N, 1)

        attention = self.softmax(energy)
        attention = attention.permute(1, 2, 0)
        # attention: (seq_length, N, 1) -> (N, 1, seq_length)
        encoder_states = encoder_states.permute(1, 0, 2)
        # encoder_states: (N, seq_length, hidden_size*2)

        context_vector = torch.bmm(attention, encoder_states).permute(1, 0, 2)
        # context_vector: (N, 1, hidden_size*2) -> (1, N, hidden_size*2)

        rnn_input = torch.cat((context_vector, embedding), dim=2)
        # rnn_input: (1, N, hidden_size*2 + embedding_size)

        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # outputs shape: (1, N, hidden_size)

        predictions = self.fc(outputs).squeeze(0)
        # predictions: (N, output_size)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        assert (
            encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        #batch_size = src.shape[1]
        batch_size = trg.shape[1]
        # source shape: (trg_len, N)
        trg_len = trg.shape[0]
        # target shape:
        trg_vocab_size = len(trg.sr_alphabet)

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)
        encoder_states, hidden, cell = self.encoder(src)

        # First input will be <SOS> token
        x = trg[0]

        for t in range(1, trg_len):
            # At every time step use encoder_states and update hidden, cell
            output, hidden, cell, _ = self.decoder(x, encoder_states, hidden, cell)

            # Store prediction for current time step
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1
            # then inputs at test time might be completely different than what the
            # network is used to. This was a long comment.
            x = trg[t] if random.random() < teacher_forcing_ratio else best_guess

        return outputs