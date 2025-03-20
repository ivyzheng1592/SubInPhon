# created 2025/01/07
# updated 2025/03/03
# An LSTM encoder-decoder with global attention based on Ben Trevett tutorial and Aladdin Persson tutorial

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
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=True)  # input embedding space, output hidden space
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)  # select the better hidden from forward and backward
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)  # select the better cell from forward and backward
        self.dropout = nn.Dropout(dropout)  # dropout probability, see https://arxiv.org/abs/1207.0580

    def forward(self, input: torch.Tensor):
        # input = [input_len, batch_size]

        embedding = self.dropout(self.embedding(input))
        # embedding = [input_len, batch_size, embedding_dim]

        encoder_states, (hidden, cell) = self.rnn(embedding)
        # encoder_states come from the top hidden layer
        # hidden and cell are stacked from all forward and backward hidden layers
        # encoder_states = [input_len, batch_size, hidden_dim * n_direction=2]
        # hidden = [n_layers=1 * n_direction=2, batch_size, hidden_dim]
        # cell = [n_layers=1 * n_direction=2, batch_size, hidden_dim]

        # hidden[-2, :, : ] is the last of the forwards RNN
        # hidden[-1, :, : ] is the last of the backwards RNN
        hidden = self.fc_hidden(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        cell = self.fc_cell(torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1))
        # hidden = [batch_size, hidden_dim]
        # cell = [batch_size, hidden_dim]
        # ignoring hidden = torch.tanh(hidden) in Ben Trevett tutorial because we are using LSTM

        return encoder_states, hidden, cell


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.fc_align = nn.Linear(hidden_dim * 3, hidden_dim)  # update weight of forward and backward encoder states and decoder hidden
        self.fc_score = nn.Linear(hidden_dim, 1)  # output a score for each alignment
        # ignoring bias=False in Ben Trevett tutorial

    def forward(self, hidden, encoder_states):
        # hidden = [batch_size, hidden_dim]
        # cell = [batch_size, hidden_dim]
        # encoder_states = [input_len, batch_size, hidden_dim * 2]

        input_len = encoder_states.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, input_len, 1)
        # hidden = [batch_size, input_len, hidden_dim]
        encoder_states = encoder_states.permute(1, 0, 2)
        # encoder_states = [batch_size, input_len, hidden_dim * 2]

        # construct a soft alignment between target decoder hidden (query) and encoder hidden of each input token (key)
        # calculate a score for each input token
        # calculate the weight of each input token by running the score through softmax
        align = torch.tanh(self.fc_align(torch.cat((hidden, encoder_states), dim=2)))
        # align = [batch_size, input_len, hidden_dim]
        score = self.fc_score(align).squeeze(2)
        # score = [batch_size, input_len]
        weight = torch.softmax(score, dim=1)
        # weight = [batch_size, input_len]

        return weight


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
        self.rnn = nn.LSTM(hidden_dim * 2 + embedding_dim, hidden_dim, n_layers)  # input forward and backward encoder hidden states and embedding
        self.fc_out = nn.Linear(hidden_dim * 3 + embedding_dim, output_dim)  # take into account context vector, decoder hidden, and embedding for the prediction
        self.dropout = nn.Dropout(dropout)
        self.attention = attention

    def forward(self, input, encoder_states, hidden, cell):
        # input = [batch_size]
        # encoder_states = [input_len, batch_size, hidden_dim * 2]
        # hidden = [batch_size, hidden_dim]
        # cell = [batch_size, hidden_dim]

        input = input.unsqueeze(0)
        # input = [input_len=1, batch_size]

        embedding = self.dropout(self.embedding(input))
        # embedding = [1, batch_size, embedding_dim]

        # calculate the attention weight with target decoder hidden (query) and all encoder hidden (key)
        # compute the context vector by multiplying the attention weight to each target embedding (value)
        # concatenate the target embedding and the context vector as the rnn input
        weight = self.attention(hidden, encoder_states)
        # weight = [batch_size, input_len]
        context_vector = torch.bmm(weight.unsqueeze(1), encoder_states.permute(1, 0, 2)).permute(1, 0, 2)
        # weight = [batch_size, 1, input_len]
        # encoder_states = [batch_size, input_len, hidden_dim * 2]
        # context_vector = [batch_size, 1, hidden_dim *2] -> context_vector = [1, batch_size, hidden_dim * 2]
        rnn_input = torch.cat((embedding, context_vector), dim=2)
        # rnn_input = [1, batch_size, hidden_dim * 2 + embedding_dim]

        decoder_state, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        # decoder_state = [1, batch_size, hidden_dim]
        # hidden = [1, batch_size, hidden_dim]
        # cell = [1, batch_size, hidden_dim]
        assert (decoder_state == hidden).all()

        embedding = embedding.squeeze(0)
        decoder_state = decoder_state.squeeze(0)
        hidden = hidden.squeeze(0)
        cell = cell.squeeze(0)
        context_vector = context_vector.squeeze(0)

        # the original manuscript uses all of embedding, decoder state, and context vector for the prediction
        output = self.fc_out(torch.cat((embedding, decoder_state, context_vector), dim=1))
        # output = [batch_size, output_dim]

        return output, hidden, cell, weight


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

        encoder_states, hidden, cell = self.encoder(src)
        # encoder_states are all hidden states of the input sequence
        # hidden and cell are the final forward and backward hidden and cell concatenated
        # encoder_states = [src_len, batch_size, encoder_hidden_dim * 2]
        # hidden = [batch_size, encoder_hidden_dim]
        # cell = [batch_size, encoder_hidden_dim]

        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        output_dim = self.decoder.output_dim
        decoder_outputs = torch.zeros(trg_len, batch_size, output_dim).to(self.device)
        predictions = torch.zeros(trg_len, batch_size).to(self.device)
        # decoder_outputs store the probability of all output vocabulary for each input token
        # predictions store the predicted output token for each input token
        # decoder_outputs = [trg_len, batch_size, output_dim]
        # predictions = [trg_len, batch_size]

        input = trg[0]  # first input to the decoder is the <SOS> token
        for t in range(1, trg_len):
            # at every time step, insert input token, encoder_states, and previous hidden and cell
            # receive output and new hidden and cell
            # and get the best word predicted by the decoder
            output, hidden, cell, _ = self.decoder(input, encoder_states, hidden, cell)
            # output = [batch_size, output_dim]
            # hidden = [batch_size, decoder_hidden_dim]
            # cell = [batch_size, decoder_hidden_dim]
            best_guess = output.argmax(1)
            # best_guess = [batch_size]

            # store output and best guess for current time step
            decoder_outputs[t] = output
            predictions[t] = best_guess

            # with probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the decoder predicted it to be
            # Teacher Forcing is used so that the model gets used to seeing similar inputs at training and testing time
            input = trg[t] if random.random() < teacher_forcing_ratio else best_guess
            # input = [batch_size]

        return decoder_outputs, predictions