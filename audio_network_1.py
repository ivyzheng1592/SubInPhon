# created 2025/03/31
# An LSTM encoder-decoder with global attention that deals with audio input

import torch
import torch.nn as nn
import random
import hyper_params as hp
from text_network import BahdanauAttention


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Linear(input_dim, embedding_dim)  # reduce the dimensionality of the input image on the frequency domain
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=True)  # input embedding space, output hidden space
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)  # select the better hidden from forward and backward
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)  # select the better cell from forward and backward
        self.dropout = nn.Dropout(dropout)  # dropout probability, see https://arxiv.org/abs/1207.0580
        self.relu = nn.ReLU()

    def forward(self, input):
        # input = [batch_size, n_channels=1, n_freq, input_len]

        input = input.squeeze(1).permute(2, 0, 1)
        # input = [batch_size, n_freq, input_len] -> input = [input_len, batch_size, n_freq]
        embedding = self.dropout(self.relu(self.embedding(input)))
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

        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.rnn = nn.LSTM(hidden_dim * 2 + embedding_dim, hidden_dim, n_layers)  # input forward and backward encoder hidden states and embedding
        self.fc_out = nn.Linear(hidden_dim * 3 + embedding_dim, output_dim)  # take into account context vector, decoder hidden, and embedding for the prediction
        self.dropout = nn.Dropout(dropout)
        self.attention = attention

    def forward(self, input, encoder_states, hidden, cell):
        # input = [batch_size, n_channels]
        # input = [batch_size, n_channels=1, n_freq, input_len]
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
        weight = self.attention(encoder_states, hidden)
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


class AudioSeq2Seq(nn.Module):
    def __init__(self, encoder_input_dim, decoder_input_dim, encoder_embedding_dim, decoder_embedding_dim,
                 n_layers, hidden_dim, output_dim, encoder_dropout, decoder_dropout):
        super(AudioSeq2Seq, self).__init__()

        self.device = device
        self.encoder_input_dim = encoder_input_dim
        self.decoder_input_dim = decoder_input_dim
        self.encoder_embedding_dim = encoder_embedding_dim
        self.decoder_embedding_dim = decoder_embedding_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout

        # model components
        self.attention = BahdanauAttention(hidden_dim)
        self.encoder = Encoder(encoder_input_dim, encoder_embedding_dim, hidden_dim,
                               n_layers, encoder_dropout).to(self.device)
        self.decoder = Decoder(decoder_input_dim, decoder_embedding_dim, hidden_dim, output_dim,
                               n_layers, decoder_dropout, self.attention).to(self.device)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [batch_size, n_channels, freq, src_len]
        # trg = [batch_size, n_channels, freq, trg_len]

        encoder_states, hidden, cell = self.encoder(src)
        # encoder_states are all hidden states of the input sequence
        # hidden and cell are the final forward and backward hidden and cell concatenated
        # encoder_states = [src_len, batch_size, encoder_hidden_dim * 2]
        # hidden = [batch_size, encoder_hidden_dim]
        # cell = [batch_size, encoder_hidden_dim]

        batch_size = trg.shape[0]
        n_channels = trg.shape[1]
        trg_len = trg.shape[3]
        decoder_outputs = torch.zeros(trg_len, batch_size, n_channels, self.output_dim).to(self.device)
        # decoder_outputs store the output for each frame
        # decoder_outputs = [trg_len, batch_size, n_channels, output_dim]

        input = torch.zeros(batch_size, n_channels, self.output_dim).to(device)  # first input to the decoder is 0 tensor
        for t in range(1, trg_len):
            # at every time step, insert input frame, encoder_states, and previous hidden and cell
            # receive output and new hidden and cell
            # and get decoder output
            output, hidden, cell, _ = self.decoder(input, encoder_states, hidden, cell)
            # output = [batch_size, n_channels, output_dim]
            # hidden = [batch_size, decoder_hidden_dim]
            # cell = [batch_size, decoder_hidden_dim]

            # store output for current time step
            decoder_outputs[t] = output

            # with probability of teacher_force_ratio we take the actual next frame
            # otherwise we take the frame that the decoder predicted it to be
            # Teacher Forcing is used so that the model gets used to seeing similar inputs at training and testing time
            input = trg[t] if random.random() < teacher_forcing_ratio else output
            # input = [batch_size]

        return decoder_outputs


if __name__ == "__main__":
    import torchinfo

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    print(" - Initializing model:")
    encoder_input_dim = 94
    decoder_input_dim = 94
    output_dim = 94

    seq2seq = AudioSeq2Seq(encoder_input_dim, decoder_input_dim, hp.encoder_embedding_dim, hp.decoder_embedding_dim,
                           hp.n_layers, hp.hidden_dim, output_dim, hp.encoder_dropout, hp.decoder_dropout, device).to(device)

    torchinfo.summary(seq2seq, input_size = [(7, 32), (7, 32)], dtypes=[torch.long, torch.long], device=device)
