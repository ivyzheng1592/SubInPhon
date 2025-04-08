# created 2025/01/07
# updated 2025/03/03
# An LSTM encoder-decoder with global attention based on Ben Trevett tutorial and Aladdin Persson tutorial

import torch
import torch.nn as nn
import random
import hyper_params as hp


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

    def forward(self, input):
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
        self.fc_W1 = nn.Linear(hidden_dim * 2, hidden_dim)  # update weight of forward and backward encoder states
        self.fc_W2 = nn.Linear(hidden_dim, hidden_dim)  # update weight of decoder hidden
        self.fc_V = nn.Linear(hidden_dim, 1)  # output a score for each alignment
        # ignoring bias=False in Ben Trevett tutorial

    def forward(self, encoder_states, hidden):
        # encoder_states = [src_len, batch_size, hidden_dim * 2]
        # hidden = [batch_size, hidden_dim]
        # cell = [batch_size, hidden_dim]

        encoder_states = encoder_states.permute(1, 0, 2)
        # encoder_states = [batch_size, src_len, hidden_dim * 2]
        src_len = encoder_states.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # hidden = [batch_size, src_len, hidden_dim]

        # construct a soft alignment between target decoder hidden (query) and encoder hidden of each input token (key)
        # calculate the score of each src input token
        # run the score through softmax to get the weight of each src input token
        score = self.fc_V(torch.tanh(self.fc_W1(encoder_states) + self.fc_W2(hidden)))
        # score = [batch_size, src_len, hidden_dim] -> score = [batch_size, src_len, 1]
        weight = torch.softmax(score.squeeze(2), dim=1)
        # weight = [batch_size, src_len]

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
        # encoder_states = [src_len, batch_size, hidden_dim * 2]
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
        # weight = [batch_size, src_len]
        context_vector = torch.bmm(weight.unsqueeze(1), encoder_states.permute(1, 0, 2)).permute(1, 0, 2)
        # weight = [batch_size, 1, src_len]
        # encoder_states = [batch_size, src_len, hidden_dim * 2]
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

        return output, hidden, cell, embedding, weight


class TextSeq2Seq(nn.Module):
    def __init__(self, encoder_input_dim, decoder_input_dim, encoder_embedding_dim, decoder_embedding_dim,
                 n_layers, hidden_dim, output_dim, encoder_dropout, decoder_dropout, device='cuda'):
        super(TextSeq2Seq, self).__init__()

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
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]

        encoder_states, hidden, cell = self.encoder(src)
        # encoder_states are all hidden states of the src input sequence
        # hidden and cell are the final forward and backward hidden and cell concatenated
        # encoder_states = [src_len, batch_size, encoder_hidden_dim * 2]
        # hidden = [batch_size, encoder_hidden_dim]
        # cell = [batch_size, encoder_hidden_dim]

        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        src_len = src.shape[0]
        decoder_outputs = torch.zeros(trg_len, batch_size, self.output_dim).to(self.device)
        predictions = torch.zeros(trg_len, batch_size).to(self.device)
        embeddings = torch.zeros(trg_len, batch_size, self.decoder_embedding_dim).to(self.device)
        attentions = torch.zeros(trg_len, batch_size, src_len).to(self.device)
        # decoder_outputs store the probability of all output vocabulary for each trg input token
        # predictions store the predicted output token for each trg input token
        # embeddings store the embedding for each trg input token
        # attentions store the attention weights for each trg input token
        # decoder_outputs = [trg_len, batch_size, output_dim]
        # predictions = [trg_len, batch_size]
        # embeddings = [trg_len, batch_size, decoder_embedding_dim]
        # attentions = [trg_len, batch_size, src_len]

        input = trg[0]  # first input to the decoder is the <SOS> token
        for t in range(1, trg_len):
            # at every time step, insert trg input token, encoder_states, and previous hidden and cell
            # receive output and new hidden and cell
            # and get the best word predicted by the decoder
            output, hidden, cell, embed, weight = self.decoder(input, encoder_states, hidden, cell)
            # output = [batch_size, output_dim]
            # hidden = [batch_size, decoder_hidden_dim]
            # cell = [batch_size, decoder_hidden_dim]
            # embed = [batch_size, decoder_embedding_dim]
            # weight = [batch_size, src_len]
            best_guess = output.argmax(1)
            # best_guess = [batch_size]

            # store output, best guess, input embedding, and attention weight for current time step
            decoder_outputs[t] = output
            predictions[t] = best_guess
            embeddings[t] = embed
            attentions[t] = weight

            # with probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the decoder predicted it to be
            # Teacher Forcing is used so that the model gets used to seeing similar inputs at training and testing time
            input = trg[t] if random.random() < teacher_forcing_ratio else best_guess
            # input = [batch_size]

        return decoder_outputs, predictions, embeddings, attentions


if __name__ == "__main__":
    import torchinfo

    print(" - Initializing model:")
    encoder_input_dim = 30
    decoder_input_dim = 30
    output_dim = 30

    seq2seq = TextSeq2Seq(encoder_input_dim, decoder_input_dim, hp.encoder_embedding_dim, hp.decoder_embedding_dim,
                          hp.n_layers, hp.hidden_dim, output_dim, hp.encoder_dropout, hp.decoder_dropout, device='cpu').to('cpu')

    torchinfo.summary(seq2seq, input_size = [(8, 32), (8, 32)], dtypes=[torch.long, torch.long], device='cpu')
