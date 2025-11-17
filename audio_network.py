# created 2025/03/31
# updated 2025/11/07
# An LSTM encoder-decoder with global attention that deals with audio input based on Translatotron 2

import torch
import torch.nn as nn
import random
import hyper_params as hp


class AudioEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AudioEncoder, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = hp.audio_embedding_dim
        self.hidden_dim = hp.audio_hidden_dim
        self.n_layers = hp.audio_n_layers
        self.dropout = hp.audio_dropout

        self.prenet = nn.Linear(input_dim, self.embedding_dim)
        # reduce the dimensionality of the input image on the frequency domain
        self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, self.n_layers, bidirectional=True)
        # input embedding space, output hidden space
        self.fc_hidden = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        # select the better hidden from forward and backward
        self.fc_cell = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        # select the better cell from forward and backward
        self.dropout = nn.Dropout(self.dropout)
        # dropout probability, see https://arxiv.org/abs/1207.0580

    def forward(self, input):
        # input = [batch_size, n_channels=1, n_freq, input_len]

        input = input.squeeze(1).permute(2, 0, 1)
        # input = [batch_size, n_freq, input_len] -> input = [input_len, batch_size, n_freq]
        embedding = self.dropout(torch.relu(self.prenet(input)))
        # embedding = [input_len, batch_size, embedding_dim]

        encoder_states, (hidden, cell) = self.rnn(embedding)
        # encoder_states = [input_len, batch_size, hidden_dim * n_direction=2]
        # hidden = [n_layers=1 * n_direction=2, batch_size, hidden_dim]
        # cell = [n_layers=1 * n_direction=2, batch_size, hidden_dim]

        # hidden[-2, :, : ] is the last of the forwards RNN
        # hidden[-1, :, : ] is the last of the backwards RNN
        hidden = self.fc_hidden(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        cell = self.fc_cell(torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1))
        # hidden = [batch_size, hidden_dim]
        # cell = [batch_size, hidden_dim]

        hidden = hidden.unsqueeze(0)
        cell = cell.unsqueeze(0)
        # hidden = [1, batch_size, hidden_dim]
        # cell = [1, batch_size, hidden_dim]

        return encoder_states, hidden, cell


class TextDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TextDecoder, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = hp.text_embedding_dim
        self.hidden_dim = hp.text_hidden_dim
        self.context_hidden_dim = hp.audio_hidden_dim * 2
        self.output_dim = output_dim
        self.n_layers = hp.text_n_layers
        self.dropout = hp.text_dropout
        assert (
            self.input_dim == self.output_dim
        ), "Decoder input dimension and output dimension must be the same!"

        self.embedding = nn.Embedding(input_dim, self.embedding_dim)
        # map the input vocabulary to a d-dimensional space
        self.rnn = nn.LSTM(self.context_hidden_dim + self.embedding_dim, self.hidden_dim, self.n_layers)
        # input context vector and embedding, output hidden space
        self.fc_out = nn.Linear(self.context_hidden_dim + self.hidden_dim + self.embedding_dim, output_dim)
        # take into account context vector, decoder hidden, and embedding for the prediction
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, input, context_vector, hidden, cell):
        # input = [batch_size]
        # context_vector = [1, batch_size, audio_hidden_dim * 2]
        # hidden = [1, batch_size, text_hidden_dim]
        # cell = [1, batch_size, text_hidden_dim]

        input = input.unsqueeze(0)
        # input = [input_len=1, batch_size]
        embedding = self.dropout(self.embedding(input))
        # embedding = [1, batch_size, embedding_dim]

        # concatenate the target embedding and the context vector as the rnn input
        rnn_input = torch.cat((embedding, context_vector), dim=2)
        # rnn_input = [1, batch_size, audio_hidden_dim * 2 + embedding_dim]
        decoder_state, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        # decoder_state = [1, batch_size, text_hidden_dim]
        # hidden = [1, batch_size, text_hidden_dim]
        # cell = [1, batch_size, text_hidden_dim]
        assert (decoder_state == hidden).all()

        embedding = embedding.squeeze(0)
        context_vector = context_vector.squeeze(0)
        decoder_state = decoder_state.squeeze(0)

        # the original manuscript uses all of embedding, decoder state, and context vector for the prediction
        output = self.fc_out(torch.cat((embedding, decoder_state, context_vector), dim=1))
        # output = [batch_size, output_dim]

        return output, decoder_state, hidden, cell


class DurationPredictor(nn.Module):
    def __init__(self):
        super(DurationPredictor, self).__init__()

        self.input_dim = hp.audio_hidden_dim * 2 + hp.text_hidden_dim
        self.hidden_dim = hp.audio_hidden_dim
        self.n_layers = hp.aux_n_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, bidirectional=True)
        self.proj = nn.Linear(self.hidden_dim * 2, 1)

    def forward(self, input):
        # input = [txt_input_len, batch_size, aud_hidden_dim * 2 + txt_hidden_dim]

        durations, _ = self.lstm(input)
        # durations = [txt_input_len, batch_size, aud_hidden_dim * 2]
        durations = self.proj(durations)
        # durations = [txt_input_len, batch_size, 1]

        return durations


class RangePredictor(nn.Module):
    def __init__(self):
        super(RangePredictor, self).__init__()

        self.input_dim = hp.audio_hidden_dim * 2 + 1
        self.hidden_dim = hp.audio_hidden_dim
        self.n_layers = hp.aux_n_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, bidirectional=True)
        self.proj = nn.Linear(self.hidden_dim * 2, 1)
        self.softplus = nn.Softplus()

    def forward(self, input, durations):
        # input = [txt_input_len, batch_size, aud_hidden_dim * 2]
        # durations = [txt_input_len, batch_size, 1]

        lstm_input = torch.cat((input, durations), dim=2)
        # lstm_input = [txt_input_len, batch_size, aud_hidden_dim * 2 + 1]
        ranges, _ = self.lstm(lstm_input)
        # ranges = [txt_input_len, batch_size, aud_hidden_dim * 2]
        ranges = self.softplus(self.proj(ranges))
        # ranges = [txt_input_len, batch_size, 1]

        return ranges


class GaussianUpsampling(nn.Module):
    def __init__(self, device):
        super(GaussianUpsampling, self).__init__()

        self.device = device

    def forward(self, input, durations, ranges):
        # input = [txt_input_len, batch_size, aud_hidden_dim * 2]
        # durations = [txt_input_len, batch_size, 1]
        # ranges = [txt_input_len, batch_size, 1]

        input_len = input.size(0)
        batch_size = input.size(1)
        output_len = int(torch.sum(durations, dim=0).max().item())
        # the maximum total duration of tokens

        # at each token, summarize duration until that token
        c = torch.cumsum(durations, dim=0).float() - 0.5 * durations
        # c = [txt_input_len, batch_size, 1]

        t = torch.arange(output_len).expand(input_len, batch_size, output_len).float().to(self.device)
        # t = [txt_input_len, batch_size, aud_output_len]

        w_1 = torch.exp(-(ranges ** -2) * ((t - c) ** 2))
        w_2 = torch.sum(torch.exp(-(ranges ** -2) * ((t - c) ** 2)), dim=0, keepdim=True) + 1e-20
        w = w_1 / w_2
        # w = [txt_input_len, batch_size, aud_output_len]

        upsamples = torch.bmm(w.transpose(1, 2), input)
        # upsamples = [batch_size, output_len, aud_hidden_dim * 2]

        return upsamples


class AudioSynthesizer(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, dropout,
                 gaussian_upsample, attention):
        super(AudioSynthesizer, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = 1  # n_channels=1 * n_frames_per_step=1
        self.n_layers = n_layers

        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.rnn_1 = nn.LSTM(hidden_dim * 2 + embedding_dim, hidden_dim)  # input embedding and encoder states, output attention input
        self.rnn_2 = nn.LSTM(hidden_dim * 3, hidden_dim)  # input encoder states and attention output
        self.linear_projection = nn.Linear(hidden_dim * 3, 1)
        self.gate = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.attention = attention

    def forward(self, input, encoder_states, decoder_states, hidden, cell):
        # input = [batch_size]
        # encoder_states = [input_len, batch_size, hidden_dim * 2]
        # hidden = [batch_size, hidden_dim]
        # cell = [batch_size, hidden_dim]

        input = input.squeeze(1).permute(2, 0, 1)
        # input = [batch_size, n_freq, input_len] -> input = [input_len, batch_size, n_freq]
        embedding = self.dropout(torch.relu(self.embedding(input)))
        # embedding = [input_len, batch_size, embedding_dim]


class AudioSeq2Seq(nn.Module):
    def __init__(self, encoder_input_dim, decoder_input_dim, synthesizer_input_dim,
                 text_output_dim, audio_output_dim, device):
        super(AudioSeq2Seq, self).__init__()

        self.device = device
        self.encoder_input_dim = encoder_input_dim
        self.decoder_input_dim = decoder_input_dim
        self.synthesizer_input_dim = synthesizer_input_dim
        self.text_output_dim = text_output_dim
        self.audio_output_dim = audio_output_dim

        self.attention = nn.MultiheadAttention(hp.audio_hidden_dim, hp.num_heads)
        self.duration_predictor = DurationPredictor()
        self.range_predictor = RangePredictor()
        self.gaussian_upsample = GaussianUpsampling()
        self.encoder = AudioEncoder(encoder_input_dim).to(self.device)
        self.decoder = TextDecoder(decoder_input_dim, text_output_dim).to(self.device)
        self.synthesizer = AudioSynthesizer(synthesizer_input_dim, audio_output_dim).to(self.device)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = ([txt_src_len, batch_size], [batch_size, n_channels, freq, aud_src_len])
        # trg = ([txt_trg_len, batch_size], [batch_size, n_channels, freq, aud_trg_len])

        encoder_states, hidden, cell = self.encoder(src[1])
        # encoder_states = [src_len, batch_size, encoder_hidden_dim * 2]
        # hidden = [1, batch_size, encoder_hidden_dim]
        # cell = [1, batch_size, encoder_hidden_dim]

        # text decoder
        txt_trg_len = trg[0].shape[0]
        batch_size = trg[0].shape[1]
        context_vectors = torch.zeros(txt_trg_len, batch_size, self.audio_hidden_dim * 2).to(self.device)
        decoder_outputs = torch.zeros(txt_trg_len, batch_size, self.text_output_dim).to(self.device)
        decoder_states = torch.zeros(txt_trg_len, batch_size, self.text_hidden_dim).to(self.device)
        predictions = torch.zeros(txt_trg_len, batch_size).to(self.device)
        # context_vectors = [txt_trg_len, batch_size, encoder_hidden_dim * 2]
        # decoder_outputs = [txt_trg_len, batch_size, output_dim]
        # decoder_states = [txt_trg_len, batch_size, decoder_hidden_dim]
        # predictions = [txt_trg_len, batch_size]

        txt_input = trg[0][0]  # first input text to the decoder is the <SOS> token
        for t in range(1, txt_trg_len):

            # calculate the attention weight with target decoder hidden (query) and all encoder hidden (key, value)
            context_vector, weight = self.attention(hidden, encoder_states, encoder_states)
            # context_vector = [1, batch_size, encoder_hidden_dim * 2]
            # weight = [batch_size, 1, src_len]

            # at every time step, insert input token, context vector, and previous hidden and cell
            # receive output, decoder state and new hidden and cell
            # and get the best word predicted by the decoder
            txt_output, decoder_state, hidden, cell = self.decoder(txt_input, context_vector, hidden, cell)
            # output = [batch_size, output_dim]
            # decoder_state = [batch_size, hidden_dim]
            # hidden = [1, batch_size, decoder_hidden_dim]
            # cell = [1, batch_size, decoder_hidden_dim]
            best_guess = txt_output.argmax(1)
            # best_guess = [batch_size]

            # store context vector, decoder output, decoder states and best guess for current time step
            context_vectors[t] = context_vector.squeeze(0)
            decoder_outputs[t] = txt_output
            decoder_states[t] = decoder_state
            predictions[t] = best_guess

            # with probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the decoder predicted it to be
            # Teacher Forcing is used so that the model gets used to seeing similar inputs at training and testing time
            txt_input = trg[t] if random.random() < teacher_forcing_ratio else best_guess
            # input = [batch_size]

        # audio synthesizer
        upsample_input = torch.cat((decoder_states, context_vectors), dim=2)
        # syn_input = [txt_trg_len, batch_size, encoder_hidden_dim * 2 + decoder_hidden_dim]
        durations = self.duration_predictor(upsample_input)
        # durations = [txt_input_len, batch_size, 1]
        ranges = self.range_predictor(upsample_input, durations)
        # ranges = [txt_input_len, batch_size, 1]
        upsamples = self.gaussian_upsample(upsample_input, durations, ranges)
        # upsamples = [batch_size, aud_output_len, aud_hidden_dim * 2]

        txt_input = trg[0][0]  # first input text to the decoder is the <SOS> token
        for t in range(1, txt_trg_len):
            # calculate the attention weight with target decoder hidden (query) and all encoder hidden (key, value)
            context_vector, weight = self.attention(hidden, encoder_states, encoder_states)
            # context_vector = [1, batch_size, encoder_hidden_dim * 2]
            # weight = [batch_size, 1, src_len]

            # at every time step, insert input token, context vector, and previous hidden and cell
            # receive output, decoder state and new hidden and cell
            # and get the best word predicted by the decoder
            txt_output, decoder_state, hidden, cell = self.decoder(txt_input, context_vector, hidden, cell)
            # output = [batch_size, output_dim]
            # decoder_state = [batch_size, hidden_dim]
            # hidden = [1, batch_size, decoder_hidden_dim]
            # cell = [1, batch_size, decoder_hidden_dim]
            best_guess = txt_output.argmax(1)
            # best_guess = [batch_size]

            # store context vector, decoder output, decoder states and best guess for current time step
            context_vectors[t] = context_vector.squeeze(0)
            decoder_outputs[t] = txt_output
            decoder_states[t] = decoder_state
            predictions[t] = best_guess

            # with probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the decoder predicted it to be
            # Teacher Forcing is used so that the model gets used to seeing similar inputs at training and testing time
            txt_input = trg[t] if random.random() < teacher_forcing_ratio else best_guess
            # input = [batch_size]

        return decoder_outputs, predictions, spectrogram


if __name__ == "__main__":
    import torchinfo

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    print(" - Initializing model:")
    encoder_input_dim = 94
    decoder_input_dim = 30
    synthesizer_input_dim = 94
    text_output_dim = 30
    audio_output_dim = 94

    seq2seq = AudioSeq2Seq(encoder_input_dim, decoder_input_dim, synthesizer_input_dim,
                           text_output_dim, audio_output_dim, device).to(device)

    # inspect model structure
    torchinfo.summary(seq2seq, input_size = [(7, 32), (7, 32)], dtypes=[torch.long, torch.long],
                      device=device)

    # inspect model parameters
    for name, param in seq2seq.named_parameters():
        print(name, param.data.shape)
