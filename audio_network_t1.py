# created 2025/03/31
# updated 2025/11/07
# An LSTM encoder-decoder with global attention that deals with audio input based on Translatotron

import torch
import torch.nn as nn
import random
import hyper_params as hp
from text_network import BahdanauAttention


class AudioEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AudioEncoder, self).__init__()

        self.input_dim = input_dim
        self.prenet_dim = hp.prenet_dim
        self.hidden_dim = hp.audio_hidden_dim
        self.n_layers = hp.audio_n_layers
        self.dropout = hp.audio_dropout

        self.prenet = nn.Linear(input_dim, self.prenet_dim)
        # reduce the dimensionality of the input spectrogram on the frequency domain
        self.rnn = nn.LSTM(self.prenet_dim, self.hidden_dim, self.n_layers, bidirectional=True)
        # input prenet, output hidden space
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
        prenet = self.dropout(torch.relu(self.prenet(input)))
        # prenet = [input_len, batch_size, prenet_dim]

        encoder_states, (hidden, cell) = self.rnn(prenet)
        # encoder_states = [input_len, batch_size, hidden_dim * 2]
        # hidden = [n_layers * 2, batch_size, hidden_dim]
        # cell = [n_layers * 2, batch_size, hidden_dim]

        # hidden[-2, :, : ] is the last of the forwards RNN
        # hidden[-1, :, : ] is the last of the backwards RNN
        hidden = self.fc_hidden(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        cell = self.fc_cell(torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1))
        # hidden = [batch_size, hidden_dim]
        # cell = [batch_size, hidden_dim]

        return encoder_states, hidden, cell


class TextDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TextDecoder, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = hp.embedding_dim
        self.hidden_dim = hp.audio_hidden_dim
        self.output_dim = output_dim
        self.n_layers = hp.text_n_layers
        self.dropout = hp.text_dropout
        assert (
            self.input_dim == self.output_dim
        ), "Decoder input dimension and output dimension must be the same!"

        self.embedding = nn.Embedding(input_dim, self.embedding_dim)
        # map the input vocabulary to a d-dimensional space
        self.rnn = nn.LSTM(self.hidden_dim * 2 + self.embedding_dim, self.hidden_dim, self.n_layers)
        # input context vector and embedding, output hidden space
        self.fc_out = nn.Linear(self.hidden_dim * 3 + self.embedding_dim, output_dim)
        # take into account context vector, decoder hidden, and embedding for the prediction
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, input, context_vector, hidden, cell):
        # input = [batch_size]
        # context_vector = [1, batch_size, hidden_dim * 2]
        # hidden = [n_layers, batch_size, hidden_dim]
        # cell = [n_layers, batch_size, hidden_dim]

        input = input.unsqueeze(0)
        # input = [input_len=1, batch_size]
        embedding = self.dropout(self.embedding(input))
        # embedding = [1, batch_size, embedding_dim]

        # concatenate the target embedding and the context vector as the rnn input
        rnn_input = torch.cat((embedding, context_vector), dim=2)
        # rnn_input = [1, batch_size, hidden_dim * 2 + embedding_dim]
        decoder_state, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # decoder_state = [1, batch_size, hidden_dim * n_layers]
        # hidden = [n_layers, batch_size, hidden_dim]
        # cell = [n_layers, batch_size, hidden_dim]

        # the original manuscript uses all of embedding, decoder state, and context vector for the prediction
        output = self.fc_out(torch.cat((embedding, decoder_state, context_vector), dim=2))
        # output = [1, batch_size, output_dim]
        output = output.squeeze(0)
        # output = [batch_size, output_dim]

        return output, hidden, cell


class MultiheadAttention(nn.Module):
    def __init__(self, qdim, kdim, vdim):
        super(MultiheadAttention, self).__init__()

        self.qdim = qdim # hidden_dim * n_layers
        self.kdim = kdim # hidden_dim * 2
        self.vdim = kdim # hidden_dim * 2
        self.num_heads = hp.num_heads

        self.mha = nn.MultiheadAttention(embed_dim=qdim, num_heads=self.num_heads,
                                         kdim=kdim, vdim=vdim)

    def forward(self, encoder_states, hidden):
        # encoder_states = [src_len, batch_size, hidden_dim * 2]
        # hidden = [n_layers, batch_size, hidden_dim]

        batch_size = hidden.shape[1]
        hidden = hidden.view(1, batch_size, -1)
        # hidden = [1, batch_size, hidden_dim * n_layers]

        context_vector, weight = self.mha(hidden, encoder_states, encoder_states)
        # context_vector = [1, batch_size, hidden_dim * n_layers]
        # weight = [batch_size, 1, src_len]
        weight = weight.squeeze(1)
        # weight = [batch_size, src_len]

        return context_vector, weight


class AudioSynthesizer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AudioSynthesizer, self).__init__()

        self.input_dim = input_dim
        self.prenet_dim = hp.prenet_dim
        self.hidden_dim = hp.audio_hidden_dim
        self.output_dim = output_dim
        self.n_layers = hp.audio_n_layers
        self.dropout = hp.audio_dropout

        self.prenet = nn.Linear(input_dim, self.prenet_dim)
        # reduce the dimensionality of the input spectrogram on the frequency domain
        self.rnn = nn.LSTM(self.hidden_dim * self.n_layers + self.prenet_dim, self.hidden_dim, self.n_layers)
        # input context vector and prenet, output hidden space
        self.fc_out = nn.Linear(self.hidden_dim * self.n_layers + self.hidden_dim + self.prenet_dim, output_dim)
        # take into account context vector, decoder hidden, and prenet for the prediction
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, input, context_vector, hidden, cell):
        # input = [batch_size, n_freq]
        # context_vector = [1, batch_size, hidden_dim * n_layers]
        # hidden = [n_layers, batch_size, hidden_dim]
        # cell = [n_layers, batch_size, hidden_dim]

        input = input.unsqueeze(0)
        # input = [1, batch_size, n_freq]
        prenet = self.dropout(torch.relu(self.prenet(input)))
        # prenet = [1, batch_size, prenet_dim]

        # concatenate the prenet and the context vector as the rnn input
        rnn_input = torch.cat((prenet, context_vector), dim=2)
        # rnn_input = [1, batch_size, hidden_dim * n_layers + prenet_dim]
        synthesizer_state, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # synthesizer_state = [1, batch_size, hidden_dim]
        # hidden = [n_layers, batch_size, hidden_dim]
        # cell = [n_layers, batch_size, hidden_dim]

        # the original manuscript uses all of embedding, decoder state, and context vector for the prediction
        output = self.fc_out(torch.cat((prenet, synthesizer_state, context_vector), dim=2))
        # output = [1, batch_size, output_dim]
        output = output.squeeze(0)
        # output = [batch_size, output_dim]

        return output, hidden, cell


class Postnet(nn.Module):
    def __init__(self):
        super(Postnet, self).__init__()

        self.cnn_depth = hp.cnn_depth
        self.dropout = hp.audio_dropout

        self.conv = nn.ModuleList()
        self.norm = nn.ModuleList()
        for i in range(self.cnn_depth):
            self.conv.append(nn.Conv2d(1, 1, (3, 3), padding='same'))
            self.norm.append(nn.BatchNorm2d(1))
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, input):

        for i in range(self.cnn_depth-1):
            output = self.conv[i](input)  # [batch_size, 1, aud_output_dim, aud_trg_len]
            output = self.norm[i](output)
            output = torch.tanh(output)

        output = self.conv[-1](output)  # [batch_size, 1, aud_output_dim, aud_trg_len]
        output = self.norm[-1](output)
        output = self.dropout(output)

        return output


class AudioSeq2Seq(nn.Module):
    def __init__(self, encoder_input_dim, decoder_input_dim, synthesizer_input_dim,
                 text_output_dim, audio_output_dim, device):
        super(AudioSeq2Seq, self).__init__()

        self.device = device
        self.encoder_input_dim = encoder_input_dim
        self.decoder_input_dim = decoder_input_dim
        self.synthesizer_input_dim = synthesizer_input_dim
        self.hidden_dim = hp.audio_hidden_dim
        self.text_output_dim = text_output_dim
        self.audio_output_dim = audio_output_dim
        self.text_n_layers = hp.text_n_layers
        self.audio_n_layers = hp.audio_n_layers

        self.single_attention = BahdanauAttention(self.hidden_dim).to(self.device)
        self.multi_attention = MultiheadAttention(self.hidden_dim * self.audio_n_layers, self.hidden_dim * 2, self.hidden_dim * 2).to(self.device)
        self.encoder = AudioEncoder(encoder_input_dim).to(self.device)
        self.decoder = TextDecoder(decoder_input_dim, text_output_dim).to(self.device)
        self.synthesizer = AudioSynthesizer(synthesizer_input_dim, audio_output_dim).to(self.device)
        self.postnet = Postnet().to(self.device)

    def forward(self, input, txt_teacher_forcing=hp.text_teacher_forcing, aud_teacher_forcing=hp.audio_teacher_forcing):
        src_txt, src_aud, trg_txt, trg_aud = input
        # src_txt = [txt_src_len, batch_size]
        # src_aud = [batch_size, n_channels, freq, aud_src_len]
        # trg_txt = [txt_trg_len, batch_size]
        # trg_aud = [batch_size, n_channels, freq, aud_trg_len]

        encoder_states, _, _ = self.encoder(src_aud)
        # encoder_states = [aud_src_len, batch_size, hidden_dim * 2]

        # text decoder
        txt_trg_len = trg_txt.shape[0]
        batch_size = trg_txt.shape[1]
        decoder_outputs = torch.zeros(txt_trg_len, batch_size, self.text_output_dim).to(self.device)
        decoder_predictions = torch.zeros(txt_trg_len, batch_size).to(self.device)
        # decoder_outputs = [txt_trg_len, batch_size, txt_output_dim]
        # decoder_predictions = [txt_trg_len, batch_size]
        aud_src_len = src_aud.shape[3]
        decoder_attentions = torch.zeros(txt_trg_len, batch_size, aud_src_len).to(self.device)
        # decoder_attentions = [txt_trg_len, batch_size, aud_src_len]

        decoder_input = trg_txt[0]  # first input text to the decoder is the <SOS> token
        hidden = torch.zeros(self.text_n_layers, batch_size, self.hidden_dim).to(self.device)
        cell = torch.zeros(self.text_n_layers, batch_size, self.hidden_dim).to(self.device)
        for t in range(1, txt_trg_len):

            # at every time step,
            # calculate the attention weight with the current decoder hidden (query) and all encoder hidden (key, value)
            context_vector, weight = self.single_attention(encoder_states, hidden)
            # context_vector = [1, batch_size, hidden_dim * 2]
            # weight = [batch_size, aud_src_len]

            # insert input token, context vector, and previous hidden and cell
            # receive output, decoder state and new hidden and cell
            # and get the best word predicted by the decoder
            decoder_output, hidden, cell = self.decoder(decoder_input, context_vector, hidden, cell)
            # output = [batch_size, output_dim]
            # hidden = [n_layers, batch_size, hidden_dim]
            # cell = [n_layers, batch_size, hidden_dim]
            best_guess = decoder_output.argmax(1)
            # best_guess = [batch_size]

            # store decoder output, best guess and attention weight for current time step
            decoder_outputs[t] = decoder_output
            decoder_predictions[t] = best_guess
            decoder_attentions[t] = weight

            # with probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the decoder predicted it to be
            # Teacher Forcing is used so that the model gets used to seeing similar inputs at training and testing time
            decoder_input = trg_txt[t] if random.random() < txt_teacher_forcing else best_guess
            # input = [batch_size]

        # audio synthesizer
        aud_trg_len = trg_aud.shape[3]
        synthesizer_outputs = torch.zeros(aud_trg_len, batch_size, self.audio_output_dim).to(self.device)
        # synthesizer_outputs = [aud_trg_len, batch_size, aud_output_dim]
        synthesizer_attentions = torch.zeros(aud_trg_len, batch_size, aud_src_len).to(self.device)
        # attentions = [aud_trg_len, batch_size, aud_src_len]

        # first input spectrogram frame are zeros
        synthesizer_input = torch.zeros(batch_size, self.audio_output_dim).to(self.device)
        hidden = torch.zeros(self.audio_n_layers, batch_size, self.hidden_dim).to(self.device)
        cell = torch.zeros(self.audio_n_layers, batch_size, self.hidden_dim).to(self.device)
        for t in range(1, aud_trg_len):
            # at every time step,
            # calculate the attention weight with the current decoder hidden (query) and all encoder hidden (key, value)
            context_vector, weight = self.multi_attention(encoder_states, hidden)
            # context_vector = [1, batch_size, hidden_dim * 2]
            # weight = [batch_size, aud_src_len]

            # insert input frame, context vector, and previous hidden and cell
            # receive output and new hidden and cell
            synthesizer_output, hidden, cell = self.synthesizer(synthesizer_input, context_vector, hidden, cell)
            # output = [batch_size, output_dim]
            # hidden = [n_layers, batch_size, hidden_dim]
            # cell = [n_layers, batch_size, hidden_dim]

            # store decoder output and attention weight for current time step
            synthesizer_outputs[t] = synthesizer_output
            synthesizer_attentions[t] = weight

            # with probability of teacher_force_ratio we take the actual next frame
            # otherwise we take the frame that the decoder predicted it to be
            curr_frame = trg_aud[:, :, :, t].squeeze(1)
            synthesizer_input = curr_frame if random.random() < aud_teacher_forcing else synthesizer_output
            # input = [batch_size, n_freq]

        synthesizer_outputs = synthesizer_outputs.unsqueeze(1).permute(2, 1, 3, 0)
        # synthesizer_outputs = [aud_trg_len, 1, batch_size, aud_output_dim] ->
        # synthesizer_outputs = [batch_size, 1, aud_output_dim, aud_trg_len]

        # postnet
        postnet_outputs = self.postnet(synthesizer_outputs)
        postnet_outputs = postnet_outputs + synthesizer_outputs

        return decoder_outputs, decoder_predictions, postnet_outputs, decoder_attentions, synthesizer_attentions


if __name__ == "__main__":
    import torchinfo

    print(" - Initializing model:")
    encoder_input_dim = 128
    decoder_input_dim = 30
    synthesizer_input_dim = 128
    text_output_dim = 30
    audio_output_dim = 128

    seq2seq = AudioSeq2Seq(encoder_input_dim, decoder_input_dim, synthesizer_input_dim,
                           text_output_dim, audio_output_dim, 'cpu').to('cpu')

    # inspect model structure
    torchinfo.summary(seq2seq, input_size = [(8, 32), (32, 1, 128, 94), (8, 32), (32, 1, 128, 94)],
                      dtypes=[torch.long, torch.float32, torch.long, torch.float32],
                      device='cpu')

    # inspect model parameters
    for name, param in seq2seq.named_parameters():
        print(name, param.data.shape)
