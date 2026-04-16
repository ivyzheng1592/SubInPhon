# created 2025/03/31
# updated 2026/04/16
# An LSTM encoder-decoder with a Translatotron 2 style acoustic synthesizer

import random
from typing import Any, Tuple

import torch
import torch.nn as nn

import hyper_params as hp


class AudioEncoder(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super(AudioEncoder, self).__init__()
        self.input_dim = input_dim
        self.prenet_dim = hp.prenet_dim
        self.hidden_dim = hp.audio_hidden_dim
        self.n_layers = hp.audio_n_layers
        self.dropout = hp.audio_dropout

        self.prenet = nn.Linear(input_dim, self.prenet_dim)
        self.rnn = nn.LSTM(
            self.prenet_dim,
            self.hidden_dim,
            num_layers=self.n_layers,
            bidirectional=True,
            dropout=self.dropout,
        )
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # input = [batch_size, n_channels=1, n_freq, input_len]
        
        input = input.squeeze(1).permute(2, 0, 1)
        # input = [input_len, batch_size, n_freq]
        prenet = self.dropout(torch.relu(self.prenet(input)))
        # prenet = [input_len, batch_size, prenet_dim]
        
        encoder_states, (hidden, cell) = self.rnn(prenet)
        # encoder_states = [input_len, batch_size, hidden_dim * 2]
        # hidden = [n_layers * 2, batch_size, hidden_dim]
        # cell = [n_layers * 2, batch_size, hidden_dim]

        return encoder_states, hidden, cell


class TextDecoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(TextDecoder, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = hp.embedding_dim
        self.hidden_dim = hp.audio_hidden_dim
        self.output_dim = output_dim
        self.n_layers = hp.text_n_layers
        self.dropout = hp.text_dropout
        assert self.input_dim == self.output_dim, (
            "Decoder input dimension and output dimension must be the same!"
        )

        self.embedding = nn.Embedding(input_dim, self.embedding_dim)
        self.rnn = nn.LSTM(
            self.hidden_dim * 2 + self.embedding_dim,
            self.hidden_dim,
            num_layers=self.n_layers,
        )
        self.fc_out = nn.Linear(self.hidden_dim * 3 + self.embedding_dim, output_dim)
        self.dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        input: torch.Tensor,
        context_vector: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # input = [batch_size]
        # context_vector = [1, batch_size, hidden_dim * 2]
        # hidden = [n_layers, batch_size, hidden_dim]
        # cell = [n_layers, batch_size, hidden_dim]
        
        input = input.unsqueeze(0)
        # input = [input_len=1, batch_size]
        embedding = self.dropout(self.embedding(input))
        # embedding = [1, batch_size, embedding_dim]
        
        rnn_input = torch.cat((embedding, context_vector), dim=2)
        # rnn_input = [1, batch_size, hidden_dim * 2 + embedding_dim]
        decoder_state, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # decoder_state = [1, batch_size, hidden_dim]
        # hidden = [n_layers, batch_size, hidden_dim]
        # cell = [n_layers, batch_size, hidden_dim]
        
        output = self.fc_out(torch.cat((embedding, decoder_state, context_vector), dim=2))
        # output = [1, batch_size, output_dim]
        output = output.squeeze(0)
        # output = [batch_size, output_dim]
        decoder_state = decoder_state.squeeze(0)
        # decoder_state = [batch_size, hidden_dim]
        
        return output, decoder_state, hidden, cell


class MultiheadAttention(nn.Module):
    def __init__(self, qdim: int, kdim: int, vdim: int) -> None:
        super(MultiheadAttention, self).__init__()
        self.qdim = qdim
        self.kdim = kdim
        self.vdim = vdim
        self.num_heads = hp.num_heads
        self.dropout = hp.audio_dropout

        self.q_proj = nn.Linear(self.qdim, self.kdim)
        self.mha = nn.MultiheadAttention(
            embed_dim=self.kdim,
            num_heads=self.num_heads,
            kdim=kdim,
            vdim=vdim,
            dropout=self.dropout,
        )

    def forward(self, encoder_states: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # encoder_states = [src_len, batch_size, hidden_dim * 2]
        # hidden = [n_layers, batch_size, hidden_dim]

        query = self.q_proj(hidden[-1])
        # query = [batch_size, kdim]
        query = query.unsqueeze(0)
        # query = [1, batch_size, hidden_dim * 2]

        # In AudioSeq2Seq, multihead attention is called as:
        # self.attention(encoder_states, hidden[-1:].contiguous())
        # so:
        # - q / query comes from the current top decoder hidden state
        # - k / key comes from all encoder audio states
        # - v / value also comes from all encoder audio states
        context_vector, weight = self.mha(query, encoder_states, encoder_states)
        # context_vector = [1, batch_size, hidden_dim * 2]
        # weight = [batch_size, 1, src_len]
        weight = weight.squeeze(1)
        # weight = [batch_size, src_len]
        
        return context_vector, weight


class DurationPredictor(nn.Module):
    def __init__(self) -> None:
        super(DurationPredictor, self).__init__()
        self.hidden_dim = hp.audio_hidden_dim
        self.n_layers = hp.aux_n_layers

        self.lstm = nn.LSTM(
            self.hidden_dim * 3,
            self.hidden_dim,
            num_layers=self.n_layers,
            bidirectional=True,
        )
        self.proj = nn.Linear(self.hidden_dim * 2, 1)
        self.softplus = nn.Softplus()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input = [txt_len, batch_size, hidden_dim * 3]
        
        durations, _ = self.lstm(input)
        # durations = [txt_len, batch_size, hidden_dim * 2]
        # Softplus keeps predicted durations positive.
        # The extra 1e-3 is a small floor so durations do not collapse too close to 0,
        # which makes Gaussian upsampling numerically more stable.
        durations = self.softplus(self.proj(durations)) + 1e-3
        # durations = [txt_len, batch_size, 1]
        
        return durations


class RangePredictor(nn.Module):
    def __init__(self) -> None:
        super(RangePredictor, self).__init__()
        self.hidden_dim = hp.audio_hidden_dim
        self.n_layers = hp.aux_n_layers

        self.lstm = nn.LSTM(
            self.hidden_dim * 3 + 1,
            self.hidden_dim,
            num_layers=self.n_layers,
            bidirectional=True,
        )
        self.proj = nn.Linear(self.hidden_dim * 2, 1)
        self.softplus = nn.Softplus()

    def forward(self, input: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        # input = [txt_len, batch_size, hidden_dim * 3]
        # durations = [txt_len, batch_size, 1]
        
        lstm_input = torch.cat((input, durations), dim=2)
        # lstm_input = [txt_len, batch_size, hidden_dim * 3 + 1]
        
        ranges, _ = self.lstm(lstm_input)
        # ranges = [txt_len, batch_size, hidden_dim * 2]
        # Softplus keeps predicted Gaussian widths positive.
        # The extra 1e-3 prevents widths from becoming too close to 0,
        # which would make the Gaussian weights overly sharp or unstable.
        ranges = self.softplus(self.proj(ranges)) + 1e-3
        # ranges = [txt_len, batch_size, 1]
        
        return ranges


class GaussianUpsampling(nn.Module):
    def __init__(self, device: Any) -> None:
        super(GaussianUpsampling, self).__init__()
        self.device = device

    def forward(
        self,
        input: torch.Tensor,
        aud_trg_len: int,
        durations: torch.Tensor,
        ranges: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # input = [txt_len, batch_size, hidden_dim * 3]
        # durations = [txt_len, batch_size, 1]
        # ranges = [txt_len, batch_size, 1]
        
        # Build one acoustic timeline [0, 1, ..., aud_trg_len - 1] that every text slot
        # will be aligned against.
        time = torch.arange(aud_trg_len, device=self.device, dtype=input.dtype).view(1, 1, aud_trg_len)
        # time = [1, 1, aud_trg_len]

        # Convert per-slot durations into the center position of each text slot
        # on the acoustic-frame timeline.
        centers = torch.cumsum(durations, dim=0) - 0.5 * durations
        # centers = [txt_len, batch_size, 1]
        
        # For every text slot and every acoustic frame, compute a Gaussian weight.
        # Nearby frames get high weight; distant frames get low weight.
        weights = torch.exp(-0.5 * ((time - centers) / ranges) ** 2)
        # weights = [txt_len, batch_size, aud_trg_len]
        # Normalize over text slots so each acoustic frame becomes a weighted average
        # of all text-slot representations.
        weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-8)
        # weights = [txt_len, batch_size, aud_trg_len]

        # Turn token-level representations into frame-level representations by taking
        # the weighted sum of all text slots at each acoustic frame.
        upsamples = torch.bmm(weights.permute(1, 2, 0), input.permute(1, 0, 2))
        # upsamples = [batch_size, aud_trg_len, hidden_dim * 3]
        
        return upsamples, weights


class AudioSynthesizer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(AudioSynthesizer, self).__init__()
        self.input_dim = input_dim
        self.prenet_dim = hp.prenet_dim
        self.hidden_dim = hp.audio_hidden_dim
        self.output_dim = output_dim
        self.n_layers = hp.audio_n_layers
        self.dropout = hp.audio_dropout

        self.prenet = nn.Linear(input_dim, self.prenet_dim)
        self.rnn = nn.LSTM(
            self.hidden_dim * 3 + self.prenet_dim,
            self.hidden_dim,
            num_layers=self.n_layers,
            dropout=self.dropout,
        )
        self.fc_out = nn.Linear(self.hidden_dim * 4 + self.prenet_dim, output_dim)
        self.dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        input: torch.Tensor,
        upsample: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # input = [batch_size, n_freq]
        # upsample = [1, batch_size, hidden_dim * 3]
        # hidden = [n_layers, batch_size, hidden_dim]
        # cell = [n_layers, batch_size, hidden_dim]
        
        input = input.unsqueeze(0)
        # input = [1, batch_size, n_freq]
        prenet = self.dropout(torch.relu(self.prenet(input)))
        # prenet = [1, batch_size, prenet_dim]
        
        rnn_input = torch.cat((prenet, upsample), dim=2)
        # rnn_input = [1, batch_size, hidden_dim * 3 + prenet_dim]
        synthesizer_state, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # synthesizer_state = [1, batch_size, hidden_dim]
        # hidden = [n_layers, batch_size, hidden_dim]
        # cell = [n_layers, batch_size, hidden_dim]
        
        output = self.fc_out(torch.cat((prenet, synthesizer_state, upsample), dim=2))
        # output = [1, batch_size, output_dim]
        output = output.squeeze(0)
        # output = [batch_size, output_dim]
        return output, hidden, cell


class Postnet(nn.Module):
    def __init__(self) -> None:
        super(Postnet, self).__init__()
        self.cnn_depth = hp.cnn_depth
        self.dropout = hp.audio_dropout

        self.conv = nn.ModuleList()
        self.norm = nn.ModuleList()
        for _ in range(self.cnn_depth):
            self.conv.append(nn.Conv2d(1, 1, kernel_size=(3, 3), padding="same"))
            self.norm.append(nn.BatchNorm2d(1))
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input = [batch_size, 1, aud_output_dim, aud_trg_len]

        output = input
        for i in range(self.cnn_depth):
            output = self.conv[i](output)
            output = self.norm[i](output)
            if i < self.cnn_depth - 1 or self.cnn_depth == 1:
                output = torch.tanh(output)
        output = self.dropout(output)
        return output


class AudioSeq2Seq(nn.Module):
    def __init__(
        self,
        encoder_input_dim: int,
        decoder_input_dim: int,
        synthesizer_input_dim: int,
        text_output_dim: int,
        audio_output_dim: int,
        device: torch.device,
    ) -> None:
        super(AudioSeq2Seq, self).__init__()
        self.device = device
        self.encoder_input_dim = encoder_input_dim
        self.decoder_input_dim = decoder_input_dim
        self.synthesizer_input_dim = synthesizer_input_dim
        self.hidden_dim = hp.audio_hidden_dim
        self.text_output_dim = text_output_dim
        self.audio_output_dim = audio_output_dim
        self.text_n_layers = hp.text_n_layers

        self.attention = MultiheadAttention(
            self.hidden_dim,
            self.hidden_dim * 2,
            self.hidden_dim * 2,
        ).to(self.device)
        self.encoder = AudioEncoder(encoder_input_dim).to(self.device)
        self.decoder = TextDecoder(decoder_input_dim, text_output_dim).to(self.device)
        self.duration_predictor = DurationPredictor().to(self.device)
        self.range_predictor = RangePredictor().to(self.device)
        self.gaussian_upsample = GaussianUpsampling(self.device).to(self.device)
        self.synthesizer = AudioSynthesizer(synthesizer_input_dim, audio_output_dim).to(self.device)
        self.postnet = Postnet().to(self.device)

    def forward(
        self,
        input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        txt_teacher_forcing: float = hp.text_teacher_forcing,
        aud_teacher_forcing: float = hp.audio_teacher_forcing,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        src_txt, src_aud, trg_txt, trg_aud = input
        # src_txt = [txt_src_len, batch_size]
        # src_aud = [batch_size, n_channels, n_freq, aud_src_len]
        # trg_txt = [txt_trg_len, batch_size]
        # trg_aud = [batch_size, n_channels, n_freq, aud_trg_len]

        encoder_states, _, _ = self.encoder(src_aud)

        # text decoder
        aud_src_len = src_aud.shape[3]
        txt_trg_len = trg_txt.shape[0]

        decoder_outputs = torch.zeros(txt_trg_len, hp.batch_size, self.text_output_dim, device=self.device)
        predictions = torch.zeros(txt_trg_len, hp.batch_size, dtype=torch.long, device=self.device)
        decoder_states = torch.zeros(txt_trg_len, hp.batch_size, self.hidden_dim, device=self.device)
        context_vectors = torch.zeros(txt_trg_len, hp.batch_size, self.hidden_dim * 2, device=self.device)
        decoder_attentions = torch.zeros(txt_trg_len, hp.batch_size, aud_src_len, device=self.device)
        # decoder_outputs = [txt_trg_len, batch_size, text_output_dim]
        # predictions = [txt_trg_len, batch_size]
        # decoder_states = [txt_trg_len, batch_size, hidden_dim]
        # context_vectors = [txt_trg_len, batch_size, hidden_dim * 2]
        # decoder_attentions = [txt_trg_len, batch_size, aud_src_len]

        decoder_input = trg_txt[0] # SOS token
        hidden = torch.zeros(self.text_n_layers, hp.batch_size, self.hidden_dim, device=self.device)
        cell = torch.zeros(self.text_n_layers, hp.batch_size, self.hidden_dim, device=self.device)

        for t in range(1, txt_trg_len):
            context_vector, weight = self.attention(encoder_states, hidden[-1:].contiguous())
            # context_vector = [1, batch_size, hidden_dim * 2]
            # weight = [batch_size, aud_src_len]
            
            decoder_output, decoder_state, hidden, cell = self.decoder(
                decoder_input, context_vector, hidden, cell
            )
            # decoder_output = [batch_size, text_output_dim]
            # decoder_state = [batch_size, hidden_dim]
            # hidden = [n_layers, batch_size, hidden_dim]
            # cell = [n_layers, batch_size, hidden_dim]
            best_guess = decoder_output.argmax(1)
            # best_guess = [batch_size]

            decoder_outputs[t] = decoder_output
            predictions[t] = best_guess
            decoder_states[t] = decoder_state
            context_vectors[t] = context_vector.squeeze(0)
            decoder_attentions[t] = weight

            decoder_input = trg_txt[t] if random.random() < txt_teacher_forcing else best_guess
            # input = [batch_size]

        # upsampler
        aud_trg_len = trg_aud.shape[3]

        # predict durations and ranges for each text token
        # then perform Gaussian upsampling to get the synthesizer input
        upsample_input = torch.cat((decoder_states, context_vectors), dim=2)
        # upsample_input = [txt_trg_len, batch_size, hidden_dim * 3]
        pred_durations = self.duration_predictor(upsample_input)
        # durations = [txt_len, batch_size, 1]
        pred_ranges = self.range_predictor(upsample_input, pred_durations)
        # ranges = [txt_len, batch_size, 1]
        upsamples, gaussian_weights = self.gaussian_upsample(
            upsample_input, aud_trg_len, pred_durations, pred_ranges
        )
        # gaussian_weights = [txt_trg_len, batch_size, aud_trg_len]
        # upsamples = [batch_size, aud_trg_len, hidden_dim * 3]

        # audio synthesizer
        synthesizer_outputs = torch.zeros(aud_trg_len, hp.batch_size, self.audio_output_dim).to(self.device)
        # synthesizer_outputs = [aud_trg_len, batch_size, aud_output_dim]
        
        synthesizer_input = torch.zeros(hp.batch_size, self.audio_output_dim, device=self.device)
        # synthesizer_input = [batch_size, n_freq]
        hidden = torch.zeros(self.synthesizer.n_layers, hp.batch_size, self.hidden_dim, device=self.device)
        cell = torch.zeros(self.synthesizer.n_layers, hp.batch_size, self.hidden_dim, device=self.device)
        # hidden = [n_layers, batch_size, hidden_dim]
        # cell = [n_layers, batch_size, hidden_dim]
        
        for t in range(aud_trg_len):
            upsample = upsamples[:, t, :].unsqueeze(0)
            # upsample = [1, batch_size, hidden_dim * 3]
            synthesizer_output, hidden, cell = self.synthesizer(synthesizer_input, upsample, hidden, cell)
            synthesizer_outputs[t] = synthesizer_output

            curr_frame = trg_aud[:, :, :, t].squeeze(1)
            synthesizer_input = curr_frame if random.random() < aud_teacher_forcing else synthesizer_output
            # input = [batch_size, n_freq]

        synthesizer_outputs = synthesizer_outputs.unsqueeze(1).permute(2, 1, 3, 0)
        # synthesizer_outputs = [aud_trg_len, 1, batch_size, aud_output_dim] ->
        # synthesizer_outputs = [batch_size, 1, aud_output_dim, aud_trg_len]

        # postnet
        postnet_outputs = self.postnet(synthesizer_outputs)
        postnet_outputs = postnet_outputs + synthesizer_outputs
        # postnet_outputs = [batch_size, 1, aud_output_dim, aud_trg_len]

        # Project the linguistic attention through Gaussian upsampling so the acoustic path
        # has a source-speech attention view aligned to output spectrogram frames.
        synthesizer_attentions = torch.einsum("tbo,tbs->obs", gaussian_weights, decoder_attentions)
        # synthesizer_attentions = [aud_trg_len, batch_size, aud_src_len]

        return (
            decoder_outputs,
            predictions,
            postnet_outputs,
            decoder_attentions,
            synthesizer_attentions,
        )


if __name__ == "__main__":
    import torchinfo

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    encoder_input_dim = hp.n_mels
    decoder_input_dim = 30
    synthesizer_input_dim = hp.n_mels
    text_output_dim = 30
    audio_output_dim = hp.n_mels

    seq2seq = AudioSeq2Seq(
        encoder_input_dim,
        decoder_input_dim,
        synthesizer_input_dim,
        text_output_dim,
        audio_output_dim,
        device,
    ).to(device)

    torchinfo.summary(
        seq2seq,
        input_size=[
            (32, 7),
            (32, 1, hp.n_mels, 94),
            (32, 7),
            (32, 1, hp.n_mels, 94),
        ],
        dtypes=[torch.long, torch.float, torch.long, torch.float],
        device=device,
    )

    for name, param in seq2seq.named_parameters():
        print(name, param.data.shape)
