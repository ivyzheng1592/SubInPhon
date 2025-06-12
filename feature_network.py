# created 2025/06/08
# An instance of encoder-decoder network including pretrained feature embeddings

from text_network import *


class FeatureEncoder(TextEncoder):
    def __init__(self, input_dim, embedding_dim, hidden_dim, embedding_weight, n_layers, dropout, freeze):
        super().__init__(input_dim, embedding_dim, hidden_dim, n_layers, dropout)
        self.embedding = self.embedding.from_pretrained(embedding_weight, freeze=freeze)  # parse pretrained weight to embedding


class FeatureDecoder(TextDecoder):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, embedding_weight,
                 n_layers, dropout, freeze, attention):
        super().__init__(input_dim, embedding_dim, hidden_dim, output_dim,
                         n_layers, dropout, attention)
        self.embedding = self.embedding.from_pretrained(embedding_weight, freeze=freeze)


class FeatureSeq2Seq(TextSeq2Seq):
    def __init__(self, encoder_input_dim, decoder_input_dim, encoder_embedding_dim, decoder_embedding_dim,
                 encoder_embedding_weight, decoder_embedding_weight, n_layers, hidden_dim, output_dim,
                 encoder_dropout, decoder_dropout, freeze, device='cuda'):
        super().__init__(encoder_input_dim, decoder_input_dim, encoder_embedding_dim, decoder_embedding_dim,
                         n_layers, hidden_dim, output_dim, encoder_dropout, decoder_dropout, device)
        self.encoder_embedding_weight = encoder_embedding_weight
        self.decoder_embedding_weight = decoder_embedding_weight
        self.freeze = freeze

        # model components
        self.encoder = FeatureEncoder(encoder_input_dim, encoder_embedding_dim, hidden_dim,
                                      encoder_embedding_weight, n_layers, encoder_dropout, freeze).to(self.device)
        self.decoder = FeatureDecoder(decoder_input_dim, decoder_embedding_dim, hidden_dim, output_dim,
                                      decoder_embedding_weight, n_layers, decoder_dropout, freeze,
                                      self.attention).to(self.device)


if __name__ == "__main__":
    import torchinfo
    from feature_dataset import FeatureDataset

    print(" - Loading dataset:")
    annotations_file = "Dataset/EnglishBH_txt_harmony.csv"
    feature_file = "Dataset/EnglishBH_features.xlsx"
    feature_dataset = FeatureDataset(annotations_file, feature_file, hp.special_tokens, device='cpu')
    encoder_embedding_weight = feature_dataset.ur_embedding
    decoder_embedding_weight = feature_dataset.sr_embedding

    print(" - Initializing model:")
    encoder_input_dim = 30
    decoder_input_dim = 30
    output_dim = 30

    seq2seq = FeatureSeq2Seq(encoder_input_dim, decoder_input_dim,
                             hp.encoder_embedding_dim, hp.decoder_embedding_dim,
                             encoder_embedding_weight, decoder_embedding_weight,
                             hp.n_layers, hp.hidden_dim, output_dim,
                             hp.encoder_dropout, hp.decoder_dropout,
                             freeze=False, device='cpu').to('cpu')

    # inspect model structure
    torchinfo.summary(seq2seq, input_size=[(8, 32), (8, 32)], dtypes=[torch.long, torch.long],
                      device='cpu')

    # inspect model parameters
    for name, param in seq2seq.named_parameters():
        print(name, param.data)