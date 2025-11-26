# created 2025/01/15
# An instance of text dataset including feature embedding

from text_dataset import *


class FeatureDataset(TextDataset):
    def __init__(self, annotations_file, feature_file, special_tokens, device='cuda'):
        super().__init__(annotations_file, special_tokens, device)
        # get the dataframe of features
        self.feature_df = pd.read_excel(feature_file, sheet_name=0, index_col=0)

        # build ur alphabet and embedding
        self.ur_alphabet = Alphabet(self.specials)
        self.ur_alphabet.build_alphabet(self.ur_words)
        self.ur_embedding = self.ur_alphabet.fea2embed(self.feature_df)

        # build sr alphabet and embedding
        self.sr_alphabet = Alphabet(self.specials)
        self.sr_alphabet.build_alphabet(self.sr_words)
        self.sr_embedding = self.sr_alphabet.fea2embed(self.feature_df)


if __name__ == "__main__":
    import hyper_params as hp

    print(" - Loading dataset:")
    annotations_file = "Dataset/EnglishBH_txt_harmony.csv"
    feature_file = "Dataset/EnglishBH_features.xlsx"
    annotations = pd.read_csv(annotations_file)
    print(f"Dataset size: {len(annotations)}")
    print(f"Sample data token: {annotations.iloc[0]}")

    print(" - Building vocabulary:")
    feature_dataset = FeatureDataset(annotations_file, feature_file, hp.special_tokens, device='cpu')
    src, trg = feature_dataset[0]
    print(f"UR vocab size: {len(feature_dataset.ur_alphabet)}")
    print(f"UR vocab size: {len(feature_dataset.sr_alphabet)}")
    print(f"Sample source: {src.shape}")
    print(f"Sample target: {trg.shape}")
    ur_embedding_weight = feature_dataset.ur_embedding
    sr_embedding_weight = feature_dataset.sr_embedding
    print(f"Feature space of UR vocab: {ur_embedding_weight.shape}")
    print(f"Feature space of SR vocab: {sr_embedding_weight.shape}")

    print(" - Creating dataloader:")
    feature_dataloader = feature_dataset.get_dataloader(feature_dataset, hp.batch_size)
    dataiter = iter(feature_dataloader)
    source, target = next(dataiter)
    print(f"Sample source: {source.shape}")
    print(f"Sample target: {target.shape}")