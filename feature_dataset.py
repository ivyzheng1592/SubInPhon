# created 2025/01/15
# An instance of text dataset including feature embedding

from text_dataset import *


class FeatureDataset(TextDataset):
    def __init__(self, annotations_file, feature_file, special_tokens, device='cuda'):
        super().__init__(annotations_file, special_tokens, device)

        # read feature files
        feature_df = pd.read_excel(feature_file, sheet_name=0, index_col=0)

        # set all features of special characters to -1
        num_feature = feature_df.shape[1]
        num_special = len(self.specials)
        special_feature_tensor = torch.full((num_special, num_feature), -1)

        # get ur feature embedding
        ur_feature_df = feature_df.rename(self.ur_alphabet.char2idx).sort_index()
        ur_feature_tensor = torch.tensor(ur_feature_df.values)
        ur_embedding = torch.cat((special_feature_tensor, ur_feature_tensor), dim=0)
        self.ur_embedding = ur_embedding.to(torch.float32)  # convert to type of pretrained weight

        # get sr feature embedding
        sr_feature_df = feature_df.rename(self.sr_alphabet.char2idx).sort_index()
        sr_feature_tensor = torch.tensor(sr_feature_df.values)
        sr_embedding = torch.cat((special_feature_tensor, sr_feature_tensor), dim=0)
        self.sr_embedding = sr_embedding.to(torch.float32)  # convert to type of pretrained weight


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
    feature_dataloader = feature_dataset.get_dataloader(hp.batch_size)
    dataiter = iter(feature_dataloader)
    source, target = next(dataiter)
    print(f"Sample source: {source.shape}")
    print(f"Sample target: {target.shape}")