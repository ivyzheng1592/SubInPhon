# created 2025/01/15
# A script to load custom text dataset with self-defined class inherited from torch Dataset

from typing import Any, Iterable, List, Tuple, Dict

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


class Alphabet:
    def __init__(self, special_tokens: List[str]) -> None:
        self.idx2char = {} # {index: character}
        self.char2idx = {}  # {character: index}
        self.char2count = {}  # {char: number of occurrences}
        self.specials = special_tokens

    def __len__(self) -> int:
        return len(self.idx2char)

    # convert each word to a vector
    def word2vec(self, word: str) -> List[int]:
        word_vector = [self.char2idx["<SOS>"]]
        word_vector.extend([self.char2idx[char] if char in self.char2idx
                            else self.char2idx["<UNK>"]
                            for char in word])
        word_vector.append(self.char2idx["<EOS>"])
        return word_vector

    # convert each vector to a word
    def vec2word(self, vector: Iterable[int]) -> Tuple[List[str], str]:
        # in list format (including <SOS> and <EOS>)
        word_list = []
        for idx in vector:
            word_list.append(self.idx2char[idx])
            if self.idx2char[idx] == "<EOS>":
                break

        # in string format (excluding <SOS> and <EOS>)
        word_string = ""
        for char in word_list:
            if char not in self.specials:
                word_string += char
        return word_list, word_string

    # build vocabulary with a list of words and special characters
    def build_alphabet(self, words: Iterable[str]) -> None:
        # add special characters to vocabulary
        self.idx2char.update({idx: char for idx, char in enumerate(self.specials)})
        self.char2idx.update({char: idx for idx, char in enumerate(self.specials)})
        idx = len(self.specials)

        # add real characters to vocabulary
        for word in words:
            for char in word:
                if char not in self.char2idx:
                    self.idx2char[idx] = char
                    self.char2idx[char] = idx
                    self.char2count[char] = 1
                    idx += 1
                else:
                    self.char2count[char] += 1

    def fea2embed(self, feature_df: pd.DataFrame) -> torch.Tensor:
        # set all features of special characters to -1
        num_feature = feature_df.shape[1]
        num_special = len(self.specials)
        special_feature_tensor = torch.full((num_special, num_feature), -1)

        # sort dataframe with char2idx and convert to tensor
        feature_df = feature_df.rename(self.char2idx).sort_index()
        feature_tensor = torch.tensor(feature_df.values)
        embedding_tensor = torch.cat((special_feature_tensor, feature_tensor), dim=0)
        embedding_tensor = embedding_tensor.to(torch.float32)  # convert to type of pretrained weight

        return embedding_tensor

    def embed2fea(self, embedding_tensor: torch.Tensor) -> Dict[str, Any]:
        embedding_list = embedding_tensor.cpu().detach().numpy()
        # embedding_tensor = [input_dim, embedding_dim]

        # add embedding of each character to entire and focus feature space
        feature_space = {char: embedding_list[idx] for idx, char in self.idx2char.items()
                         if char not in self.specials}
        return feature_space

class TextDataset(Dataset):
    def __init__(self, annotations_file: str, special_tokens: List[str], device: str = 'cuda') -> None:
        self.device = device

        # get the list of ur and sr words
        self.annotations = pd.read_csv(annotations_file)
        self.ur_words = self.annotations["ur_string"]
        self.sr_words = self.annotations["sr_string"]

        # define special characters
        self.specials = special_tokens
        self.pad_idx = self.specials.index("<PAD>")

        # build ur alphabet
        self.ur_alphabet = Alphabet(self.specials)
        self.ur_alphabet.build_alphabet(self.ur_words)

        # build sr alphabet
        self.sr_alphabet = Alphabet(self.specials)
        self.sr_alphabet.build_alphabet(self.sr_words)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # get source word
        src_word = self.ur_words[index]
        src_vector = self.ur_alphabet.word2vec(src_word)
        src_tensor = torch.tensor(src_vector).to(self.device)

        # get target word
        trg_word = self.sr_words[index]
        trg_vector = self.sr_alphabet.word2vec(trg_word)
        trg_tensor = torch.tensor(trg_vector).to(self.device)

        return src_tensor, trg_tensor

    def split_dataset(self, data_split_ratio: List[float]) -> Any:
        return random_split(self, data_split_ratio)

    # Build the custom collate function used by this dataset's dataloader.
    def _get_collate_fn(self):
        def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
            srcs = [item[0] for item in batch]
            srcs = nn.utils.rnn.pad_sequence(srcs, batch_first=False, padding_value=self.pad_idx)

            trgs = [item[1] for item in batch]
            trgs = nn.utils.rnn.pad_sequence(trgs, batch_first=False, padding_value=self.pad_idx)
            return srcs, trgs
        return collate_fn

    def get_dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._get_collate_fn(),
            drop_last=True  # drop incomplete batch
        )
        return data_loader


if __name__ == "__main__":
    import hyper_params as hp
    from language_registry import languages

    print(" - Loading dataset:")
    language = languages[hp.lang_name]
    annotations_file = language.generate_stimuli(
        seed=hp.base_seed,
        sample_proportion=hp.data_proportion,
        property=hp.property,
        directionality=hp.directionality[0],
        output_dir="data",
    )[hp.conditions[0]]
    annotations = pd.read_csv(annotations_file)
    print(f"Dataset size: {len(annotations)}")
    print(f"Sample data token: {annotations.iloc[0]}")

    print(" - Building vocabulary:")
    text_dataset = TextDataset(annotations_file, hp.special_tokens, device='cpu')
    src, trg = text_dataset[0]
    print(f"UR vocab size: {len(text_dataset.ur_alphabet)}")
    print(f"UR vocab size: {len(text_dataset.sr_alphabet)}")
    print(f"Sample source: {src.shape}")
    print(f"Sample target: {trg.shape}")

    print(" - Creating dataloader:")
    text_dataloader = text_dataset.get_dataloader(text_dataset, hp.batch_size)
    dataiter = iter(text_dataloader)
    source, target = next(dataiter)
    print(f"Sample source: {source.shape}")
    print(f"Sample target: {target.shape}")
