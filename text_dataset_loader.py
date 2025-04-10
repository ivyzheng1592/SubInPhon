# created 2025/01/15
# A script to load custom text dataset with self-defined class inherited from torch Dataset

import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence


class Alphabet:
    def __init__(self, name, special_tokens):
        self.name = name
        self.idx2char = {} # {index: char}
        self.char2idx = {}  # {char: index}
        self.char2count = {}  # {char: number of occurrences}
        self.specials = special_tokens

    def __len__(self):
        return len(self.idx2char)

    # convert each word to a vector
    def word2vec(self, word):
        word_vector = [self.char2idx["<SOS>"]]
        word_vector.extend([self.char2idx[char] if char in self.char2idx
                            else self.char2idx["<UNK>"]
                            for char in word])
        word_vector.append(self.char2idx["<EOS>"])
        return word_vector

    # convert each vector to a word
    def vec2word(self, vector):
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
    def build_alphabet(self, words):
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


class TextDataset(Dataset):
    def __init__(self, annotations_file, special_tokens, device='cuda'):
        self.device = device

        # get the list of ur and sr words
        # and randomize word order for each instance of dataset
        self.annotations = pd.read_csv(annotations_file)
        self.annotations = self.annotations.sample(frac=1).reset_index(drop=True)
        self.ur = self.annotations["ur"]
        self.sr = self.annotations["sr"]

        # define special characters
        self.specials = special_tokens
        self.pad_idx = self.specials.index("<PAD>")

        # build ur alphabet
        self.ur_name = re.split('[/_.]', annotations_file)[2] + "_ur"
        self.ur_alphabet = Alphabet(self.ur_name, self.specials)
        self.ur_alphabet.build_alphabet(self.ur)

        # build sr alphabet
        self.sr_name = re.split('[/_.]', annotations_file)[2] + "_sr"
        self.sr_alphabet = Alphabet(self.sr_name, self.specials)
        self.sr_alphabet.build_alphabet(self.sr)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # get source word
        src = self.ur[index]
        src_vector = self.ur_alphabet.word2vec(src)
        src_tensor = torch.tensor(src_vector).to(self.device)

        # get target word
        trg = self.sr[index]
        trg_vector = self.sr_alphabet.word2vec(trg)
        trg_tensor = torch.tensor(trg_vector).to(self.device)

        return src_tensor, trg_tensor

    def split_dataset(self, data_split_ratio):
        return random_split(self, data_split_ratio)

    # a closure of customized collate_fn
    def get_collate_fn(self):
        def collate_fn(batch):
            srcs = [item[0] for item in batch]
            srcs = pad_sequence(srcs, batch_first=False, padding_value=self.pad_idx)

            trgs = [item[1] for item in batch]
            trgs = pad_sequence(trgs, batch_first=False, padding_value=self.pad_idx)
            return srcs, trgs
        return collate_fn

    def get_dataloader(self, batch_size, shuffle=True):
        collate_fn = self.get_collate_fn()

        data_loader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn
        )
        return data_loader


if __name__ == "__main__":
    import hyper_params as hp
    import utils

    print(" - Loading dataset:")
    annotations_file = "Dataset/EnglishBH_txt_harmony.csv"
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
    text_dataloader = text_dataset.get_dataloader(hp.batch_size)
    dataiter = iter(text_dataloader)
    source, target = next(dataiter)
    print(f"Sample source: {source.shape}")
    print(f"Sample target: {target.shape}")