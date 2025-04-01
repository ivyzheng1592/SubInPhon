# 2025/01/15
# A script to load custom text dataset with self-defined class inherited from torch Dataset

import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence


class Alphabet:
    def __init__(self, name):
        self.name = name
        self.idx2char = {} # {index: char}
        self.char2idx = {}  # {char: index}
        self.char2count = {}  # {char: number of occurrences}

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
        word = [self.idx2char[idx] for idx in vector]
        return word

    # build vocabulary with a list of words and special characters
    def build_alphabet(self, words, specials):
        # add special characters to vocabulary
        self.idx2char.update({idx: char for idx, char in enumerate(specials)})
        self.char2idx.update({char: idx for idx, char in enumerate(specials)})
        idx = len(specials)

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
        self.annotations = pd.read_csv(annotations_file)
        self.ur = self.annotations["ur"]
        self.sr = self.annotations["sr"]

        # define special characters
        self.specials = special_tokens
        self.pad_idx = self.specials.index("<PAD>")

        # build ur alphabet
        self.ur_name = re.split('[/_.]', annotations_file)[2] + "_ur"
        self.ur_alphabet = Alphabet(self.ur_name)
        self.ur_alphabet.build_alphabet(self.ur, self.specials)

        # build sr alphabet
        self.sr_name = re.split('[/_.]', annotations_file)[2] + "_sr"
        self.sr_alphabet = Alphabet(self.sr_name)
        self.sr_alphabet.build_alphabet(self.sr, self.specials)

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

    print(" - Loading dataset:")
    annotations_file = "Dataset/English_txt_harmony.csv"
    annotations = pd.read_csv(annotations_file)
    print("Dataset size: ", len(annotations),
          "\nSample data token:", annotations.iloc[0])

    print(" - Building vocabulary:")
    text_dataset = TextDataset(annotations_file, hp.special_tokens, device='cpu')
    src, trg = text_dataset[0]
    print("UR vocab size:", len(text_dataset.ur_alphabet),
          "\nSR vocab size:", len(text_dataset.ur_alphabet),
          "\nSample source:", src.shape,
          "\nSample target:", trg.shape)

    print(" - Creating dataloader:")
    text_dataloader = text_dataset.get_dataloader(hp.batch_size)
    dataiter = iter(text_dataloader)
    source, target = next(dataiter)
    print("Sample source:", source.shape,
          "\nSample target:", target.shape)