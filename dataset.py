"""
dataset.py — Multi30k Dataset Loader
DA6401 Assignment 3: "Attention Is All You Need"
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import spacy
from collections import Counter


# Special token indices (fixed)
UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]


class Vocabulary:
    """Simple vocabulary object mapping tokens <-> indices."""

    def __init__(self):
        self.stoi = {}   # string → index
        self.itos = []   # index → string

    def build(self, token_lists, min_freq=1):
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)

        self.itos = list(SPECIAL_TOKENS)
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}

        for token, freq in counter.most_common():
            if freq >= min_freq and token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    def __len__(self):
        return len(self.itos)

    def lookup_indices(self, tokens):
        return [self.stoi.get(tok, UNK_IDX) for tok in tokens]

    def lookup_token(self, idx):
        return self.itos[idx] if 0 <= idx < len(self.itos) else "<unk>"


class Multi30kDataset(Dataset):
    def __init__(self, split='train', src_vocab=None, tgt_vocab=None,
                 spacy_de=None, spacy_en=None):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split

        # Load dataset from Hugging Face
        raw = load_dataset("bentrevett/multi30k", split=split)
        self.raw_data = raw

        # Load spacy tokenizers
        if spacy_de is None:
            self.spacy_de = spacy.load("de_core_news_sm")
        else:
            self.spacy_de = spacy_de

        if spacy_en is None:
            self.spacy_en = spacy.load("en_core_web_sm")
        else:
            self.spacy_en = spacy_en

        # Tokenize all sentences
        self.src_tokens = [self.tokenize_de(ex["de"]) for ex in raw]
        self.tgt_tokens = [self.tokenize_en(ex["en"]) for ex in raw]

        # Build or reuse vocabularies
        if src_vocab is None:
            self.src_vocab = Vocabulary()
            self.src_vocab.build(self.src_tokens, min_freq=2)
        else:
            self.src_vocab = src_vocab

        if tgt_vocab is None:
            self.tgt_vocab = Vocabulary()
            self.tgt_vocab.build(self.tgt_tokens, min_freq=2)
        else:
            self.tgt_vocab = tgt_vocab

        # Convert to indices
        self.src_data = self.process_data(self.src_tokens, self.src_vocab)
        self.tgt_data = self.process_data(self.tgt_tokens, self.tgt_vocab)

    def tokenize_de(self, text):
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    def build_vocab(self):
        """Rebuild vocab from current token lists (for standalone use)."""
        self.src_vocab = Vocabulary()
        self.src_vocab.build(self.src_tokens, min_freq=2)
        self.tgt_vocab = Vocabulary()
        self.tgt_vocab.build(self.tgt_tokens, min_freq=2)

    def process_data(self, token_lists, vocab):
        """Convert token lists to index lists with <sos>/<eos> framing."""
        data = []
        for tokens in token_lists:
            indices = [SOS_IDX] + vocab.lookup_indices(tokens) + [EOS_IDX]
            data.append(torch.tensor(indices, dtype=torch.long))
        return data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]


def collate_fn(batch):
    """Pad sequences in a batch to the same length."""
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded


def build_dataloaders(batch_size=128, device="cpu"):
    """
    Build train/val/test DataLoaders for Multi30k.
    Returns (train_loader, val_loader, test_loader, src_vocab, tgt_vocab).
    """
    import spacy
    spacy_de = spacy.load("de_core_news_sm")
    spacy_en = spacy.load("en_core_web_sm")

    train_dataset = Multi30kDataset("train", spacy_de=spacy_de, spacy_en=spacy_en)
    src_vocab = train_dataset.src_vocab
    tgt_vocab = train_dataset.tgt_vocab

    val_dataset  = Multi30kDataset("validation", src_vocab=src_vocab,
                                   tgt_vocab=tgt_vocab,
                                   spacy_de=spacy_de, spacy_en=spacy_en)
    test_dataset = Multi30kDataset("test", src_vocab=src_vocab,
                                   tgt_vocab=tgt_vocab,
                                   spacy_de=spacy_de, spacy_en=spacy_en)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
