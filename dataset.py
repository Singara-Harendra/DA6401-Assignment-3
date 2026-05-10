'''"""
dataset.py
"""
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from collections import Counter, OrderedDict

# Special token constants
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ['<unk>', '<pad>', '<sos>', '<eos>']

# ══════════════════════════════════════════════════════════════════════
#  CUSTOM VOCAB BUILDER (Replaces torchtext)
# ══════════════════════════════════════════════════════════════════════
class SimpleVocab:
    """A pure-Python replacement for torchtext.vocab.Vocab"""
    def __init__(self, token_to_idx, idx_to_token):
        self.stoi = token_to_idx
        self.itos = idx_to_token
        self.default_index = 0

    def __getitem__(self, token):
        return self.stoi.get(token, self.default_index)

    def set_default_index(self, index):
        self.default_index = index

    def get_stoi(self):
        return self.stoi

    def get_itos(self):
        return self.itos

    def __len__(self):
        return len(self.itos)

def build_vocab_from_iterator(iterator, min_freq=1, specials=None):
    """Builds a vocabulary dictionary from an iterator of tokens."""
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)

    # Sort by frequency, then alphabetically
    sorted_tokens = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    token_to_idx = OrderedDict()
    idx_to_token = []
    idx = 0

    if specials:
        for tok in specials:
            token_to_idx[tok] = idx
            idx_to_token.append(tok)
            idx += 1

    for tok, freq in sorted_tokens:
        if freq >= min_freq:
            if tok not in token_to_idx:
                token_to_idx[tok] = idx
                idx_to_token.append(tok)
                idx += 1

    return SimpleVocab(token_to_idx, idx_to_token)


# ══════════════════════════════════════════════════════════════════════
#  DATASET & DATALOADER
# ══════════════════════════════════════════════════════════════════════

class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """Pads batches dynamically to the longest sequence in the batch."""
    src_batch, tgt_batch = [], []
    for src_item, tgt_item in batch:
        src_batch.append(torch.tensor(src_item, dtype=torch.long))
        tgt_batch.append(torch.tensor(tgt_item, dtype=torch.long))
    
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch, tgt_batch

class Multi30kDataset:
    def __init__(self):
        print("Loading Spacy tokenizers...")
        self.spacy_de = spacy.load('de_core_news_sm')
        self.spacy_en = spacy.load('en_core_web_sm')
        
        print("Loading HuggingFace Multi30k dataset...")
        self.dataset = load_dataset("bentrevett/multi30k")
        
        self.src_vocab = None
        self.tgt_vocab = None
        self.processed_data = {}

    def tokenize_de(self, text):
        return [tok.text for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        return [tok.text for tok in self.spacy_en.tokenizer(text)]

    def yield_tokens(self, data_iter, language):
        for item in data_iter:
            if language == 'de':
                yield self.tokenize_de(item['de'])
            else:
                yield self.tokenize_en(item['en'])

    def build_vocab(self):
        print("Building source and target vocabularies...")
        train_data = self.dataset['train']
        
        self.src_vocab = build_vocab_from_iterator(
            self.yield_tokens(train_data, 'de'),
            min_freq=2,
            specials=SPECIAL_TOKENS
        )
        self.src_vocab.set_default_index(UNK_IDX)

        self.tgt_vocab = build_vocab_from_iterator(
            self.yield_tokens(train_data, 'en'),
            min_freq=2,
            specials=SPECIAL_TOKENS
        )
        self.tgt_vocab.set_default_index(UNK_IDX)

    def process_data(self):
        print("Numericalizing dataset...")
        for split in ['train', 'validation', 'test']:
            dict_split = 'val' if split == 'validation' else split
            self.processed_data[dict_split] = []
            
            for item in self.dataset[split]:
                src_toks = self.tokenize_de(item['de'])
                tgt_toks = self.tokenize_en(item['en'])
                
                src_tensor = [SOS_IDX] + [self.src_vocab[tok] for tok in src_toks] + [EOS_IDX]
                tgt_tensor = [SOS_IDX] + [self.tgt_vocab[tok] for tok in tgt_toks] + [EOS_IDX]
                
                self.processed_data[dict_split].append((src_tensor, tgt_tensor))

    def get_dataloader(self, split, batch_size, shuffle):
        data = self.processed_data[split]
        ds = TranslationDataset(data)
        return DataLoader(
            ds, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            collate_fn=collate_fn,
            drop_last=False
        )'''


"""
dataset.py — Multi30k Dataset Loading & Vocabulary
DA6401 Assignment 3: "Attention Is All You Need"

Loads the bentrevett/multi30k dataset from HuggingFace,
tokenises with spaCy, and builds word-index vocabularies.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from datasets import load_dataset
import spacy


# ══════════════════════════════════════════════════════════════════════
#  VOCABULARY
# ══════════════════════════════════════════════════════════════════════

class Vocabulary:
    """Simple word ↔ index vocabulary with special tokens."""

    PAD_TOKEN = '<pad>'   # index 1 (matches model default pad_idx)
    UNK_TOKEN = '<unk>'   # index 0
    SOS_TOKEN = '<sos>'   # index 2
    EOS_TOKEN = '<eos>'   # index 3

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.stoi: dict = {}   # string → index
        self.itos: dict = {}   # index  → string

    def build(self, token_lists):
        """
        Build vocab from a list of token lists.
        Reserves indices 0-3 for special tokens.
        """
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)

        # Special tokens first (fixed indices)
        specials = [self.UNK_TOKEN, self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        self.stoi = {tok: idx for idx, tok in enumerate(specials)}

        # Regular tokens sorted by frequency
        idx = len(specials)
        for word, freq in counter.most_common():
            if freq >= self.min_freq and word not in self.stoi:
                self.stoi[word] = idx
                idx += 1

        self.itos = {idx: tok for tok, idx in self.stoi.items()}

    def __len__(self):
        return len(self.stoi)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.stoi[self.UNK_TOKEN])

    def get(self, token: str, default=None):
        return self.stoi.get(token, default)

    def lookup_token(self, idx: int) -> str:
        return self.itos.get(idx, self.UNK_TOKEN)

    @property
    def pad_idx(self):
        return self.stoi[self.PAD_TOKEN]

    @property
    def sos_idx(self):
        return self.stoi[self.SOS_TOKEN]

    @property
    def eos_idx(self):
        return self.stoi[self.EOS_TOKEN]


# ══════════════════════════════════════════════════════════════════════
#  MULTI30K DATASET
# ══════════════════════════════════════════════════════════════════════

class Multi30kDataset(Dataset):
    """
    Wraps the bentrevett/multi30k HuggingFace dataset.

    German (de) → English (en) translation.
    Tokenisation via spaCy.
    """

    def __init__(
        self,
        split: str = 'train',
        src_vocab: Vocabulary = None,
        tgt_vocab: Vocabulary = None,
        min_freq: int = 2,
    ):
        """
        Args:
            split     : 'train', 'validation', or 'test'
            src_vocab : Pre-built source vocab (pass None to build from train)
            tgt_vocab : Pre-built target vocab (pass None to build from train)
            min_freq  : Minimum token frequency for vocab inclusion
        """
        self.split = split

        # Load spaCy tokenisers
        try:
            self.nlp_de = spacy.load('de_core_news_sm')
        except OSError:
            raise OSError(
                "German spaCy model not found. Run:\n"
                "  python -m spacy download de_core_news_sm"
            )
        try:
            self.nlp_en = spacy.load('en_core_web_sm')
        except OSError:
            raise OSError(
                "English spaCy model not found. Run:\n"
                "  python -m spacy download en_core_web_sm"
            )

        # Load HuggingFace dataset
        raw = load_dataset('bentrevett/multi30k', split=split)
        self.raw_data = raw  # list-like of {'en': str, 'de': str}

        # Tokenise all sentences
        self.src_tokens = [self._tokenize_de(ex['de']) for ex in self.raw_data]
        self.tgt_tokens = [self._tokenize_en(ex['en']) for ex in self.raw_data]

        # Build or reuse vocabularies
        if src_vocab is None:
            self.src_vocab = Vocabulary(min_freq=min_freq)
            self.src_vocab.build(self.src_tokens)
        else:
            self.src_vocab = src_vocab

        if tgt_vocab is None:
            self.tgt_vocab = Vocabulary(min_freq=min_freq)
            self.tgt_vocab.build(self.tgt_tokens)
        else:
            self.tgt_vocab = tgt_vocab

        # Convert to index sequences
        self.src_data, self.tgt_data = self._process()

    # ── Tokenisers ────────────────────────────────────────────────

    def _tokenize_de(self, text: str):
        return [tok.text.lower() for tok in self.nlp_de(text)]

    def _tokenize_en(self, text: str):
        return [tok.text.lower() for tok in self.nlp_en(text)]

    # ── Index conversion ──────────────────────────────────────────

    def _to_indices(self, tokens, vocab: Vocabulary):
        return (
            [vocab.sos_idx]
            + [vocab[t] for t in tokens]
            + [vocab.eos_idx]
        )

    def _process(self):
        src_data = [
            torch.tensor(self._to_indices(t, self.src_vocab), dtype=torch.long)
            for t in self.src_tokens
        ]
        tgt_data = [
            torch.tensor(self._to_indices(t, self.tgt_vocab), dtype=torch.long)
            for t in self.tgt_tokens
        ]
        return src_data, tgt_data

    # ── PyTorch Dataset interface ─────────────────────────────────

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]


# ══════════════════════════════════════════════════════════════════════
#  COLLATE & DATALOADER HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_collate_fn(pad_idx: int):
    """Returns a collate function that pads variable-length sequences."""
    def collate_fn(batch):
        src_batch, tgt_batch = zip(*batch)
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
        return src_padded, tgt_padded
    return collate_fn


def build_dataloaders(
    batch_size: int = 128,
    min_freq: int = 2,
    num_workers: int = 0,
):
    """
    Build train / validation / test DataLoaders for Multi30k.

    Returns:
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    """
    print("Loading train split…")
    train_ds = Multi30kDataset(split='train', min_freq=min_freq)

    print("Loading validation split…")
    val_ds = Multi30kDataset(
        split='validation',
        src_vocab=train_ds.src_vocab,
        tgt_vocab=train_ds.tgt_vocab,
    )

    print("Loading test split…")
    test_ds = Multi30kDataset(
        split='test',
        src_vocab=train_ds.src_vocab,
        tgt_vocab=train_ds.tgt_vocab,
    )

    src_vocab = train_ds.src_vocab
    tgt_vocab = train_ds.tgt_vocab
    pad_idx   = src_vocab.pad_idx
    collate   = make_collate_fn(pad_idx)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        collate_fn=collate, num_workers=num_workers
    )

    print(f"Source vocab size : {len(src_vocab)}")
    print(f"Target vocab size : {len(tgt_vocab)}")
    print(f"Train / Val / Test: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab