"""
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
        )