"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  │  Transformer.infer(german_sentence)          → str             │
  └─────────────────────────────────────────────────────────────────┘

AUTOGRADER USAGE PATTERN:
    model = Transformer()          # zero-arg construction
    model.eval()
    english = model.infer(german)  # end-to-end single sentence
"""

import math
import copy
import os
from typing import Optional, Tuple
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#  GOOGLE DRIVE FILE ID
#  After training: upload checkpoint to Google Drive, get the file ID
#  from the share URL and paste it here before final submission.
#  Share URL looks like: drive.google.com/file/d/FILE_ID_HERE/view
# ══════════════════════════════════════════════════════════════════════
GDRIVE_FILE_ID    = "YOUR_GDRIVE_FILE_ID_HERE"   # ← replace after training
CHECKPOINT_FILENAME = "da6401_a3_checkpoint.pt"


# ══════════════════════════════════════════════════════════════════════
#  STANDALONE ATTENTION FUNCTION
#  Exposed at module level so the autograder can import & test it
#  independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    d_k    = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    attn_w = F.softmax(scores, dim=-1)
    attn_w = torch.nan_to_num(attn_w, nan=0.0)   # handle all-pad rows

    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    tgt_len = tgt.size(1)
    device  = tgt.device

    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)          # (B,1,1,T)
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)                                     # (1,1,T,T)

    return pad_mask | causal_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention, §3.2.2.  No torch.nn.MultiheadAttention used.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_k        = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.dropout      = nn.Dropout(p=dropout)
        self.attn_weights = None   # stored for Experiment 3 visualisation

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq, d_model) → (batch, heads, seq, d_k)"""
        B, S, _ = x.size()
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]; True → masked out

        Returns:
            output : shape [batch, seq_q, d_model]
        """
        B = query.size(0)

        Q = self._split_heads(self.W_Q(query))
        K = self._split_heads(self.W_K(key))
        V = self._split_heads(self.W_V(value))

        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)

        out, self.attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_O(out)


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding, §3.5.
    Registered as a buffer — NOT a trainable parameter.

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)

        self.register_buffer('pe', pe)   # buffer, not parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]
        Returns:
            Tensor of same shape [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise FFN, §3.3:
        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  (Post-LayerNorm — original paper style)
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """x → [Self-Attn → Add & Norm] → [FFN → Add & Norm]"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  DECODER LAYER  (Post-LayerNorm)
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    x → [Masked Self-Attn → Add & Norm]
      → [Cross-Attn(memory) → Add & Norm]
      → [FFN → Add & Norm]
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#  FULL TRANSFORMER
#
#  ★ ALL arguments have defaults → autograder can call Transformer()
#  ★ __init__ loads spaCy, builds vocab, builds arch, loads weights
#  ★ infer(german_str) → english_str  (no other arguments)
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for German→English translation.

    Designed so the autograder can do exactly:
        model = Transformer()
        model.eval()
        english = model.infer(german_sentence)

    __init__ handles everything internally:
      1. Loads spaCy German & English tokenisers
      2. Builds vocabularies from Multi30k training split
      3. Constructs the model architecture
      4. Downloads trained weights from Google Drive via gdown
      5. Loads those weights

    Args (all have defaults):
        d_model        : Model dimensionality              (default 512)
        N              : Encoder/decoder stack depth       (default 6)
        num_heads      : Attention heads                   (default 8)
        d_ff           : FFN inner dimension               (default 2048)
        dropout        : Dropout probability               (default 0.1)
        pad_idx        : <pad> token index                 (default 1)
        min_freq       : Min token freq for vocab          (default 2)
        max_infer_len  : Max tokens generated at inference (default 50)
        load_weights   : Download & load checkpoint        (default True)
        checkpoint_path: Local path for checkpoint file
    """

    def __init__(
        self,
        d_model:         int   = 512,
        N:               int   = 6,
        num_heads:       int   = 8,
        d_ff:            int   = 2048,
        dropout:         float = 0.1,
        pad_idx:         int   = 1,
        min_freq:        int   = 2,
        max_infer_len:   int   = 50,
        load_weights:    bool  = True,
        checkpoint_path: str   = CHECKPOINT_FILENAME,
    ) -> None:
        super().__init__()

        self.d_model       = d_model
        self.pad_idx       = pad_idx
        self.max_infer_len = max_infer_len

        # ── 1. spaCy tokenisers ───────────────────────────────────────
        self.nlp_de, self.nlp_en = self._load_spacy()

        # ── 2. Vocabularies ───────────────────────────────────────────
        self.src_vocab, self.tgt_vocab = self._build_vocabs(min_freq)
        src_vocab_size = len(self.src_vocab['stoi'])
        tgt_vocab_size = len(self.tgt_vocab['stoi'])
        print(f"[Transformer] vocab sizes — src: {src_vocab_size}  tgt: {tgt_vocab_size}")

        # ── 3. Architecture ───────────────────────────────────────────
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc   = PositionalEncoding(d_model, dropout)

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)

        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # Xavier uniform initialisation (paper convention)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # ── 4 & 5. Download + load checkpoint ─────────────────────────
        if load_weights:
            self._download_and_load(checkpoint_path)
        else:
            print("[Transformer] load_weights=False — using random weights.")

    # ── INTERNAL HELPERS ──────────────────────────────────────────────

    @staticmethod
    def _load_spacy():
        """Load de + en spaCy models, auto-downloading if missing."""
        import spacy
        def load_model(name):
            try:
                return spacy.load(name)
            except OSError:
                print(f"[Transformer] Downloading spaCy model '{name}'…")
                os.system(f"python -m spacy download {name}")
                import importlib
                importlib.reload(spacy)
                return spacy.load(name)
        return load_model('de_core_news_sm'), load_model('en_core_web_sm')

    def _build_vocabs(self, min_freq: int):
        """
        Load Multi30k train split, tokenise with spaCy,
        build stoi/itos dicts with specials at fixed indices:
            <unk>=0  <pad>=1  <sos>=2  <eos>=3
        """
        from datasets import load_dataset
        print("[Transformer] Building vocabularies from Multi30k train split…")
        raw = load_dataset('bentrevett/multi30k', split='train')

        src_tok_lists = [[t.text.lower() for t in self.nlp_de(ex['de'])] for ex in raw]
        tgt_tok_lists = [[t.text.lower() for t in self.nlp_en(ex['en'])] for ex in raw]

        def _make_vocab(tok_lists):
            counter = Counter()
            for toks in tok_lists:
                counter.update(toks)
            specials = ['<unk>', '<pad>', '<sos>', '<eos>']
            stoi = {t: i for i, t in enumerate(specials)}
            idx  = len(specials)
            for word, freq in counter.most_common():
                if freq >= min_freq and word not in stoi:
                    stoi[word] = idx
                    idx += 1
            itos = {i: t for t, i in stoi.items()}
            return {'stoi': stoi, 'itos': itos}

        return _make_vocab(src_tok_lists), _make_vocab(tgt_tok_lists)

    def _download_and_load(self, checkpoint_path: str):
        """Download checkpoint from Google Drive (if needed) then load it."""
        if not os.path.exists(checkpoint_path):
            print(f"[Transformer] Checkpoint not found locally. Downloading from Drive…")
            try:
                import gdown
                gdown.download(id=GDRIVE_FILE_ID, output=checkpoint_path, quiet=False)
            except Exception as e:
                print(f"[Transformer] WARNING — gdown failed: {e}")
                print("[Transformer] Continuing with random weights.")
                return

        if not os.path.exists(checkpoint_path):
            print("[Transformer] WARNING — checkpoint still missing after download attempt.")
            return

        print(f"[Transformer] Loading weights from '{checkpoint_path}'…")
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
            self.load_state_dict(state)
            print("[Transformer] Weights loaded successfully ✓")
        except Exception as e:
            print(f"[Transformer] WARNING — could not load weights: {e}")

    # ── AUTOGRADER HOOKS ──────────────────────────────────────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]
        Returns:
            memory : shape [batch, src_len, d_model]
        """
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.output_projection(x)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        """
        End-to-end German → English translation for a single sentence.

        Tokenises the input with spaCy internally, runs greedy
        autoregressive decoding, and returns the detokenised English string.

        Args:
            src_sentence : Raw German text string.
        Returns:
            Translated English string.
        """
        self.eval()
        device = next(self.parameters()).device

        src_stoi    = self.src_vocab['stoi']
        tgt_stoi    = self.tgt_vocab['stoi']
        tgt_itos    = self.tgt_vocab['itos']

        unk_idx     = src_stoi.get('<unk>', 0)
        src_sos     = src_stoi.get('<sos>', 2)
        src_eos     = src_stoi.get('<eos>', 3)
        tgt_sos     = tgt_stoi.get('<sos>', 2)
        tgt_eos     = tgt_stoi.get('<eos>', 3)

        # Tokenise German source
        tokens      = [tok.text.lower() for tok in self.nlp_de(src_sentence)]
        src_ids     = [src_sos] + [src_stoi.get(t, unk_idx) for t in tokens] + [src_eos]
        src_tensor  = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
        src_mask    = make_src_mask(src_tensor, pad_idx=self.pad_idx)

        with torch.no_grad():
            memory = self.encode(src_tensor, src_mask)
            ys     = torch.tensor([[tgt_sos]], dtype=torch.long, device=device)

            for _ in range(self.max_infer_len):
                tgt_mask = make_tgt_mask(ys, pad_idx=self.pad_idx)
                logits   = self.decode(memory, src_mask, ys, tgt_mask)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys       = torch.cat([ys, next_tok], dim=1)
                if next_tok.item() == tgt_eos:
                    break

        # Detokenise: skip <sos>, stop at <eos>
        out = []
        for idx in ys[0, 1:].tolist():
            if idx == tgt_eos:
                break
            out.append(tgt_itos.get(idx, '<unk>'))

        return ' '.join(out)