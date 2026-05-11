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
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import pickle
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#  STANDALONE ATTENTION FUNCTION
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
               Positions where mask is True are MASKED OUT.

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)
    # Scaled scores: (..., seq_q, seq_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn_w = F.softmax(scores, dim=-1)
    # Replace NaN (rows that are all -inf) with 0
    attn_w = torch.nan_to_num(attn_w, nan=0.0)

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
    # (batch, src_len) → (batch, 1, 1, src_len)
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
    batch_size, tgt_len = tgt.size()

    # Padding mask: (batch, 1, 1, tgt_len)
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)

    # Causal (look-ahead) mask: upper triangle = True
    # shape (1, 1, tgt_len, tgt_len)
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)

    # Combine: position is masked if it's a pad OR a future token
    return pad_mask | causal_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

        # Store last attention weights for visualization
        self.attn_weights = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, seq, d_model] → [batch, heads, seq, d_k]"""
        batch, seq, _ = x.size()
        x = x.view(batch, seq, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, heads, seq, d_k] → [batch, seq, d_model]"""
        batch, _, seq, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq, self.d_model)

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
                    [batch, num_heads, seq_q, seq_k]

        Returns:
            output : shape [batch, seq_q, d_model]
        """
        Q = self._split_heads(self.W_q(query))   # [batch, h, seq_q, d_k]
        K = self._split_heads(self.W_k(key))     # [batch, h, seq_k, d_k]
        V = self._split_heads(self.W_v(value))   # [batch, h, seq_k, d_k]

        # Expand mask for heads dim if needed
        if mask is not None and mask.dim() == 4 and mask.size(1) == 1:
            mask = mask.expand(-1, self.num_heads, -1, -1)

        output, attn_w = scaled_dot_product_attention(Q, K, V, mask)
        self.attn_weights = attn_w.detach()      # save for visualization

        output = self._merge_heads(output)       # [batch, seq_q, d_model]
        return self.W_o(output)


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build PE table: shape [1, max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)   # [max_len, 1]
        # div_term shape [d_model/2]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # [1, max_len, d_model]

        # Register as buffer (not a trainable parameter)
        self.register_buffer("pe", pe)

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
    Position-wise Feed-Forward Network, §3.3:

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
#  ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer (Post-LayerNorm):
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    We use Post-LayerNorm as in the original paper.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # Sub-layer 1: Self-attention + Add & Norm
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Sub-layer 2: FFN + Add & Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


# ══════════════════════════════════════════════════════════════════════
#  DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer (Post-LayerNorm):
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn   = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn         = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1       = nn.LayerNorm(d_model)
        self.norm2       = nn.LayerNorm(d_model)
        self.norm3       = nn.LayerNorm(d_model)
        self.dropout     = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Sub-layer 1: Masked self-attention
        attn1 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))
        # Sub-layer 2: Cross-attention over encoder memory
        attn2 = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(attn2))
        # Sub-layer 3: FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape)

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
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for German→English translation.

    The __init__ signature is kept flexible so autograder can call:
        model = Transformer()
    with no arguments (all defaults).

    Per the announcement: vocabs, tokenizers, and weights are all
    loaded inside __init__.
    """

    # ── Google Drive file IDs — FILL THESE IN after you train ──────────
    # Weight file (.pt) drive ID:
    WEIGHT_DRIVE_ID  = "1evwsUXhuih8Ili9VW2dVT9a03_c-aOVj"
    # Vocab file (.pkl) drive ID (src_vocab + tgt_vocab pickled):
    VOCAB_DRIVE_ID   = "1jRW_mB6f4pekD52qd9PC4aM4AQ5iy2wk"
    # ────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        src_vocab_size: int   = None,   # inferred from vocab if None
        tgt_vocab_size: int   = None,
        d_model:   int   = 256,
        N:         int   = 3,
        num_heads: int   = 8,
        d_ff:      int   = 512,
        dropout:   float = 0.1,
        pad_idx:   int   = 1,
        checkpoint_path: str = None,
    ) -> None:
        super().__init__()

        self.d_model  = d_model
        self.pad_idx  = pad_idx

        # ── Step 1: load vocabs & tokenizers ─────────────────────────
        self._load_vocab_and_tokenizers()

        if src_vocab_size is None:
            src_vocab_size = len(self.src_vocab)
        if tgt_vocab_size is None:
            tgt_vocab_size = len(self.tgt_vocab)

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        # ── Step 2: build architecture ───────────────────────────────
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc   = PositionalEncoding(d_model, dropout)

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)

        self.projection = nn.Linear(d_model, tgt_vocab_size)

        # Weight tying: share embedding weights with projection
        # (optional but common practice)
        # self.projection.weight = self.tgt_embed.weight

        # Xavier/Glorot init
        self._init_weights()

        # ── Step 3: load trained weights ────────────────────────────
        self._load_weights(checkpoint_path)

    # ── internal helpers ─────────────────────────────────────────────

    def _load_vocab_and_tokenizers(self):
        """
        Download vocab file from Drive and load, or build fresh.
        Called inside __init__.
        """
        import spacy

        vocab_path = "vocab.pkl"

        # Try to download from Drive
        if not os.path.exists(vocab_path):
            try:
                if self.VOCAB_DRIVE_ID and self.VOCAB_DRIVE_ID != "YOUR_VOCAB_FILE_DRIVE_ID":
                    gdown.download(id=self.VOCAB_DRIVE_ID, output=vocab_path, quiet=False)
            except Exception:
                pass

        if os.path.exists(vocab_path):
            with open(vocab_path, "rb") as f:
                vocabs = pickle.load(f)
            self.src_vocab = vocabs["src_vocab"]
            self.tgt_vocab = vocabs["tgt_vocab"]
        else:
            # Build from training data on the fly
            from dataset import Multi30kDataset
            train_ds = Multi30kDataset("train")
            self.src_vocab = train_ds.src_vocab
            self.tgt_vocab = train_ds.tgt_vocab
            # Save for future use
            with open(vocab_path, "wb") as f:
                pickle.dump({"src_vocab": self.src_vocab,
                             "tgt_vocab": self.tgt_vocab}, f)

        # Load spacy tokenizers
        try:
            self.spacy_de = spacy.load("de_core_news_sm")
        except Exception:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "de_core_news_sm"])
            self.spacy_de = spacy.load("de_core_news_sm")

        try:
            self.spacy_en = spacy.load("en_core_web_sm")
        except Exception:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.spacy_en = spacy.load("en_core_web_sm")

    def _load_weights(self, checkpoint_path):
        """Download checkpoint from Drive and load weights."""
        weight_path = checkpoint_path or "best_model.pt"

        if not os.path.exists(weight_path):
            try:
                if self.WEIGHT_DRIVE_ID and self.WEIGHT_DRIVE_ID != "YOUR_WEIGHT_FILE_DRIVE_ID":
                    gdown.download(id=self.WEIGHT_DRIVE_ID, output=weight_path, quiet=False)
            except Exception as e:
                print(f"[Transformer] Warning: could not download weights: {e}")

        if os.path.exists(weight_path):
            ckpt = torch.load(weight_path, map_location="cpu")
            state = ckpt.get("model_state_dict", ckpt)
            self.load_state_dict(state, strict=False)
            print(f"[Transformer] Loaded weights from {weight_path}")
        else:
            print("[Transformer] No pretrained weights found; using random init.")

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── AUTOGRADER HOOKS ─────────────────────────────────────────────

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Run the full encoder stack.
        src: [batch, src_len]  →  memory: [batch, src_len, d_model]
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
        Run the full decoder stack + linear projection.
        Returns logits: [batch, tgt_len, tgt_vocab_size]
        """
        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.projection(x)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.
        Returns logits: [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        """
        Translates a German sentence to English using greedy decoding.

        Args:
            src_sentence: Raw German text string.

        Returns:
            Translated English string (clean, detokenized).
        """
        from dataset import PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX

        self.eval()
        device = next(self.parameters()).device

        # Tokenize German sentence
        tokens = [tok.text.lower() for tok in self.spacy_de.tokenizer(src_sentence)]
        indices = (
            [SOS_IDX]
            + [self.src_vocab.stoi.get(t, UNK_IDX) for t in tokens]
            + [EOS_IDX]
        )
        src = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
        src_mask = make_src_mask(src, pad_idx=PAD_IDX)

        with torch.no_grad():
            memory = self.encode(src, src_mask)

            # Greedy autoregressive decoding
            ys = torch.tensor([[SOS_IDX]], dtype=torch.long, device=device)
            max_len = src.size(1) + 50

            for _ in range(max_len):
                tgt_mask = make_tgt_mask(ys, pad_idx=PAD_IDX)
                logits = self.decode(memory, src_mask, ys, tgt_mask)
                # Pick last token prediction
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_token], dim=1)
                if next_token.item() == EOS_IDX:
                    break

        # Convert indices → tokens (skip SOS/EOS/PAD)
        skip = {SOS_IDX, EOS_IDX, PAD_IDX}
        out_tokens = [
            self.tgt_vocab.itos[idx]
            for idx in ys[0].tolist()
            if idx not in skip
        ]

        return " ".join(out_tokens)
