"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

import wandb
#from evaluate import load as load_metric

from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need".

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    The <pad> position always gets 0 probability.
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]
            target : shape [batch * tgt_len]

        Returns:
            Scalar loss value (mean over non-pad tokens).
        """
        # Build smooth distribution
        with torch.no_grad():
            smooth_dist = torch.full_like(
                logits, self.smoothing / (self.vocab_size - 2)
            )
            # Correct-token position gets confidence
            smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            # PAD gets 0
            smooth_dist[:, self.pad_idx] = 0.0
            # Mask pad positions completely (don't contribute to loss)
            pad_mask = target == self.pad_idx
            smooth_dist[pad_mask] = 0.0

        log_probs = nn.functional.log_softmax(logits, dim=-1)
        loss = -(smooth_dist * log_probs).sum(dim=-1)

        # Average over non-pad tokens
        n_non_pad = (~pad_mask).sum().float().clamp(min=1)
        return loss.sum() / n_non_pad


# ══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Returns:
        avg_loss : Average loss over the epoch (float).
    """
    model.train() if is_train else model.eval()

    total_loss  = 0.0
    total_tokens = 0
    pad_idx = model.pad_idx

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for batch_idx, (src, tgt) in enumerate(data_iter):
            src = src.to(device)   # [batch, src_len]
            tgt = tgt.to(device)   # [batch, tgt_len]

            # Teacher-forcing: decoder input is tgt[:-1], label is tgt[1:]
            tgt_input  = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = make_src_mask(src, pad_idx)
            tgt_mask = make_tgt_mask(tgt_input, pad_idx)

            # Forward pass
            logits = model(src, tgt_input, src_mask, tgt_mask)
            # logits: [batch, tgt_len-1, vocab_size]

            # Flatten for loss
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.contiguous().view(-1, vocab_size)
            target_flat = tgt_output.contiguous().view(-1)

            loss = loss_fn(logits_flat, target_flat)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping (standard for Transformer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            # Accumulate
            n_tokens = (tgt_output != pad_idx).sum().item()
            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens

            # W&B step-level logging (train only)
            if is_train and wandb.run is not None:
                step = epoch_num * len(data_iter) + batch_idx
                current_lr = optimizer.param_groups[0]["lr"] if optimizer else 0.0
                wandb.log({
                    "train/step_loss": loss.item(),
                    "train/lr":        current_lr,
                    "global_step":     step,
                })

    avg_loss = total_loss / max(total_tokens, 1)

    # W&B epoch-level logging
    if wandb.run is not None:
        prefix = "train" if is_train else "val"
        wandb.log({
            f"{prefix}/epoch_loss": avg_loss,
            f"{prefix}/perplexity": torch.exp(torch.tensor(avg_loss)).item(),
            "epoch": epoch_num,
        })

    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.
    """
    model.eval()
    src      = src.to(device)
    src_mask = src_mask.to(device)

    with torch.no_grad():
        memory = model.encode(src, src_mask)

    ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)  # [1, 1]

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, model.pad_idx).to(device)

        with torch.no_grad():
            logits = model.decode(memory, src_mask, ys, tgt_mask)
            # logits: [1, cur_len, vocab_size]

        # Take the last token's prediction
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
        ys = torch.cat([ys, next_token], dim=1)

        if next_token.item() == end_symbol:
            break

    return ys


# ══════════════════════════════════════════════════════════════════════
#  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def _vocab_lookup_token(tgt_vocab, idx: int) -> str:
    """
    Look up a token string by index from a vocab object.
    Supports both torchtext (.get_itos()) and autograder-contract APIs
    (.itos attribute or .lookup_token() method).
    """
    if hasattr(tgt_vocab, "lookup_token"):
        return tgt_vocab.lookup_token(idx)
    elif hasattr(tgt_vocab, "itos"):
        return tgt_vocab.itos[idx]
    else:
        # torchtext build_vocab_from_iterator Vocab
        return tgt_vocab.get_itos()[idx]


def _vocab_get_stoi(tgt_vocab) -> dict:
    """
    Get the string-to-index dict from a vocab object.
    Supports both torchtext (.get_stoi()) and autograder-contract APIs
    (.stoi attribute).
    """
    if hasattr(tgt_vocab, "get_stoi"):
        return tgt_vocab.get_stoi()
    elif hasattr(tgt_vocab, "stoi"):
        return tgt_vocab.stoi
    else:
        raise AttributeError("Vocab object has neither get_stoi() nor .stoi")


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).
    """
    from evaluate import load as load_metric
    
    bleu_metric = load_metric("sacrebleu")
    model.eval()

    pad_idx  = model.pad_idx
    stoi     = _vocab_get_stoi(tgt_vocab)
    sos_idx  = stoi["<sos>"]
    eos_idx  = stoi["<eos>"]

    # Build a set of special indices to skip during detokenisation
    special_idx = {sos_idx, eos_idx, pad_idx}

    all_predictions = []   # list of strings  (one per sentence)
    all_references  = []   # list of [string] (one ref per sentence)

    with torch.no_grad():
        for src, tgt in test_dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            for i in range(src.size(0)):
                src_i    = src[i].unsqueeze(0)          # [1, src_len]
                src_mask = make_src_mask(src_i, pad_idx)

                pred = greedy_decode(
                    model, src_i, src_mask,
                    max_len=max_len,
                    start_symbol=sos_idx,
                    end_symbol=eos_idx,
                    device=device,
                )

                # Convert prediction token indices → words → sentence string
                pred_tokens = pred.squeeze(0).tolist()
                pred_words  = [
                    _vocab_lookup_token(tgt_vocab, t)
                    for t in pred_tokens
                    if t not in special_idx
                ]
                pred_sentence = " ".join(pred_words)

                # Convert reference token indices → words → sentence string
                ref_tokens = tgt[i].tolist()
                ref_words  = [
                    _vocab_lookup_token(tgt_vocab, t)
                    for t in ref_tokens
                    if t not in special_idx
                ]
                ref_sentence = " ".join(ref_words)

                # sacrebleu expects: predictions=[str], references=[[str]]
                all_predictions.append(pred_sentence)
                all_references.append([ref_sentence])

    result = bleu_metric.compute(
        predictions=all_predictions,
        references=all_references,
    )
    # sacrebleu returns score in 0–100 directly
    bleu_score = float(result["score"])
    return bleu_score


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'
    """
    model_config = {
        "src_vocab_size": model.src_vocab_size,
        "tgt_vocab_size": model.tgt_vocab_size,
        "d_model":        model.d_model,
        "N":              model.N,
        "num_heads":      model.num_heads,
        "d_ff":           model.d_ff,
        "dropout":        model.dropout_p,
        "pad_idx":        model.pad_idx,
    }
    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config":         model_config,
        },
        path,
    )
    print(f"Checkpoint saved → {path}  (epoch {epoch})")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    print(f"Loaded checkpoint from {path}  (epoch {epoch})")
    return epoch


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Full training experiment with W&B logging.
    """
    import torch.optim as optim
    from dataset import Multi30kDataset, PAD_IDX
    from lr_scheduler import NoamScheduler

    # ── Config ───────────────────────────────────────────────────────
    CONFIG = {
        "d_model":       256,
        "N":             3,
        "num_heads":     8,
        "d_ff":          512,
        "dropout":       0.1,
        "batch_size":    128,
        "num_epochs":    15,
        "warmup_steps":  4000,
        "label_smoothing": 0.1,
        "max_len":       100,
    }
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # ── W&B Init ─────────────────────────────────────────────────────
    wandb.init(
        project="da6401-a3",
        config=CONFIG,
        name="baseline-noam-ls01",   # intuitive: scheduler + label-smoothing value
    )
    cfg = wandb.config

    # ── Dataset ──────────────────────────────────────────────────────
    print("Loading dataset…")
    ds = Multi30kDataset()
    ds.build_vocab()
    ds.process_data()

    train_loader = ds.get_dataloader("train",      batch_size=cfg.batch_size, shuffle=True)
    val_loader   = ds.get_dataloader("val",        batch_size=cfg.batch_size, shuffle=False)
    test_loader  = ds.get_dataloader("test",       batch_size=1,              shuffle=False)

    src_vocab_size = len(ds.src_vocab)
    tgt_vocab_size = len(ds.tgt_vocab)
    print(f"Vocab sizes: src={src_vocab_size}, tgt={tgt_vocab_size}")

    # ── Model ─────────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=cfg.d_model,
        N=cfg.N,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        pad_idx=PAD_IDX,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    wandb.config.update({"n_params": n_params})

    # ── Optimizer + Scheduler ─────────────────────────────────────────
    optimizer = optim.Adam(
        model.parameters(),
        lr=1.0,          # base_lr=1.0; Noam formula handles actual scale
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps)

    # ── Loss ──────────────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(
        vocab_size=tgt_vocab_size,
        pad_idx=PAD_IDX,
        smoothing=cfg.label_smoothing,
    )

    # ── Training Loop ─────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_ckpt_path = "best_model.pt"

    for epoch in range(cfg.num_epochs):
        print(f"\nEpoch {epoch+1}/{cfg.num_epochs}")

        train_loss = run_epoch(
            train_loader, model, loss_fn,
            optimizer, scheduler,
            epoch_num=epoch,
            is_train=True,
            device=DEVICE,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn,
            None, None,
            epoch_num=epoch,
            is_train=False,
            device=DEVICE,
        )

        print(f"  Train loss: {train_loss:.4f}  |  Val loss: {val_loss:.4f}")

        # Save every epoch checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, path=f"checkpoint_epoch{epoch+1}.pt")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, path=best_ckpt_path)
            print(f"  ✓ New best val loss: {best_val_loss:.4f}")
            if wandb.run is not None:
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_epoch"]    = epoch

    # ── Final BLEU ────────────────────────────────────────────────────
    print("\nComputing test BLEU…")
    # Load best checkpoint for evaluation
    load_checkpoint(best_ckpt_path, model)
    model.eval()

    bleu = evaluate_bleu(model, test_loader, ds.tgt_vocab, device=DEVICE)
    print(f"Test BLEU: {bleu:.2f}")

    if wandb.run is not None:
        wandb.log({"test_bleu": bleu})
        wandb.run.summary["test_bleu"] = bleu

    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()