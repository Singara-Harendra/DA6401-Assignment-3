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
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

import wandb

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import PAD_IDX, SOS_IDX, EOS_IDX


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    The <pad> token always gets 0 probability.
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
            Scalar loss value.
        """
        log_probs = torch.log_softmax(logits, dim=-1)   # [N, V]

        # Build smooth target distribution
        with torch.no_grad():
            smooth_val = self.smoothing / (self.vocab_size - 2)  # -2: pad + gold
            targets_smooth = torch.full_like(log_probs, smooth_val)
            targets_smooth.scatter_(1, target.unsqueeze(1), self.confidence)
            targets_smooth[:, self.pad_idx] = 0.0           # no probability mass on <pad>

        # Zero out rows where target == pad (don't penalise padding)
        pad_mask = (target == self.pad_idx).unsqueeze(1)
        targets_smooth = targets_smooth.masked_fill(pad_mask, 0.0)

        loss = -(targets_smooth * log_probs).sum()
        # Normalise by number of non-pad tokens
        n_tokens = (target != self.pad_idx).sum().float()
        return loss / (n_tokens + 1e-9)


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
    Returns avg_loss (float).
    """
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_tokens = 0
    step_count = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for batch_idx, (src, tgt) in enumerate(data_iter):
            src = src.to(device)   # [batch, src_len]
            tgt = tgt.to(device)   # [batch, tgt_len]

            # Decoder input: all tokens except last
            tgt_input  = tgt[:, :-1]
            # Decoder target: all tokens except first (SOS)
            tgt_output = tgt[:, 1:]

            src_mask = make_src_mask(src, pad_idx=PAD_IDX)
            tgt_mask = make_tgt_mask(tgt_input, pad_idx=PAD_IDX)

            logits = model(src, tgt_input, src_mask, tgt_mask)
            # logits: [batch, tgt_len-1, vocab_size]

            batch_size, seq_len, vocab_size = logits.shape
            logits_flat  = logits.contiguous().view(-1, vocab_size)
            targets_flat = tgt_output.contiguous().view(-1)

            loss = loss_fn(logits_flat, targets_flat)

            n_tokens = (targets_flat != PAD_IDX).sum().item()
            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            step_count += 1
            if batch_idx % 50 == 0:
                mode = "Train" if is_train else "Val"
                print(f"  [{mode}] Epoch {epoch_num} | Step {batch_idx} | "
                      f"Loss {loss.item():.4f}")

    avg_loss = total_loss / (total_tokens + 1e-9)
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
    Returns ys: shape [1, out_len]
    """
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=PAD_IDX)
            logits = model.decode(memory, src_mask, ys, tgt_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            if next_token.item() == end_symbol:
                break

    return ys


# ══════════════════════════════════════════════════════════════════════
#  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.
    Returns bleu_score (float, range 0–100).
    """
    from evaluate import load as hf_load
    bleu_metric = hf_load("bleu")

    model.eval()
    predictions = []
    references  = []
    skip = {SOS_IDX, EOS_IDX, PAD_IDX}

    with torch.no_grad():
        for src, tgt in test_dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            src_mask = make_src_mask(src, pad_idx=PAD_IDX)

            for i in range(src.size(0)):
                src_i    = src[i].unsqueeze(0)
                mask_i   = src_mask[i].unsqueeze(0)
                tgt_i    = tgt[i].tolist()

                out = greedy_decode(
                    model, src_i, mask_i,
                    max_len=max_len,
                    start_symbol=SOS_IDX,
                    end_symbol=EOS_IDX,
                    device=device,
                )

                pred_tokens = [
                    tgt_vocab.itos[idx]
                    for idx in out[0].tolist()
                    if idx not in skip
                ]
                ref_tokens = [
                    tgt_vocab.itos[idx]
                    for idx in tgt_i
                    if idx not in skip
                ]

                predictions.append(" ".join(pred_tokens))
                references.append([" ".join(ref_tokens)])

    result = bleu_metric.compute(
        predictions=predictions,
        references=references,
    )
    # HuggingFace evaluate returns 0–1; multiply by 100
    return result["bleu"] * 100.0


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
    """Save model + optimiser + scheduler state to disk."""
    model_config = {
        "src_vocab_size": model.src_vocab_size,
        "tgt_vocab_size": model.tgt_vocab_size,
        "d_model":        model.d_model,
        "N":              len(model.encoder.layers),
        "num_heads":      model.encoder.layers[0].self_attn.num_heads,
        "d_ff":           model.encoder.layers[0].ffn.linear1.out_features,
        "dropout":        model.encoder.layers[0].dropout.p,
        "pad_idx":        model.pad_idx,
    }
    torch.save({
        "epoch":               epoch,
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config":        model_config,
    }, path)
    print(f"[Checkpoint] Saved epoch {epoch} → {path}")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """Restore model (and optionally optimizer/scheduler) state from disk."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    epoch = ckpt.get("epoch", 0)
    print(f"[Checkpoint] Loaded epoch {epoch} from {path}")
    return epoch


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment(config_overrides: dict = None) -> None:
    """
    Set up and run the full training experiment.
    Pass config_overrides dict to run ablation experiments.
    """
    # ── Default config ───────────────────────────────────────────────
    config = {
        "d_model":       256,
        "N":             3,
        "num_heads":     8,
        "d_ff":          512,
        "dropout":       0.25,     
        "batch_size":    64,       
        "num_epochs":    21,       
        "warmup_steps":  1500,
        "label_smoothing": 0.1,
        "use_noam":      True,
        "use_scaling":   True,   
        "max_len":       100,
    }
    if config_overrides:
        config.update(config_overrides)

    # ── W&B init ─────────────────────────────────────────────────────
    run = wandb.init(
        project="da6401-a3",
        config=config,
    )
    cfg = wandb.config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] device={device}")

    # ── Dataset & Dataloaders ────────────────────────────────────────
    from dataset import build_dataloaders
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = \
        build_dataloaders(batch_size=cfg.batch_size, device=device)

    # ── Model ────────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=cfg.d_model,
        N=cfg.N,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        pad_idx=PAD_IDX,
    ).to(device)

    wandb.watch(model, log="gradients", log_freq=100)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Parameters: {n_params:,}")

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )

    # ── Scheduler ─────────────────────────────────────────────────────
    from lr_scheduler import NoamScheduler
    if cfg.use_noam:
        scheduler = NoamScheduler(optimizer, d_model=cfg.d_model,
                                  warmup_steps=cfg.warmup_steps)
    else:
        # Fixed LR experiment
        for pg in optimizer.param_groups:
            pg["lr"] = 1e-4
        scheduler = None

    # ── Loss ──────────────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=PAD_IDX,
        smoothing=cfg.label_smoothing,
    )

    # ── Training loop ─────────────────────────────────────────────────
    best_val_bleu = -1.0    # <--- NOW TRACKING BEST BLEU SCORE
    for epoch in range(cfg.num_epochs):
        print(f"\n[Epoch {epoch+1}/{cfg.num_epochs}]")

        train_loss = run_epoch(
            train_loader, model, loss_fn,
            optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn,
            None, None,
            epoch_num=epoch, is_train=False, device=device,
        )

        # <--- EVALUATING BLEU EVERY EPOCH TO FIND THE TRUE MAXIMUM
        val_bleu = evaluate_bleu(model, val_loader, tgt_vocab,
                                 device=device, max_len=cfg.max_len)

        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_bleu={val_bleu:.2f}")

        wandb.log({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "val_bleu":   val_bleu,
            "lr":         optimizer.param_groups[0]["lr"],
        })

        # <--- SAVE CHECKPOINT BASED ON HIGHEST BLEU SCORE
        if val_bleu > best_val_bleu:
            best_val_bleu = val_bleu
            save_checkpoint(model, optimizer, scheduler or torch.optim.Adam([]),
                            epoch, path="best_model.pt")

        # Also save latest
        save_checkpoint(model, optimizer, scheduler or torch.optim.Adam([]),
                        epoch, path="latest_model.pt")

    # ── Final BLEU on test set ────────────────────────────────────────
    print("\n[Train] Evaluating on test set …")
    # Reload best checkpoint
    load_checkpoint("best_model.pt", model)
    model.to(device)
    test_bleu = evaluate_bleu(model, test_loader, tgt_vocab,
                              device=device, max_len=cfg.max_len)
    print(f"[Train] Test BLEU: {test_bleu:.2f}")
    wandb.log({"test_bleu": test_bleu})

    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()