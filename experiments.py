"""
experiments.py — W&B Ablation Experiments for DA6401 Assignment 3

Run all required experiments for the W&B report:
  1. Noam scheduler vs Fixed LR
  2. Scaling factor 1/√dk ablation
  3. Attention head visualization (head specialization)
  4. Sinusoidal PE vs Learned Embeddings
  5. Label smoothing ε=0.1 vs ε=0.0

Usage:
    python experiments.py --exp all
    python experiments.py --exp noam
    python experiments.py --exp scaling
    python experiments.py --exp heads
    python experiments.py --exp pe
    python experiments.py --exp smoothing
"""

import argparse
import math
import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
import numpy as np

from model import (
    Transformer, scaled_dot_product_attention,
    make_src_mask, make_tgt_mask
)
from dataset import PAD_IDX, SOS_IDX, EOS_IDX, build_dataloaders
from train import (
    LabelSmoothingLoss, run_epoch, evaluate_bleu,
    save_checkpoint, load_checkpoint
)
from lr_scheduler import NoamScheduler


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT 1: Noam scheduler vs Fixed LR
# ══════════════════════════════════════════════════════════════════════

def exp_noam_vs_fixed():
    """Train two models: one with Noam LR, one with fixed LR=1e-4."""
    print("\n=== Experiment 1: Noam vs Fixed LR ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, _, src_vocab, tgt_vocab = \
        build_dataloaders(batch_size=128)

    for use_noam in [True, False]:
        run_name = "noam_scheduler" if use_noam else "fixed_lr_1e-4"
        print(f"\n  Running: {run_name}")

        run = wandb.init(
            project="da6401-a3",
            name=run_name,
            group="exp1_lr_comparison",
            config={
                "d_model": 256, "N": 3, "num_heads": 8,
                "d_ff": 512, "dropout": 0.1, "num_epochs": 15,
                "use_noam": use_noam,
            },
            reinit=True,
        )

        model = Transformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=256, N=3, num_heads=8, d_ff=512,
            pad_idx=PAD_IDX,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
        )

        if use_noam:
            scheduler = NoamScheduler(optimizer, d_model=256, warmup_steps=4000)
        else:
            for pg in optimizer.param_groups:
                pg["lr"] = 1e-4
            scheduler = None

        loss_fn = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, smoothing=0.1)

        for epoch in range(15):
            train_loss = run_epoch(
                train_loader, model, loss_fn,
                optimizer, scheduler, epoch, is_train=True, device=device
            )
            val_loss = run_epoch(
                val_loader, model, loss_fn,
                None, None, epoch, is_train=False, device=device
            )
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            })

        wandb.finish()


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT 2: Scaling factor ablation
# ══════════════════════════════════════════════════════════════════════

def exp_scaling_ablation():
    """
    Compare training with and without 1/√dk scaling.
    Log gradient norms of Q and K weights for first 1000 steps.
    """
    print("\n=== Experiment 2: Scaling Factor Ablation ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, _, src_vocab, tgt_vocab = \
        build_dataloaders(batch_size=128)

    for use_scaling in [True, False]:
        run_name = "with_scaling" if use_scaling else "no_scaling"
        print(f"\n  Running: {run_name}")

        run = wandb.init(
            project="da6401-a3",
            name=run_name,
            group="exp2_scaling_ablation",
            config={"use_scaling": use_scaling},
            reinit=True,
        )

        model = Transformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=256, N=3, num_heads=8, d_ff=512,
            pad_idx=PAD_IDX,
        ).to(device)

        # Monkey-patch attention if ablating scaling
        if not use_scaling:
            import model as model_module
            _orig_sdpa = model_module.scaled_dot_product_attention

            def sdpa_no_scale(Q, K, V, mask=None):
                # Same as original but without /√dk
                scores = torch.matmul(Q, K.transpose(-2, -1))  # no scaling
                if mask is not None:
                    scores = scores.masked_fill(mask, float("-inf"))
                attn_w = torch.softmax(scores, dim=-1)
                attn_w = torch.nan_to_num(attn_w, nan=0.0)
                return torch.matmul(attn_w, V), attn_w

            model_module.scaled_dot_product_attention = sdpa_no_scale

        optimizer = torch.optim.Adam(
            model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
        )
        scheduler = NoamScheduler(optimizer, d_model=256, warmup_steps=4000)
        loss_fn = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, smoothing=0.1)

        model.train()
        global_step = 0

        for src, tgt in train_loader:
            if global_step >= 1000:
                break

            src, tgt = src.to(device), tgt.to(device)
            tgt_in  = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask = make_src_mask(src)
            tgt_mask = make_tgt_mask(tgt_in)

            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss = loss_fn(logits.view(-1, len(tgt_vocab)), tgt_out.contiguous().view(-1))

            optimizer.zero_grad()
            loss.backward()

            # Log gradient norms of Q and K projections in first encoder layer
            enc0_attn = model.encoder.layers[0].self_attn
            q_grad = enc0_attn.W_q.weight.grad
            k_grad = enc0_attn.W_k.weight.grad
            q_norm = q_grad.norm().item() if q_grad is not None else 0.0
            k_norm = k_grad.norm().item() if k_grad is not None else 0.0

            wandb.log({
                "step": global_step,
                "train_loss": loss.item(),
                "grad_norm_Q": q_norm,
                "grad_norm_K": k_norm,
            })

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

        # Restore original if patched
        if not use_scaling:
            model_module.scaled_dot_product_attention = _orig_sdpa

        wandb.finish()


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT 3: Attention Head Visualization
# ══════════════════════════════════════════════════════════════════════

def exp_attention_heads(checkpoint_path: str = "best_model.pt"):
    """
    Extract and visualize attention weights from the last encoder layer.
    Log heatmaps for each head to W&B.
    """
    print("\n=== Experiment 3: Attention Head Visualization ===")

    import spacy
    from dataset import Multi30kDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load a trained model
    train_ds = Multi30kDataset("train")
    src_vocab = train_ds.src_vocab
    tgt_vocab = train_ds.tgt_vocab

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=256, N=3, num_heads=8, d_ff=512,
        pad_idx=PAD_IDX,
    ).to(device)

    if checkpoint_path and __import__("os").path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, model)

    model.eval()

    # Pick a sample sentence
    sample_de = "Ein Mann mit einem roten Hut spielt Gitarre ."
    spacy_de = spacy.load("de_core_news_sm")
    tokens = ["<sos>"] + [t.text.lower() for t in spacy_de.tokenizer(sample_de)] + ["<eos>"]
    indices = [src_vocab.stoi.get(t, 0) for t in tokens]
    src = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    src_mask = make_src_mask(src)

    run = wandb.init(
        project="da6401-a3",
        name="attention_head_viz",
        group="exp3_attention_heads",
        reinit=True,
    )

    with torch.no_grad():
        _ = model.encode(src, src_mask)

    # Get attention weights from last encoder layer
    last_layer = model.encoder.layers[-1]
    attn_weights = last_layer.self_attn.attn_weights  # [1, heads, seq, seq]

    if attn_weights is None:
        print("  Warning: no attention weights captured.")
        wandb.finish()
        return

    attn_weights = attn_weights[0].cpu().numpy()  # [heads, seq, seq]
    num_heads = attn_weights.shape[0]

    figs = []
    for h in range(num_heads):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(attn_weights[h], cmap="viridis", aspect="auto")
        ax.set_title(f"Head {h+1} — Last Encoder Layer")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        figs.append(wandb.Image(fig, caption=f"Head {h+1}"))
        plt.close(fig)

    wandb.log({"attention_heads": figs})
    wandb.finish()
    print(f"  Logged {num_heads} head heatmaps.")


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT 4: Sinusoidal PE vs Learned Positional Embeddings
# ══════════════════════════════════════════════════════════════════════

class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings as an alternative to sinusoidal PE."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.pos_embed(positions))


def exp_pe_vs_learned():
    """Compare sinusoidal PE vs learned positional embeddings."""
    print("\n=== Experiment 4: PE vs Learned Embeddings ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = \
        build_dataloaders(batch_size=128)

    for pe_type in ["sinusoidal", "learned"]:
        print(f"\n  Running: {pe_type}")

        run = wandb.init(
            project="da6401-a3",
            name=f"pe_{pe_type}",
            group="exp4_pe_comparison",
            config={"pe_type": pe_type},
            reinit=True,
        )

        model = Transformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=256, N=3, num_heads=8, d_ff=512,
            pad_idx=PAD_IDX,
        ).to(device)

        # Replace PE with learned version if needed
        if pe_type == "learned":
            model.pos_enc = LearnedPositionalEncoding(
                d_model=256, dropout=0.1
            ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
        )
        scheduler = NoamScheduler(optimizer, d_model=256, warmup_steps=4000)
        loss_fn = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, smoothing=0.1)

        best_val_bleu = 0.0
        for epoch in range(15):
            run_epoch(train_loader, model, loss_fn,
                      optimizer, scheduler, epoch, True, device)
            val_loss = run_epoch(val_loader, model, loss_fn,
                                 None, None, epoch, False, device)

            val_bleu = 0.0
            if (epoch + 1) % 5 == 0:
                val_bleu = evaluate_bleu(model, val_loader, tgt_vocab, device)
                if val_bleu > best_val_bleu:
                    best_val_bleu = val_bleu

            wandb.log({
                "epoch": epoch,
                "val_loss": val_loss,
                "val_bleu": val_bleu,
            })

        wandb.log({"best_val_bleu": best_val_bleu})
        wandb.finish()


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT 5: Label Smoothing ablation
# ══════════════════════════════════════════════════════════════════════

def exp_label_smoothing():
    """Compare label smoothing ε=0.1 vs ε=0.0 (standard cross-entropy)."""
    print("\n=== Experiment 5: Label Smoothing Ablation ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, _, src_vocab, tgt_vocab = \
        build_dataloaders(batch_size=128)

    for smoothing in [0.1, 0.0]:
        run_name = f"label_smooth_{smoothing}"
        print(f"\n  Running: {run_name}")

        run = wandb.init(
            project="da6401-a3",
            name=run_name,
            group="exp5_label_smoothing",
            config={"label_smoothing": smoothing},
            reinit=True,
        )

        model = Transformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=256, N=3, num_heads=8, d_ff=512,
            pad_idx=PAD_IDX,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
        )
        scheduler = NoamScheduler(optimizer, d_model=256, warmup_steps=4000)
        loss_fn = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, smoothing=smoothing)

        for epoch in range(15):
            model.train()
            total_loss = 0.0
            total_confidence = 0.0
            n_batches = 0

            for src, tgt in train_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_in  = tgt[:, :-1]
                tgt_out = tgt[:, 1:]

                src_mask = make_src_mask(src)
                tgt_mask = make_tgt_mask(tgt_in)

                logits = model(src, tgt_in, src_mask, tgt_mask)
                loss = loss_fn(logits.view(-1, len(tgt_vocab)), tgt_out.contiguous().view(-1))

                # Track prediction confidence (softmax prob of correct token)
                with torch.no_grad():
                    probs = torch.softmax(logits.view(-1, len(tgt_vocab)), dim=-1)
                    target_flat = tgt_out.contiguous().view(-1)
                    mask = target_flat != PAD_IDX
                    if mask.sum() > 0:
                        correct_probs = probs[mask].gather(
                            1, target_flat[mask].unsqueeze(1)
                        ).squeeze(1)
                        total_confidence += correct_probs.mean().item()
                        n_batches += 1

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_confidence = total_confidence / max(n_batches, 1)

            val_loss = run_epoch(val_loader, model, loss_fn,
                                 None, None, epoch, False, device)
            wandb.log({
                "epoch":              epoch,
                "train_loss":         total_loss / max(n_batches, 1),
                "val_loss":           val_loss,
                "prediction_confidence": avg_confidence,
            })

        wandb.finish()


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="all",
                        choices=["all", "noam", "scaling", "heads", "pe", "smoothing"])
    args = parser.parse_args()

    if args.exp in ("all", "noam"):
        exp_noam_vs_fixed()
    if args.exp in ("all", "scaling"):
        exp_scaling_ablation()
    if args.exp in ("all", "heads"):
        exp_attention_heads()
    if args.exp in ("all", "pe"):
        exp_pe_vs_learned()
    if args.exp in ("all", "smoothing"):
        exp_label_smoothing()
