# DA6401 Assignment 3 — Transformer for German→English NMT

## Project structure

```
da6401_assignment_3/
├── model.py          # Full Transformer implementation
├── dataset.py        # Multi30k loader + vocabulary
├── lr_scheduler.py   # Noam LR scheduler
├── train.py          # Training loop, BLEU eval, checkpointing
├── experiments.py    # All 5 W&B ablation experiments
├── requirements.txt
└── README.md
```

---

## Step-by-step setup

### 1. Clone your repo / copy files
```bash
git clone https://github.com/YOUR_USERNAME/da6401_assignment_3
cd da6401_assignment_3
# Copy all the .py files here
```

### 2. Install dependencies
```bash
pip install -r requirements.txt

# Download spaCy models (German + English)
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

### 3. Log in to W&B
```bash
wandb login
# paste your API key when prompted
```

---

## Training

### Main training run (baseline)
```bash
python train.py
```
This runs `run_training_experiment()` with defaults:
- d_model=256, N=3, num_heads=8, d_ff=512
- 20 epochs, batch_size=128
- Noam scheduler (warmup=4000), label_smoothing=0.1
- Saves `best_model.pt` (best val loss) and `vocab.pkl`

Training time estimate:
- GPU (T4/V100): ~1–2 hours
- CPU: ~8–12 hours (reduce num_epochs to 10)

---

## Running ablation experiments (W&B report)

```bash
# All 5 experiments
python experiments.py --exp all

# Or individually:
python experiments.py --exp noam       # Noam vs Fixed LR
python experiments.py --exp scaling    # With/without 1/√dk
python experiments.py --exp heads      # Attention head heatmaps
python experiments.py --exp pe         # Sinusoidal vs Learned PE
python experiments.py --exp smoothing  # Label smoothing ablation
```

---

## After training: upload weights for autograder

### 1. Upload `best_model.pt` and `vocab.pkl` to Google Drive
   - Right-click each file → Share → Anyone with link → Viewer
   - Copy the file ID from the URL:
     `https://drive.google.com/file/d/FILE_ID_HERE/view`

### 2. Paste IDs into `model.py`
```python
# In class Transformer:
WEIGHT_DRIVE_ID = "1AbCdEfGhIjKlMnOpQrStUvWxYz123456"   # your .pt ID
VOCAB_DRIVE_ID  = "1AbCdEfGhIjKlMnOpQrStUvWxYz789012"   # your vocab.pkl ID
```

### 3. Test the autograder flow locally
```python
from model import Transformer

model = Transformer()   # downloads weights + vocab automatically
model.eval()

translation = model.infer("Ein Mann spielt Gitarre .")
print(translation)
```

---

## Hyperparameter choices explained

| Param | Value | Reason |
|-------|-------|--------|
| d_model | 256 | Smaller than paper's 512 — Multi30k is tiny (29k pairs) |
| N | 3 | 3 layers instead of 6 — avoids overfit on small dataset |
| num_heads | 8 | Keeps d_k=32, reasonable per-head capacity |
| d_ff | 512 | 2×d_model (paper uses 4×, scaled down for dataset size) |
| warmup_steps | 4000 | As in original paper |
| batch_size | 128 | Fits comfortably on GPU |
| label_smoothing | 0.1 | Exactly as specified |

---

## LayerNorm choice: Post-LayerNorm

We implement **Post-LayerNorm** (as in the original paper):
```
x = LayerNorm(x + sublayer(x))
```

**Justification**: The original "Attention Is All You Need" paper uses Post-LN.
Pre-LN (applied before the sublayer) generally trains more stably for very deep
networks, but for N=3 or N=6, Post-LN matches the paper specification exactly
and is what the autograder tests expect.

---

## Common issues

**`OSError: [E050] Can't find model 'de_core_news_sm'`**
```bash
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

**CUDA out of memory**
Reduce `batch_size` to 64 or lower in `run_training_experiment()`.

**gdown fails silently**
Make sure your Drive sharing is set to "Anyone with the link" (not just specific people).
