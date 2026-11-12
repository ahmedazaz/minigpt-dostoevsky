import json
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, Subset

import config as C
from data.dataset import CharDataset
from model import MiniGPT

# ---------------- Device: Apple GPU only ----------------
if not torch.backends.mps.is_available():
    raise RuntimeError("Apple GPU (mps) is not available on this machine.")
device = "mps"
print("Device:", device)

torch.manual_seed(C.SEED)

# ---------------- Data ----------------
with open(C.DATA_DIR / "meta.json", encoding="utf-8") as f:
    chars = json.load(f)["chars"]
vocab_size = len(chars)

train_dataset = CharDataset(C.DATA_DIR / "train.bin", C.CONTEXT_LENGTH)
val_dataset = CharDataset(C.DATA_DIR / "val.bin", C.CONTEXT_LENGTH)
test_dataset = CharDataset(C.DATA_DIR / "test.bin", C.CONTEXT_LENGTH)
train_tokens = len(train_dataset.data)

# ---------------- Model ----------------
model_config = dict(
    vocab_size=vocab_size,
    embedding_dim=C.EMBEDDING_DIM,
    context_length=C.CONTEXT_LENGTH,
    num_layers=C.NUM_LAYERS,
    num_heads=C.NUM_HEADS,
    dropout=C.DROPOUT,
)
model = MiniGPT(**model_config).to(device)
num_params = sum(p.numel() for p in model.parameters())

# ---------------- Chinchilla training budget ----------------
tokens_per_step = C.BATCH_SIZE * C.CONTEXT_LENGTH
chinchilla_tokens = C.TOKENS_PER_PARAM * num_params
max_data_tokens = C.MAX_EPOCHS * train_tokens
budget_tokens = min(chinchilla_tokens, max_data_tokens)
max_steps = math.ceil(budget_tokens / tokens_per_step)

# Optional override, e.g. MAX_STEPS=5000 python3 train.py
if os.environ.get("MAX_STEPS"):
    max_steps = int(os.environ["MAX_STEPS"])
    budget_tokens = max_steps * tokens_per_step

print(f"Parameters:        {num_params:,}")
print(f"Train characters:  {train_tokens:,}")
print(f"Chinchilla tokens: {chinchilla_tokens:,} (20 x params)")
print(f"Data limit:        {max_data_tokens:,} ({C.MAX_EPOCHS} epochs)")
print(f"Training on:       {budget_tokens:,} tokens -> {max_steps} steps "
      f"({budget_tokens / train_tokens:.1f} epochs)")

# ---------------- Loaders ----------------
train_loader = DataLoader(
    train_dataset,
    batch_size=C.BATCH_SIZE,
    sampler=RandomSampler(train_dataset, replacement=True, num_samples=max_steps * C.BATCH_SIZE),
)


def eval_loader(dataset):
    return DataLoader(dataset, batch_size=C.BATCH_SIZE, shuffle=True)


# ---------------- Optimizer: weight decay only on 2D weights ----------------
decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [
        {"params": decay, "weight_decay": C.WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=C.LEARNING_RATE,
    betas=(0.9, 0.95),
)
loss_fn = nn.CrossEntropyLoss()


def get_lr(step):
    """Linear warmup, then cosine decay to MIN_LEARNING_RATE."""
    if step < C.WARMUP_STEPS:
        return C.LEARNING_RATE * (step + 1) / C.WARMUP_STEPS
    progress = (step - C.WARMUP_STEPS) / max(1, max_steps - C.WARMUP_STEPS)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return C.MIN_LEARNING_RATE + cosine * (C.LEARNING_RATE - C.MIN_LEARNING_RATE)


def compute_loss(x, y):
    logits = model(x)
    return loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


@torch.no_grad()
def evaluate(dataset, num_batches=C.EVAL_BATCHES):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(eval_loader(dataset)):
        if i >= num_batches:
            break
        losses.append(compute_loss(x.to(device), y.to(device)).item())
    model.train()
    return sum(losses) / len(losses)


def bpc(loss):
    """Bits per character = cross-entropy (nats) / ln 2."""
    return loss / math.log(2)


# ---------------- Training loop ----------------
C.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
checkpoint_path = C.CHECKPOINT_DIR / "best_model.pt"
best_val_loss = float("inf")
start = time.time()
model.train()

for step, (x, y) in enumerate(train_loader):
    lr = get_lr(step)
    for group in optimizer.param_groups:
        group["lr"] = lr

    x, y = x.to(device), y.to(device)
    loss = compute_loss(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
    optimizer.step()

    is_last = step == max_steps - 1
    if (step + 1) % C.EVAL_INTERVAL == 0 or is_last:
        train_loss = evaluate(train_dataset)
        val_loss = evaluate(val_dataset)
        elapsed = time.time() - start
        print(
            f"Step {step + 1}/{max_steps} | lr {lr:.2e} | "
            f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
            f"val bpc {bpc(val_loss):.3f} | {elapsed:.0f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": model_config,
                    "chars": chars,
                    "step": step + 1,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            print(f"    saved best model -> {checkpoint_path}")

# ---------------- Final test on the best checkpoint ----------------
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model"])
# Full test set, split into non-overlapping windows
test_windows = Subset(test_dataset, range(0, len(test_dataset), C.CONTEXT_LENGTH))
test_loss = evaluate(test_windows, num_batches=10**9)

print("-" * 60)
print(f"Best val loss: {best_val_loss:.4f} (bpc {bpc(best_val_loss):.3f}) at step {checkpoint['step']}")
print(f"Test loss:     {test_loss:.4f} (bpc {bpc(test_loss):.3f})")
