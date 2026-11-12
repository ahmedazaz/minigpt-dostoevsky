"""Evaluate checkpoints on the test set and compare against simple baselines.

Metrics
    bpc          bits per character (cross-entropy / ln 2), lower is better
    top-1 acc    next character predicted exactly
    top-5 acc    true next character is among the 5 most likely
    word valid   % of words in generated text that appear in the training text

Usage:
    python3 evaluate.py
"""
import json
import math
import re
from collections import Counter

import numpy as np
import torch

import config as C
from model import MiniGPT

CHECKPOINTS = {
    "MiniGPT, 971 steps (Chinchilla)": C.CHECKPOINT_DIR / "chinchilla_971steps.pt",
    "MiniGPT, 5000 steps": C.CHECKPOINT_DIR / "steps_5000.pt",
}
NUM_SAMPLES = 20
SAMPLE_LENGTH = 500
TEMPERATURE = 0.8
WORD = re.compile(r"[a-z]+")

if not torch.backends.mps.is_available():
    raise RuntimeError("Apple GPU (mps) is not available on this machine.")
device = "mps"

with open(C.DATA_DIR / "meta.json", encoding="utf-8") as f:
    chars = json.load(f)["chars"]
V = len(chars)

train = np.fromfile(C.DATA_DIR / "train.bin", dtype=np.uint16).astype(np.int64)
test = np.fromfile(C.DATA_DIR / "test.bin", dtype=np.uint16).astype(np.int64)

train_text = "".join(chars[i] for i in train)
train_words = set(WORD.findall(train_text.lower()))


def word_validity(text):
    words = WORD.findall(text.lower())
    return sum(w in train_words for w in words) / max(1, len(words))


results = []

# ---------------- Baseline 1: always predict the most common character ----------------
unigram = np.bincount(train, minlength=V) + 1
unigram_p = unigram / unigram.sum()
targets = test[1:]
results.append({
    "model": "Baseline: most common char",
    "bpc": float(-np.log2(unigram_p[targets]).mean()),
    "top1": float((targets == unigram_p.argmax()).mean()),
    "top5": float(np.isin(targets, np.argsort(-unigram_p)[:5]).mean()),
})

# ---------------- Baseline 2: bigram (looks at the previous character only) ----------------
bigram = np.ones((V, V))  # add-one smoothing
np.add.at(bigram, (train[:-1], train[1:]), 1)
bigram_p = bigram / bigram.sum(axis=1, keepdims=True)
prev = test[:-1]
top5_bigram = np.argsort(-bigram_p, axis=1)[:, :5]
results.append({
    "model": "Baseline: bigram",
    "bpc": float(-np.log2(bigram_p[prev, targets]).mean()),
    "top1": float((bigram_p[prev].argmax(axis=1) == targets).mean()),
    "top5": float((top5_bigram[prev] == targets[:, None]).any(axis=1).mean()),
})


# ---------------- MiniGPT checkpoints ----------------
@torch.no_grad()
def eval_checkpoint(path):
    ckpt = torch.load(path, map_location=device)
    model = MiniGPT(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    T = ckpt["config"]["context_length"]
    n_windows = (len(test) - 1) // T
    x = torch.from_numpy(test[: n_windows * T].reshape(n_windows, T)).to(device)
    y = torch.from_numpy(test[1 : n_windows * T + 1].reshape(n_windows, T)).to(device)

    nll, top1, top5, count = 0.0, 0, 0, 0
    for i in range(0, n_windows, C.BATCH_SIZE):
        logits = model(x[i : i + C.BATCH_SIZE])
        yy = y[i : i + C.BATCH_SIZE]
        nll += torch.nn.functional.cross_entropy(
            logits.reshape(-1, V), yy.reshape(-1), reduction="sum"
        ).item()
        top = logits.topk(5, dim=-1).indices
        top1 += (top[..., 0] == yy).sum().item()
        top5 += (top == yy.unsqueeze(-1)).any(-1).sum().item()
        count += yy.numel()

    # Generated text quality
    torch.manual_seed(0)
    start = torch.full((NUM_SAMPLES, 1), chars.index("\n"), device=device)
    out = model.generate(start, SAMPLE_LENGTH, temperature=TEMPERATURE)
    samples = ["".join(chars[i] for i in row) for row in out.tolist()]

    return {
        "bpc": nll / count / math.log(2),
        "top1": top1 / count,
        "top5": top5 / count,
        "word_valid": sum(word_validity(s) for s in samples) / len(samples),
    }


for name, path in CHECKPOINTS.items():
    results.append({"model": name, **eval_checkpoint(path)})

# Reference: real Dostoevsky text from the test set
real_test = "".join(chars[i] for i in test)
results.append({"model": "Reference: real test text", "word_valid": word_validity(real_test)})

# ---------------- Report ----------------
def fmt(r, key, pct):
    if key not in r:
        return "-"
    return f"{100 * r[key]:.1f}%" if pct else f"{r[key]:.3f}"


print(f"{'Model':<34} {'bpc':>7} {'top-1':>7} {'top-5':>7} {'word valid':>11}")
print("-" * 70)
for r in results:
    print(f"{r['model']:<34} {fmt(r, 'bpc', False):>7} {fmt(r, 'top1', True):>7} "
          f"{fmt(r, 'top5', True):>7} {fmt(r, 'word_valid', True):>11}")
