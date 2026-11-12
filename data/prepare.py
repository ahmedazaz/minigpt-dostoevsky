"""Clean the raw novels, split each one 90/7/3, and encode them at character level.

Outputs (in data/):
    train.bin, val.bin, test.bin  -> uint16 arrays of character ids
    meta.json                     -> the vocabulary (list of characters)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR, DATA_RAW, TRAIN_SPLIT, VAL_SPLIT

START_MARKER = re.compile(r"\*\*\* ?START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*")
END_MARKER = re.compile(r"\*\*\* ?END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*")


def strip_gutenberg(text: str, name: str) -> str:
    """Keep only the novel itself: drop the Gutenberg header and license footer."""
    start = START_MARKER.search(text)
    end = END_MARKER.search(text)
    if not start or not end:
        raise ValueError(f"Gutenberg markers not found in {name}")
    return text[start.end():end.start()].strip()


def split_text(text: str):
    n = len(text)
    train_end = int(n * TRAIN_SPLIT)
    val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))
    return text[:train_end], text[train_end:val_end], text[val_end:]


def prepare():
    splits = {"train": [], "val": [], "test": []}

    for path in sorted(DATA_RAW.glob("*.txt")):
        raw = path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")
        novel = strip_gutenberg(raw, path.name)

        train, val, test = split_text(novel)
        splits["train"].append(train)
        splits["val"].append(val)
        splits["test"].append(test)

        print(f"{path.name}: {len(raw):,} -> {len(novel):,} chars after cleaning")

    texts = {name: "\n\n".join(parts) for name, parts in splits.items()}

    chars = sorted(set("".join(texts.values())))
    stoi = {ch: i for i, ch in enumerate(chars)}

    for name, text in texts.items():
        ids = np.array([stoi[ch] for ch in text], dtype=np.uint16)
        ids.tofile(DATA_DIR / f"{name}.bin")
        print(f"{name}: {len(ids):,} characters")

    with open(DATA_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"chars": chars}, f, ensure_ascii=False)

    print(f"vocab size: {len(chars)}")


if __name__ == "__main__":
    prepare()
