"""Generate Dostoevsky-style text from the trained character-level MiniGPT.

Example:
    python3 generate.py --prompt "Raskolnikov looked at" --length 500
"""
import argparse

import torch

import config as C
from model import MiniGPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="\n", help="text to start from")
    parser.add_argument("--length", type=int, default=500, help="characters to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="lower = safer, higher = more creative")
    parser.add_argument("--top_k", type=int, default=None, help="sample only from the k most likely characters")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", default=str(C.CHECKPOINT_DIR / "best_model.pt"))
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple GPU (mps) is not available on this machine.")
    device = "mps"

    if args.seed is not None:
        torch.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    chars = checkpoint["chars"]
    stoi = {ch: i for i, ch in enumerate(chars)}

    model = MiniGPT(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    unknown = sorted({ch for ch in args.prompt if ch not in stoi})
    if unknown:
        print(f"[warning] skipping characters not in vocabulary: {unknown}")
    prompt_ids = [stoi[ch] for ch in args.prompt if ch in stoi]
    if not prompt_ids:
        raise ValueError("Prompt has no known characters.")

    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(ids, args.length, temperature=args.temperature, top_k=args.top_k)

    print("".join(chars[i] for i in out[0].tolist()))


if __name__ == "__main__":
    main()
