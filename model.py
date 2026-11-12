import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. Multi-Head Causal Self-Attention
class CausalSelfAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads, context_length, dropout):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Q, K, V for all heads in one projection
        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.proj = nn.Linear(embedding_dim, embedding_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, context_length, context_length), persistent=False)

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(C, dim=2)

        # (B, T, C) -> (B, heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))

        out = weights @ v  # (B, heads, T, head_dim)

        # Concatenate heads back: (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.proj(out))


# 2. Feed Forward
class FeedForward(nn.Module):

    def __init__(self, embedding_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# 3. Transformer Block (pre-norm)
class TransformerBlock(nn.Module):

    def __init__(self, embedding_dim, num_heads, context_length, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = CausalSelfAttention(embedding_dim, num_heads, context_length, dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ff = FeedForward(embedding_dim, dropout)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# 4. Mini GPT
class MiniGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        context_length=256,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(embedding_dim, num_heads, context_length, dropout)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size, bias=False)

        # Weight tying: input embedding and output head share the same matrix
        self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, x):
        T = x.shape[1]
        positions = torch.arange(T, device=x.device)

        x = self.dropout(self.token_embedding(x) + self.position_embedding(positions))
        x = self.blocks(x)
        x = self.norm(x)

        return self.head(x)  # logits: (B, T, vocab_size)

    @torch.no_grad()
    def generate(self, ids, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively sample max_new_tokens characters after ids (B, T)."""
        for _ in range(max_new_tokens):
            context = ids[:, -self.context_length:]
            logits = self(context)[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)

        return ids


if __name__ == "__main__":
    model = MiniGPT(vocab_size=99)
    x = torch.randint(0, 99, (8, 256))
    print("logits:", model(x).shape)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
