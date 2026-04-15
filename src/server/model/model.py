"""
model.py — Modular Transformer Architecture with Router

This file implements a modular GPT-style transformer architecture designed
for Scout's long‑term expansion into a multi‑module routed system.

Current training phase
----------------------

TinyStories language bootstrapping.

Architecture:

tokens
  │
shared embedding
  │
router (currently trivial)
  │
TinyStories transformer module
  │
shared output head
  │
logits

Future architecture
-------------------

tokens
  │
shared embedding
  │
router
  │
 ┌──────────────┬──────────────┐
 │ TinyStories  │ reasoning    │
 │ module       │ module       │
 └──────────────┴──────────────┘
  │
shared output head
  │
logits

Design goals
------------

• Shared token embedding space
• Modular transformer blocks
• Router capable of selecting modules
• Modules can be frozen independently
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Rotary Position Embeddings
# ──────────────────────────────────────────────────────────────────────────────

def precompute_rope_freqs(head_dim: int, max_seq: int, theta: float = 10000.0):

    assert head_dim % 2 == 0

    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq).float()

    angles = torch.outer(positions, freqs)
    angles = torch.cat([angles, angles], dim=-1)

    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):

    d = x.shape[-1]

    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]

    x_rot = torch.cat([-x2, x1], dim=-1)

    cos = cos[: x.shape[2]].unsqueeze(0).unsqueeze(0)
    sin = sin[: x.shape[2]].unsqueeze(0).unsqueeze(0)

    return x * cos + x_rot * sin


# ──────────────────────────────────────────────────────────────────────────────
# Causal Self Attention
# ──────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):

    def __init__(self, dim, heads, max_seq, dropout):

        super().__init__()

        assert dim % heads == 0

        self.heads = heads
        self.head_dim = dim // heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

        cos, sin = precompute_rope_freqs(self.head_dim, max_seq)

        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x):

        B, T, C = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out(out)


# ──────────────────────────────────────────────────────────────────────────────
# Transformer Block
# ──────────────────────────────────────────────────────────────────────────────

class Block(nn.Module):

    def __init__(self, dim, heads, max_seq, mlp_ratio, dropout):

        super().__init__()

        self.ln1 = nn.LayerNorm(dim)

        self.attn = CausalSelfAttention(dim, heads, max_seq, dropout)

        self.ln2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=False),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = x + self.dropout(self.attn(self.ln1(x)))

        x = x + self.dropout(self.mlp(self.ln2(x)))

        return x


# ──────────────────────────────────────────────────────────────────────────────
# Transformer Module
# ──────────────────────────────────────────────────────────────────────────────

class TransformerModule(nn.Module):
    """
    A self-contained transformer module.

    Each module processes token representations and returns updated
    hidden states.

    Future modules can be added alongside the TinyStories module.
    """

    def __init__(self, dim, layers, heads, max_seq, mlp_ratio, dropout):

        super().__init__()

        self.blocks = nn.ModuleList(
            [Block(dim, heads, max_seq, mlp_ratio, dropout) for _ in range(layers)]
        )

        self.ln = nn.LayerNorm(dim)

    def forward(self, x):

        for block in self.blocks:
            x = block(x)

        x = self.ln(x)

        return x


# ──────────────────────────────────────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────────────────────────────────────

class Router(nn.Module):
    """
    Simple routing layer.

    Currently:
        selects the TinyStories module (index 0).

    Future versions:
        learn routing decisions between multiple modules.
    """

    def __init__(self, dim, num_modules):

        super().__init__()

        self.num_modules = num_modules

        # lightweight router network
        self.router = nn.Linear(dim, num_modules)

    def forward(self, x):
        pooled = x.mean(dim=1)
        logits = self.router(pooled)
        probs = torch.softmax(logits, dim=-1)
        # Return probs for weighted combination during training
        # Return argmax for hard selection during inference
        if self.training:
            module_index = probs
        else:
            module_index = torch.argmax(probs, dim=-1)
        return module_index

# ──────────────────────────────────────────────────────────────────────────────
# Shared Language Core
# ──────────────────────────────────────────────────────────────────────────────

class LanguageCore(nn.Module):
    """
    Shared language layer.

    Provides:
        token embeddings
        output projection head

    All modules operate in this shared embedding space.
    """

    def __init__(self, vocab_size, dim):

        super().__init__()

        self.emb = nn.Embedding(vocab_size, dim)

        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.emb.weight

        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def embed(self, idx):

        return self.emb(idx)

    def logits(self, x):

        return self.head(x)


# ──────────────────────────────────────────────────────────────────────────────
# Scout Modular Model
# ──────────────────────────────────────────────────────────────────────────────

class ScoutModel(nn.Module):

    def __init__(self, vocab_size, config):

        super().__init__()

        dim = config["dim"]

        self.max_seq = config["block_size"]

        # Shared language layer
        self.language = LanguageCore(vocab_size, dim)

        # Router
        self.router = Router(dim, num_modules=1)

        # Module list
        self.modules_list = nn.ModuleList([
            TransformerModule(
                dim=dim,
                layers=config["layer"],
                heads=config["heads"],
                max_seq=config["block_size"],
                mlp_ratio=config["mlp_ratio"],
                dropout=config["dropout"],
            )
        ])

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.max_seq
        x = self.language.embed(idx)
        module_indices = self.router(x)
        # Currently always selects module 0
        module = self.modules_list[0]
        x = module(x)
        logits = self.language.logits(x)
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# Parameter Count Utility
# ──────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module):

    total = sum(p.numel() for p in model.parameters())

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_millions": round(total / 1e6, 2),
    }