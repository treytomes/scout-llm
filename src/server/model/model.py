"""
model.py — Scout Modular Transformer Architecture

Scout grows developmentally through sequential stacking of transformer modules.
Each phase adds a new module that operates on the *output* of all previous
modules — later modules focus and refine what earlier ones established.

Current architecture (Phase 0 — TinyStories linguistic foundation):

    tokens
      │
    shared embedding (LanguageCore)
      │
    Module 0 — TinyStories linguistic foundation
      │
    shared output head (LanguageCore)
      │
    logits

After Phase 1 (conversational layer added):

    tokens
      │
    shared embedding
      │
    Module 0 — TinyStories (frozen)
      │
    Module 1 — Conversational (trains on Scout's synthetic corpus)
      │
    shared output head
      │
    logits

Design principles
─────────────────

Sequential stacking — not mixture-of-experts.
    Later modules see the *output* of earlier ones, not the raw embeddings.
    Module 0 converts tokens into linguistic representations. Module 1
    refines those for conversational register. Module 2 refines further
    for reflective first-person voice. Each phase focuses the previous.

Freezing preserves developmental stages.
    Before adding Module 1, freeze Module 0. It becomes a deterministic
    transformation — a fixed lens through which all later modules see
    language. New modules train freely in the representation space that
    frozen modules have established.

Shared embedding space.
    All modules operate in the same dim-dimensional space (set by
    LanguageCore). Weight tying between embedding and output head reduces
    parameters and keeps the vocabulary grounded across phases.

Router — retained, currently dormant.
    A Router is available for future use. One possible application:
    sequence-level conditioning of how much each module contributes at
    inference time. Currently unused — the sequential stack is sufficient
    and simpler. The router can be activated if per-content-type routing
    proves useful at a later developmental stage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ──────────────────────────────────────────────────────────────────────────────
# Rotary Position Embeddings
# ──────────────────────────────────────────────────────────────────────────────

def precompute_rope_freqs(head_dim: int, max_seq: int, theta: float = 10000.0):
    """
    Precompute cosine and sine frequency tables for RoPE.

    Rotary embeddings encode position by rotating query and key vectors.
    Unlike learned absolute embeddings, RoPE generalises to unseen sequence
    lengths and encodes relative rather than absolute position — important
    for a model whose context window may grow across developmental stages.
    """
    assert head_dim % 2 == 0

    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq).float()
    angles = torch.outer(positions, freqs)
    angles = torch.cat([angles, angles], dim=-1)

    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Apply rotary position embeddings to query or key tensor."""
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
            q, k, v,
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
    One developmental layer in Scout's sequential stack.

    Each module receives the output of all preceding modules and returns
    refined hidden states in the same embedding space. When frozen, it
    acts as a deterministic transformation that later modules build on.

    Planned developmental sequence:
        Module 0 — TinyStories linguistic foundation
        Module 1 — Conversational pattern and register
        Module 2 — First-person reflective voice
        Module 3 — Inner voice and metacognition
    """

    def __init__(self, dim, layers, heads, max_seq, mlp_ratio, dropout):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(dim, heads, max_seq, mlp_ratio, dropout)
            for _ in range(layers)
        ])
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.ln(x)


# ──────────────────────────────────────────────────────────────────────────────
# Router (retained for future use)
# ──────────────────────────────────────────────────────────────────────────────

class Router(nn.Module):
    """
    Sequence-level routing layer — retained for future use.

    Currently dormant. The sequential stack does not require routing
    because all modules always run in order. This router is preserved
    for possible future applications such as per-content-type conditioning
    or weighted blending of module contributions at inference time.

    If activated for soft blending:
        Training:   returns (B, num_modules) softmax probabilities
        Inference:  returns (B,) integer module index (argmax)
    """

    def __init__(self, dim, num_modules):
        super().__init__()

        self.num_modules = num_modules
        self.gate = nn.Linear(dim, num_modules)

        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x):
        pooled = x.mean(dim=1)          # (B, dim)
        logits = self.gate(pooled)
        probs = torch.softmax(logits, dim=-1)

        if self.training:
            return probs                            # (B, num_modules)
        else:
            return torch.argmax(probs, dim=-1)      # (B,)


# ──────────────────────────────────────────────────────────────────────────────
# Shared Language Core
# ──────────────────────────────────────────────────────────────────────────────

class LanguageCore(nn.Module):
    """
    Shared token embedding and output projection.

    All modules operate in this embedding space. Weight tying between
    the embedding matrix and the output head reduces parameter count and
    keeps the vocabulary representation stable across developmental phases.

    Frozen after Phase 0 — all subsequent modules learn to work within
    the vocabulary space the linguistic foundation established.
    """

    def __init__(self, vocab_size, dim):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.emb.weight  # weight tying

        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def embed(self, idx):
        return self.emb(idx)

    def logits(self, x):
        return self.head(x)


# ──────────────────────────────────────────────────────────────────────────────
# Scout Model
# ──────────────────────────────────────────────────────────────────────────────

class ScoutModel(nn.Module):
    """
    Scout's developmentally-staged language model.

    Modules stack sequentially — each receives the output of all previous
    modules. Frozen modules become fixed transformations that new modules
    build on, preserving developmental stages while adding new capacity.

    Growth protocol per phase:
        1. Train current top module to convergence.
        2. Call freeze_module(index) on it.
        3. Optionally call freeze_language_core() after Phase 0.
        4. Call add_module(config) to append the next developmental layer.
        5. Train the new top module on the new phase corpus.

    The Router is retained but dormant. The forward pass runs all modules
    in sequence unconditionally — every token passes through every module
    in order, with earlier modules frozen into deterministic transformations.
    """

    def __init__(self, vocab_size, cfg):
        super().__init__()

        self._dim = cfg["dim"]
        self._max_seq = cfg["block_size"]

        self.language = LanguageCore(vocab_size, self._dim)

        # Router retained for future use — not used in forward pass currently
        self.router = Router(self._dim, num_modules=1)

        self.expert_modules = nn.ModuleList([
            TransformerModule(
                dim=cfg["dim"],
                layers=cfg["layer"],
                heads=cfg["heads"],
                max_seq=cfg["block_size"],
                mlp_ratio=cfg["mlp_ratio"],
                dropout=cfg["dropout"],
            )
        ])

    @property
    def max_seq(self):
        return self._max_seq

    def forward(self, idx):
        B, T = idx.shape

        assert T <= self._max_seq, (
            f"Sequence length {T} exceeds block size {self._max_seq}"
        )

        # Embed tokens into shared representation space
        x = self.language.embed(idx)

        # Pass through all modules sequentially
        # Each module refines the representation established by previous ones.
        # Frozen modules act as deterministic transformations.
        for module in self.expert_modules:
            x = module(x)

        return self.language.logits(x)

    # ──────────────────────────────────────────────────────────────────────────
    # Developmental Growth API
    # ──────────────────────────────────────────────────────────────────────────

    def freeze_module(self, module_index: int):
        """
        Freeze a transformer module by index.

        The frozen module becomes a deterministic transformation that all
        subsequent modules build on. Its learned representations are
        preserved as the foundation for the next developmental phase.
        """
        if module_index >= len(self.expert_modules):
            raise IndexError(
                f"Module index {module_index} out of range "
                f"({len(self.expert_modules)} modules)"
            )

        for param in self.expert_modules[module_index].parameters():
            param.requires_grad = False

        print(f"Module {module_index} frozen.")

    def freeze_language_core(self):
        """
        Freeze the shared embedding and output head.

        Call after Phase 0 to protect the vocabulary space the linguistic
        foundation established. All subsequent modules train within this
        space without altering it.
        """
        for param in self.language.parameters():
            param.requires_grad = False

        print("Language core frozen.")

    def add_module(self, cfg: dict):
        """
        Append a new transformer module to the sequential stack.

        The new module is randomly initialised and receives as input the
        output of all preceding (frozen) modules. It trains freely in the
        representation space those modules have established.

        The dim must match the existing architecture — all modules share
        the same embedding space.

        Args:
            cfg: Module configuration dict with keys: dim, layer, heads,
                 block_size, mlp_ratio, dropout.
        """
        assert cfg["dim"] == self._dim, (
            f"New module dim {cfg['dim']} must match existing dim {self._dim}"
        )

        new_module = TransformerModule(
            dim=cfg["dim"],
            layers=cfg["layer"],
            heads=cfg["heads"],
            max_seq=cfg["block_size"],
            mlp_ratio=cfg["mlp_ratio"],
            dropout=cfg["dropout"],
        )

        self.expert_modules.append(new_module)

        # Expand router to track module count (dormant but kept consistent)
        num_modules = len(self.expert_modules)
        old_router = self.router
        new_router = Router(self._dim, num_modules=num_modules)

        with torch.no_grad():
            new_router.gate.weight[:num_modules - 1] = old_router.gate.weight
            new_router.gate.bias[:num_modules - 1] = old_router.gate.bias
            new_router.gate.bias[num_modules - 1] = -2.0

        self.router = new_router

        print(
            f"Module {num_modules - 1} added to sequential stack. "
            f"{num_modules} modules total."
        )

    def unfreeze_module(self, module_index: int):
        """
        Unfreeze a module for continued training.

        Use with care — unfreezing a frozen module risks disturbing the
        representational foundation that later modules depend on.
        Prefer adding new modules over retraining existing ones.
        """
        if module_index >= len(self.expert_modules):
            raise IndexError(
                f"Module index {module_index} out of range "
                f"({len(self.expert_modules)} modules)"
            )

        for param in self.expert_modules[module_index].parameters():
            param.requires_grad = True

        print(f"Module {module_index} unfrozen. Proceed with care.")


# ──────────────────────────────────────────────────────────────────────────────
# Parameter Count Utility
# ──────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module):
    """
    Return total, trainable, and frozen parameter counts.

    The trainable/frozen split is the primary diagnostic for confirming
    the developmental freezing strategy is working correctly. Before
    adding a new module, verify that only the new module is trainable.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    per_module = {}
    for i, module in enumerate(model.expert_modules):
        m_total = sum(p.numel() for p in module.parameters())
        m_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        per_module[f"module_{i}"] = {
            "total": m_total,
            "trainable": m_trainable,
            "frozen": m_total - m_trainable,
            "total_millions": round(m_total / 1e6, 2),
        }

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_millions": round(total / 1e6, 2),
        "trainable_millions": round(trainable / 1e6, 2),
        "per_module": per_module,
    }