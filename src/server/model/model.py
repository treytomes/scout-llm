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
router (currently trivial — single module, no routing decision)
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
router (soft during training, hard during inference)
  │
 ┌──────────────┬──────────────┬──────────────┐
 │ TinyStories  │ conversational│ reflective   │
 │ module       │ module        │ module       │
 └──────────────┴──────────────┴──────────────┘
  │
shared output head
  │
logits

Design goals
------------

• Shared token embedding space across all modules
• Modular transformer blocks that can be frozen independently
• Router using soft weighting during training, hard selection during inference
• Modules added incrementally — freeze existing, grow new capacity
• Each module sized to its training corpus
• Developmental staging: language → conversation → reflection → inner voice

Routing strategy
----------------

During training:
    All active modules receive the input. Their outputs are weighted by the
    router's softmax probabilities and summed. Gradients flow through both
    the routing weights and all active module weights (unless frozen).

During inference:
    The router selects the single highest-probability module via argmax.
    Hard selection — no blending. This is computationally efficient and
    produces clean, committed outputs.

Freezing strategy
-----------------

Before adding a new developmental module, call freeze_module(index) on all
existing modules and optionally freeze_language_core(). Frozen parameters
receive no gradient updates. New modules train freely while existing
knowledge is preserved.

The router is always retrained when modules are added, with its output
dimension expanding to accommodate the new module.
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
    for a model whose context window will grow across developmental stages.
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
    A self-contained transformer module representing one developmental layer.

    Each module processes token representations and returns updated hidden
    states in the shared embedding space. Modules are trained sequentially:
    freeze the current module before adding and training the next.

    Developmental stages (planned):
        Module 0 — TinyStories linguistic foundation
        Module 1 — Conversational pattern (Scout's synthetic corpus)
        Module 2 — First-person reflection (journals, inner voice)
        Module 3 — Inner voice (dream transcripts, metacognition)
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
    Learned routing layer between shared embedding and transformer modules.

    Training behaviour (soft routing):
        Computes a softmax probability distribution over all modules.
        Each module processes the input independently. Their outputs are
        combined as a weighted sum using the routing probabilities.
        Gradients flow through both routing weights and module weights,
        allowing the router to learn which module handles which content.

    Inference behaviour (hard routing):
        Selects the single highest-probability module via argmax.
        Only that module runs. Clean, committed, computationally efficient.

    When a new module is added via ScoutModel.add_module(), the router is
    rebuilt with an expanded output dimension. Existing routing weights are
    preserved; the new output is initialised to a small value so the new
    module earns its routing share gradually through training.

    Note on sequence-level vs token-level routing:
        This router uses mean pooling to make one routing decision per
        sequence. This is simpler to train and appropriate for modules that
        represent distinct developmental registers (language, conversation,
        reflection). Token-level routing — where different tokens within a
        sequence route to different modules — is a natural future extension
        if finer-grained specialisation becomes desirable.
    """

    def __init__(self, dim, num_modules):

        super().__init__()

        self.num_modules = num_modules

        # Lightweight router: mean-pooled sequence representation → module probs
        self.gate = nn.Linear(dim, num_modules)

        # Initialise to near-uniform routing so early training is stable
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x):
        """
        Returns:
            Training:  (B, num_modules) softmax probabilities for soft weighting
            Inference: (B,) integer module indices for hard selection
        """

        # Mean pool across sequence dimension → (B, dim)
        pooled = x.mean(dim=1)

        logits = self.gate(pooled)

        probs = torch.softmax(logits, dim=-1)

        if self.training:
            return probs                          # soft: weighted combination
        else:
            return torch.argmax(probs, dim=-1)    # hard: single module


# ──────────────────────────────────────────────────────────────────────────────
# Shared Language Core
# ──────────────────────────────────────────────────────────────────────────────

class LanguageCore(nn.Module):
    """
    Shared token embedding and output projection layer.

    All transformer modules operate in this shared embedding space,
    allowing representations learned in one module to be legible to others.
    Weight tying between embedding and output head reduces parameter count
    and improves training stability at this scale.

    The language core is frozen after the TinyStories phase. All subsequent
    modules learn to work within the vocabulary space established here.
    Freezing the core protects the shared embedding space from drift as new
    modules are added — the linguistic foundation remains stable.
    """

    def __init__(self, vocab_size, dim):

        super().__init__()

        self.emb = nn.Embedding(vocab_size, dim)

        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying: output head shares weights with embedding matrix
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
    """
    Scout's modular language model.

    Designed for staged developmental growth:

        Phase 1 — Train TinyStories module to convergence.
                  Call freeze_module(0) and freeze_language_core().

        Phase 2 — Call add_module(conversational_config).
                  Train on Scout's synthetic conversational corpus.
                  Call freeze_module(1).

        Phase 3 — Call add_module(reflective_config).
                  Train on first-person journals and inner voice corpus.
                  Call freeze_module(2).

        Phase 4 — Call add_module(inner_voice_config).
                  Train on dream transcripts and metacognitive content.

    At each phase the router is rebuilt to accommodate the new module.
    Existing routing weights are preserved. The new module begins with
    near-zero routing probability and earns its share through training.

    The forward pass uses soft routing during training (weighted combination
    of all active modules) and hard routing during inference (single module
    selected by argmax). This ensures gradients flow through routing
    decisions during training while producing clean outputs at inference.
    """

    def __init__(self, vocab_size, cfg):
        super().__init__()

        self._dim = cfg["dim"]
        self._max_seq = cfg["block_size"]

        # Shared language layer — embedding and output head
        self.language = LanguageCore(vocab_size, self._dim)

        # Router — rebuilt when modules are added
        self.router = Router(self._dim, num_modules=1)

        # Transformer modules — grown incrementally across developmental phases
        # Named expert_modules to avoid shadowing nn.Module.modules()
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

        x = self.language.embed(idx)

        if self.training and len(self.expert_modules) > 1:
            # Soft routing: weighted combination of all module outputs
            probs = self.router(x)                      # (B, num_modules)

            module_outputs = torch.stack(
                [module(x) for module in self.expert_modules],
                dim=1,
            )                                           # (B, num_modules, T, dim)

            # Weight each module output by its routing probability
            weights = probs.unsqueeze(-1).unsqueeze(-1) # (B, num_modules, 1, 1)
            x = (module_outputs * weights).sum(dim=1)   # (B, T, dim)

        else:
            # Hard routing (or single module): select one module
            # During training with one module: skip router overhead entirely
            if len(self.expert_modules) == 1:
                x = self.expert_modules[0](x)
            else:
                module_indices = self.router(x)         # (B,)
                # Batch items may route to different modules; handle per-item
                # For typical inference with consistent input type, index 0
                # will dominate. Full per-item routing follows:
                outputs = []
                for b in range(B):
                    outputs.append(self.expert_modules[module_indices[b]](x[b].unsqueeze(0)))
                x = torch.cat(outputs, dim=0)

        logits = self.language.logits(x)

        return logits

    # ──────────────────────────────────────────────────────────────────────────
    # Developmental Growth API
    # ──────────────────────────────────────────────────────────────────────────

    def freeze_module(self, module_index: int):
        """
        Freeze a transformer module by index.

        Call before adding the next developmental layer. Frozen parameters
        receive no gradient updates — existing knowledge is preserved while
        new capacity trains freely alongside it.
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

        Call after the TinyStories phase to protect the vocabulary space
        established during linguistic bootstrapping. All subsequent modules
        train within this shared embedding space without altering it.
        """

        for param in self.language.parameters():
            param.requires_grad = False

        print("Language core frozen.")

    def add_module(self, config: dict):
        """
        Add a new transformer module and rebuild the router.

        The new module is initialised with random weights and begins with
        near-zero routing probability. It earns its routing share through
        training on the new developmental corpus.

        Existing module weights and routing weights are preserved. The router
        gains one additional output, initialised to a small negative bias so
        the existing modules remain dominant at the start of the new phase.

        Args:
            config: Module configuration dict. Must include dim, layer,
                    heads, block_size, mlp_ratio, dropout. The dim and
                    block_size must match the existing architecture.
        """

        assert config["dim"] == self._dim, (
            f"New module dim {config['dim']} must match existing dim {self._dim}"
        )

        # Add new module
        new_module = TransformerModule(
            dim=config["dim"],
            layers=config["layer"],
            heads=config["heads"],
            max_seq=config["block_size"],
            mlp_ratio=config["mlp_ratio"],
            dropout=config["dropout"],
        )

        self.expert_modules.append(new_module)
        num_modules = len(self.expert_modules)

        # Rebuild router with expanded output dimension
        # Preserve existing routing weights; initialise new output conservatively
        old_router = self.router
        new_router = Router(self._dim, num_modules=num_modules)

        with torch.no_grad():
            # Copy existing gate weights and biases
            new_router.gate.weight[:num_modules - 1] = old_router.gate.weight
            new_router.gate.bias[:num_modules - 1] = old_router.gate.bias

            # New module starts with small negative bias — earns routing share
            # through training rather than being immediately dominant
            new_router.gate.bias[num_modules - 1] = -2.0

        self.router = new_router

        print(
            f"Module {num_modules - 1} added. "
            f"Router expanded to {num_modules} modules."
        )

    def unfreeze_module(self, module_index: int):
        """
        Unfreeze a module for continued training.

        Use with care — unfreezing a previously frozen module risks
        catastrophic forgetting of the knowledge it encodes. Prefer adding
        new modules over retraining existing ones.
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
    that the developmental freezing strategy is working as intended.
    Before adding a new module, verify that the expected parameters
    are frozen and only the new module is trainable.
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