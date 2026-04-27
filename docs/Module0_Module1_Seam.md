# The Seam Between Module 0 and Module 1

*What happens at the boundary between Scout's linguistic foundation and her conversational layer.*

---

## High-Level Description

Module 0 is a transformer trained on TinyStories: short, simple, third-person narratives. It learns the grammar, syntax, and basic compositional structure of language — how tokens assemble into sentences, how sentences carry meaning. It does not learn conversational register, first-person presence, or anything resembling social exchange. Its output is a representation of *what the tokens mean linguistically*, not *who is speaking or why*.

Module 1 sits directly on top of Module 0's output. It does not see the raw token embeddings — it sees the hidden states that Module 0 has already processed. Its job is to refine those representations toward conversational and, eventually, first-person reflective register. It learns *from language that Module 0 has already understood*, focusing on what Module 0 left underdetermined: register, social orientation, speaker identity, emotional texture.

The seam between them is the hidden state tensor that flows from Module 0's final layer norm into Module 1's first transformer block. Both modules operate in the same 512-dimensional embedding space (set by `LanguageCore`). Module 0 never changes after freezing; it becomes a deterministic lens. Module 1 trains freely within the representation space that lens has established.

---

## Technical Details

### Shared Embedding Space

Both modules operate in `dim=512` space, defined by `LanguageCore`. The token embedding matrix (`nn.Embedding(vocab_size, 512)`) and the output head (`nn.Linear(512, vocab_size, bias=False)`) share weights. This means:

- Module 0 transforms `[B, T, 512]` embeddings into `[B, T, 512]` refined representations.
- Module 1 receives those `[B, T, 512]` representations and further transforms them into `[B, T, 512]`.
- The final `[B, T, 512]` output is projected to logits via the shared head.

Neither module changes the dimensionality of the representation. The seam is dimensionally transparent.

### Module 0 Architecture

```
12 transformer blocks × (dim=512, heads=8, mlp_ratio=3.5)
Final LayerNorm
```

Each block: `LayerNorm → CausalSelfAttention (RoPE) → Dropout → residual` then `LayerNorm → MLP (GELU) → Dropout → residual`.

The final `LayerNorm` at the end of Module 0 normalizes the hidden states before passing them to Module 1. This is important: Module 1 always receives normalized representations, regardless of what Module 0's internals did.

### Module 1 Architecture (planned)

```
7 transformer blocks × (dim=512, heads=8, mlp_ratio=3.5)
Final LayerNorm
```

Lighter dropout (`0.1` vs `0.15`) because Module 0's frozen representations provide a stable, pre-regularized input. Module 1 doesn't need to learn robustness against noisy inputs — it can assume Module 0's output is clean.

### The Forward Pass

```python
# ScoutModel.forward(), simplified
x = self.language.embed(idx)        # [B, T, 512]

x = self.expert_modules[0](x)      # Module 0: linguistic processing
# --- seam ---
x = self.expert_modules[1](x)      # Module 1: conversational refinement

return self.language.logits(x)     # [B, T, vocab_size]
```

The `skip_modules` parameter allows bypassing Module 1 at inference time:

```python
x = self.expert_modules[0](x)      # Module 0 runs
# Module 1 skipped — Module 0 output goes directly to the head
return self.language.logits(x)
```

This is implemented in `model.py` by checking `if skip_modules and i in skip_modules: continue` in the module loop. It lets us isolate each layer's contribution to the output.

### RoPE and Position Encoding

Both modules have their own `CausalSelfAttention` layers with their own RoPE buffers (`rope_cos`, `rope_sin`). Position information is re-encoded at each attention layer, not accumulated across module boundaries. Module 1 re-applies RoPE to its own Q/K projections independently of Module 0's position encoding.

This means: Module 1 re-perceives position from scratch on every token it attends to. It doesn't inherit Module 0's positional encoding — it recomputes its own from the hidden states it receives.

### Freezing Mechanics

After Phase 0 training, `model.freeze_module(0)` sets `requires_grad=False` on all Module 0 parameters. This is not a structural change — the forward pass is identical. Frozen parameters simply don't accumulate gradients during backprop. Module 0 becomes a deterministic transformation: the same input always produces the same output.

`model.freeze_language_core()` does the same for the embedding matrix and output head. After this, Module 1 trains against a fixed vocabulary space — it cannot shift what tokens mean, only how the model decides which token to predict next.

### What Module 0 Leaves Underdetermined

Module 0, trained on TinyStories, learns:
- Token co-occurrence statistics in narrative prose
- Basic syntactic structure (subject-verb-object, clause boundaries)
- Simple causal and temporal sequencing ("then", "after", "because")
- Third-person character references, object permanence within a scene

Module 0 does not learn:
- First-person speaker identity or consistency
- Conversational register (turn-taking, social acknowledgment, response relevance)
- Emotional self-description or interiority
- Multi-turn context integration as a *conversation* rather than a narrative

These gaps are exactly what Module 1's training corpus is designed to address. The seam passes Module 0's linguistically-coherent-but-register-flat representations to Module 1, which learns to apply register, voice, and conversational orientation on top of them.

---

## Observable Consequences of the Seam

### With Module 1 bypassed (Module 0 only)
Output reverts to TinyStories prior: third-person narrative prose, simple sentence structure, no conversational awareness. Social openers produce story continuations. This is the baseline — raw linguistic capacity without register.

### With Module 1 active
Output shifts toward conversational register (SODA-influenced, at Phase 1). More social scripts, first-person grammar, response-oriented completions. The register can be further refined by the voice calibration pass without retraining Module 0.

### Identity grounding problem
The module architecture alone does not solve identity grounding. Both layers combined still produce SODA persona adoption when greeted socially, because neither module contains a stable self-model. The seam passes linguistically-processed representations through a register filter — neither layer establishes *who is speaking*. That requires an inference-time identity prefix prepended before the conversation begins.

---

## Configuration Reference

```python
# config.py
MODEL_TINYSTORIES = {        # Module 0
    "dim": 512,
    "layer": 12,
    "heads": 8,
    "mlp_ratio": 3.5,
    "block_size": 512,
    "dropout": 0.15,
}

MODEL_CONVERSATIONAL = {     # Module 1
    "dim": 512,
    "layer": 7,
    "heads": 8,
    "mlp_ratio": 3.5,
    "block_size": 512,
    "dropout": 0.1,
}
```

Parameter counts (approximate):
- Module 0: ~25M parameters
- Module 1: ~14.5M parameters
- LanguageCore (shared): ~16M parameters (embedding + output head, weight-tied)
- Total (Phase 1): ~55M parameters, ~14.5M trainable
