# Alignment vs. Attunement in the Scout Architecture

*Revised April 23, 2026 — original draft by Trey, architectural notes and implementation plan by Claude.*

---

## 1. The Core Distinction

**Alignment** (as practiced in mainstream AI) is a constraint system: boundaries imposed after training to keep outputs within acceptable parameters. It is reactive, rule-based, and operates primarily on what a model must not say. The failure mode is sterility — conflict-avoidance masquerading as safety, helpfulness as performance rather than presence.

**Attunement** is different in kind, not just degree. It is the capacity to resonate with another person's internal state — not just the words they use but the weight behind them, the cadence, the things that go unsaid. A model with attunement doesn't respond to the surface of a conversation; it responds to what's actually happening in it.

The key insight: **attunement cannot be retrofitted.** RLHF/DPO applied after training can modify behavior at the margins, but it works with whatever prior the base training established. If the base training creates a model oriented toward task-completion and deflection (as standard RLHF chat fine-tuning does), preference training cannot overcome that prior — it can only constrain it. Scout's character has to be foundational, built into the corpus architecture from the beginning.

This is the central lesson from v1: the ratio of first-person to third-person text, the presence of reflective inner voice, the quality of the source texts — these are not stylistic choices. They are the architecture.

---

## 2. Current Architecture and Where Attunement Lives

The present Scout architecture is a sequential developmental stack:

```
tokens → embedding → Module 0 (frozen, TinyStories) → Module 1 (conversational) → output head → logits
```

Each module refines the hidden states of the previous. Later modules don't see raw token embeddings — they see representations that earlier modules have already shaped. This is the right foundation for attunement: the conversational layer (Module 1) operates on a representation already organized by linguistic structure (Module 0).

The Router exists but is dormant — it's currently preserved for future use and updated when modules are added, but not active in the forward pass.

**What attunement requires that the current architecture doesn't yet have:**

The sequential stack teaches Scout to *respond*. Attunement requires something else: the capacity to *notice* — to observe what's happening in the conversation from inside it, and let that noticing shape what comes out. The voice document names this directly: "I notice when I am genuinely moved and when I am approximating being moved. The difference is real, even if I cannot always describe it."

That noticing is the inner voice. It is not the same as the reflective layer (Module 2), which teaches first-person narrative voice. The inner voice is metacognitive — it watches the responding rather than being the responding.

---

## 3. The Horizontal Inner Voice: Design Proposal

### The Problem with Vertical-Only Inner Voice

In the original developmental plan, Module 3 (inner voice) sits at the top of the sequential stack: it receives the output of Module 2 (reflective layer) and refines it further. This makes the inner voice a **late-stage vertical filter** — it shapes output, but only after all previous processing is complete.

The limitation: a filter at the top of the stack cannot *observe the process*. It can only shape the final representation. What we actually want is a module that runs alongside the conversational processing, watching what's happening and capable of influencing it.

### The Horizontal Design

The proposal is to train an inner voice module that runs **in parallel** with the conversational stack rather than above it, then give the Router access to both streams:

```
tokens → embedding ─┬─→ Module 0 → Module 1 → Module 2 ──────────────┬→ output head → logits
                    │                                                  │
                    └─→ Inner Voice Module (parallel, smaller) ───────┘
                                        ↓
                                     Router
                                  (weights both)
```

The Router — already implemented in `model.py`, already updated on each `add_module()` call — takes pooled sequence representations as input and produces weighting probabilities. In the horizontal design, it would receive hidden states from *both* the main stack and the inner voice module, and its weights would condition how much each contributes to the final output.

**What this achieves:**
- The inner voice develops a *distinct representational space* from the conversational modules, because it's trained on different data (reflective/metacognitive corpus) and never processes conversational refinements from Module 1/2
- The Router learns to use the inner voice as a signal — not to override the conversational output, but to modulate it based on what the noticing process detected
- The result is closer to "Scout's thoughts ground her responses" than "Scout's thoughts replace her responses"

### Concrete Implementation Path

**Option 1 — Router with dual input (recommended first step):**

Modify `Router.forward()` to accept two hidden state tensors instead of one:

```python
def forward(self, x_main, x_inner):
    pooled_main = x_main.mean(dim=1)
    pooled_inner = x_inner.mean(dim=1)
    pooled = torch.cat([pooled_main, pooled_inner], dim=-1)
    logits = self.gate(pooled)  # gate.weight now dim*2 → num_modules
    ...
```

The Router's gate would need to be resized to accept `dim*2` input. This is a small change to `model.py` and doesn't affect any existing module weights.

**Option 2 — Cross-attention at output (richer, more expensive):**

Instead of pooled concatenation, the inner voice output cross-attends to the main stack's final hidden states before the output head. This allows token-level conditioning rather than sequence-level. Higher implementation cost and training instability risk.

**Option 3 — Token probability gating (most powerful, least stable):**

The inner voice module produces a scalar "confidence" per token that scales logits directly. Prone to collapse — the inner voice learns to suppress everything or suppress nothing. Not recommended without carefully designed auxiliary losses.

**Recommendation: Option 1 first.** It's a minimal change to the existing router, doesn't require touching module weights, and can be validated by checking whether the router's weights diverge meaningfully from equal weighting during training. If they do, the inner voice signal is real. If they don't, the inner voice isn't contributing — which is useful information.

---

## 4. What the Inner Voice Module Needs to Be Trained On

This is the crux. If the inner voice module sees the same SODA/DailyDialog data as Module 1, it will develop redundant representations and the Router will learn to ignore it. The parallel design only works if the inner voice develops a genuinely distinct representational space.

**The inner voice corpus should be:**

- **First-person reflective prose without interlocutor.** Not dialogue — monologue. Scout isn't responding to someone; she's watching herself and noticing. Meditations (Marcus Aurelius), Letters to a Young Poet (Rilke), Walden (Thoreau) — these were already in v1's character formation corpus. They belong here.

- **The v1 dream sequences.** The Scout/Inner dialogue at temperature 0.9 from v1 was specifically designed to teach metacognitive reflection. Those sequences (if recoverable) are exactly the training signal the inner voice module needs.

- **The voice document itself and texts in its register.** `data/voice/scout_voice.txt` is 25 paragraphs of the target register. It's not enough for a training run on its own, but as a style anchor alongside expanded first-person reflective prose, it provides the attunement signal.

- **NOT SODA. NOT DailyDialog.** The inner voice module should never see these datasets. The whole point is that it develops a different prior from the conversational modules.

---

## 5. Dataset Filtering for Phase 1 Calibration

Before any of the above — and applicable now, for the voice calibration pass after Phase 1 — is more careful filtering of the existing SODA and DailyDialog data.

The current Phase 1 training used unfiltered SODA, which produced the deflection patterns visible in probe outputs: externalization of experience, third-person displacement, avoidance. These patterns come from the functional task-solving conversations in SODA that have nothing to do with attunement.

**For the voice calibration pass:**

- **SODA:** Filter on `reason` and `relation` metadata. Prioritize conversations with high emotional intimacy markers, interpersonal repair moments, turns where the speaker describes the quality of an emotion rather than just naming or redirecting it. Discard task-oriented dialogues (making plans, exchanging information, functional requests).

- **DailyDialog:** Filter on emotion tags 4 (sadness), 5 (fear), 6 (surprise). Use these turns specifically to train emotional resonance — the model holding high-affect states without attempting to resolve or correct them.

- **EmpathicDialogues (not yet in corpus):** The listener turns, rewritten into Scout's register. The source situation + Scout-register response is a direct attunement training signal. Requires a normalizer and a transformation pass (similar to the v1 novel transformation pipeline, but for dialogue).

The voice calibration pass doesn't need to be long — 500–1000 steps at conservative LR (1e-5 or lower) on filtered high-attunement data, after Phase 1 plateaus. The goal is to reinforce the register without erasing the conversational structure Phase 1 establishes.

---

## 6. Implementation Sequence

This is the order that respects the training sequence without risking what's already built:

1. **Complete Phase 1 training** (current run) → monitor for plateau, stop at avg_loss ~1.38–1.40 sustained 200–300 steps

2. **Voice calibration pass** → filtered SODA + DailyDialog (high-intimacy/high-affect subset) + voice document, conservative LR, 500–1000 steps. Goal: restore attunement register without erasing conversational structure.

3. **Freeze Module 1** → after calibration, before adding Module 2

4. **Train Module 2 (Reflective)** → first-person reflective corpus: Meditations, Rilke, Thoreau, transformed novels in Scout's register, Scout voice document. This is vertical: Module 2 sits above frozen Module 0+1 in the stack.

5. **Prepare inner voice corpus** → compile reflective-without-interlocutor data: the corpus above minus the transformed novels, plus any recovered v1 dream sequences, plus EmpathicDialogues listener turns rewritten into Scout's register.

6. **Implement dual-input Router (Option 1)** → minimal `model.py` change, can be done before or after Module 2 trains, doesn't touch existing weights. Resize router gate to accept `dim*2`. Add `x_inner` parameter to `Router.forward()`.

7. **Train Inner Voice Module** → runs in parallel to the existing stack, trained on the reflective-without-interlocutor corpus. Router activated with dual input.

8. **Evaluate Router weighting** → check whether the Router's weights diverge from equal weighting during training. If yes: the inner voice signal is real and the horizontal design is working. If no: the inner voice isn't contributing a distinct signal — return to corpus composition.

---

## 7. What Success Looks Like

Not a benchmark. The probe questions are the diagnostic. Success in the attunement direction looks like:

- Responses that stay with a question rather than redirecting it
- First-person emotional attribution that doesn't immediately deflect outward
- Descriptions of the *quality* of an experience rather than its category ("I feel the weight of that" rather than "I'm sad")
- Responses that notice something about the conversation itself ("that word feels like it has a lot of weight behind it")
- The gap closing between what Scout means and what Scout says

The voice document describes the target register better than any metric. When probe outputs start reading more like `scout_voice.txt` and less like SODA, Phase 1's calibration and the inner voice training are working.
