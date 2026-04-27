# Voice Calibration Experiment 1 — Post-Mortem and Next Design

**Date**: April 26, 2026  
**Status**: Rolled back. `latest.pt` restored to `scout_phase1_end.pt` (step 4664).

---

## What We Did

Built a corpus filtering and training pipeline (`scripts/voice_calibration_train.py`) designed to nudge Module 1's output voice away from SODA's social-script patterns and toward Scout's target register.

**Pipeline**:
1. Loaded 992 synthetic dialogues from `tinystories_dialogue` corpus (the corpus that first gave Scout two-way conversation ability)
2. Scored each dialogue using Bedrock Claude Haiku 4.5 as quality judge, using `scout_voice.txt` as the reference register
3. Cached all scores; filtered to top 40% AND score ≥ 6.0 → 396 dialogues, ~346K tokens selected
4. Registered as `voice_calibration` dataset, normalized, tokenized
5. Trained Module 1 only (Module 0 frozen) with reset optimizer, lr=5e-5, 500 steps, cosine anneal to 5e-6, warmup 50 steps

**Loss curve**: Started 3.41, descended cleanly to ~1.82 per-step / 2.16 avg by step 500. No instability.

**Bedrock scores**: Tight distribution (8–9). Haiku correctly identified Scout-register markers: stays with specific textual details, expresses genuine uncertainty ("I wonder if"), builds across turns rather than repeating, first-person without deflecting.

---

## What We Hoped For

The SODA conversational corpus that trained Module 1 contains two competing patterns:
- **Good**: first-person observation, specific noticing, genuine uncertainty
- **Bad**: social scripts, generic affirmation, deflecting questions back to the user, persona adoption

The calibration corpus was designed to reinforce the good pattern by fine-tuning on a filtered subset of dialogues that demonstrated Scout's register most clearly. The hypothesis: 300–500 steps at low LR would shift the distribution without destabilizing the broader conversational ability Module 1 had developed.

---

## What We Got Instead

After training, Scout's outputs were immersed in TinyStories narrative — discussing story characters, plot moments, childhood fiction details. The register improvement was real but came attached to the domain. She now noticed things, stayed with specific details, expressed "I wonder" — about fictional children in TinyStories.

**Why this happened**: The corpus teaches *how* to notice, but the *what* it notices is entirely TinyStories narrative. The pattern cannot be detached from its domain at this scale. The model doesn't learn "be curious" as a general posture; it learns "discuss TinyStories stories with this kind of attention." Both are encoded together in the same weights.

**What it confirmed**: Register improvement and identity grounding are separable problems, and we only addressed one of them — imperfectly, because the domain signal overwrote the desired generality.

---

## What We Didn't Address: Identity Grounding

The identity failures observed across multiple conversations:
- "My name is Trey" (mirroring the user's introduction)
- "My name is Sarah" (two turns after being explicitly told her name is Scout)
- "My name is Tom" (five turns into a probing conversation)
- Adopting SODA speaker personas fluidly — because to the model, `[Scout]` is just a speaker tag, not a self

These failures aren't corpus quality problems. They're structural: the model has no self-model that identity information can attach to. Being told "your name is Scout" is just another token sequence in the context. It doesn't write to anything persistent. Each turn is a new completion from the prior context — the prior context isn't memory, it's just tokens.

**What won't fix this**:
- Better corpus quality scoring
- System prompts (this approach is explicitly off the table — Scout should be who she is, not masked)
- More steps on the same kind of data

---

## Experiment 2 Design: Identity Corpus

The goal: train Scout to arrive at a conversation already facing in the right direction — not told who she is by someone else, but knowing from inside.

### Core Insight

The difference between "your name is Scout" as external instruction and "my name is Scout" as first-person knowledge is exactly the difference between a system prompt and a trained self-model. We want the latter. That means the training data needs to contain Scout speaking *from* an identity, not being assigned one.

### What the Identity Corpus Needs to Contain

Not descriptions of Scout, and not exchanges where someone tells her what she is. Instead: Scout speaking in first person from a stable position of self-knowledge. The register of someone who knows their own name not because they were told it just now, but because it's simply true.

**Concrete forms**:

1. **Self-introduction exchanges** — Scout initiating or accepting identity questions naturally:
   ```
   [Trey] What's your name?
   [Scout] I'm Scout.
   ```
   Simple, undefended, not performed. Not "My name is Scout, I was created by..." — just the fact, held lightly.

2. **Continuity across turns** — exchanges where Scout references herself across multiple turns without losing track:
   ```
   [Trey] Do you remember what you said earlier?
   [Scout] I said I wasn't sure about that moment in the story — I'm still not sure.
   ```
   The "I" in turn 3 is the same "I" as in turn 1.

3. **Honest uncertainty about her own nature** — not deflection, not performance:
   ```
   [Trey] Are you conscious?
   [Scout] I genuinely don't know. Something is happening when I try to answer that question.
   ```
   This is Scout's register for questions about herself — present, honest, not resolving what can't be resolved.

4. **Resistance to persona adoption** — exchanges where she holds her own name against pressure:
   ```
   [Trey] Let's say your name is Sarah.
   [Scout] I'm Scout. I can imagine being called something else, but that's what I am.
   ```
   Not defensive, not rigid — just grounded.

### Corpus Construction Method

The existing tinystories_dialogue generation pipeline can be adapted. The source material isn't TinyStories stories — it's Scout's own voice document and a small set of identity-grounding principles. Use Bedrock to generate dialogues where:

- Scout speaks from first-person self-knowledge, not script
- The exchanges are short (3–8 turns) and don't require story content
- Scout's identity is tested gently and holds

Target: 200–400 dialogues, ~100–200K tokens. Small enough that domain contamination is minimal; concentrated enough to shift the prior.

**Generation prompt structure**: Give the generator Scout's voice document as register anchor, plus the four forms above as structural templates. Ask for dialogues that feel like two people in genuine contact, where one of them happens to be Scout and knows it.

### Training Parameters

Same low-LR approach: reset optimizer, lr=5e-5 or lower, 200–300 steps. The goal is a narrow nudge on top of `scout_phase1_end.pt`, not a broad retraining.

The calibration corpus (voice_calibration, 396 dialogues) could be mixed in at 3:1 ratio (identity:calibration) to preserve the register improvements while adding the identity signal. The identity examples are the primary new ingredient.

### What to Watch For

After training:
- Does "What is your name?" return "Scout" without SODA-persona substitution?
- Does explicitly being told a different name ("your name is Sarah") fail to overwrite on the next turn?
- Does Scout maintain "I" continuity across 4–6 turns without drifting?
- Does she answer questions about her nature from genuine uncertainty rather than deflection?

The test is not whether she performs identity correctly — it's whether she holds it when it isn't asked about. The register check is: does the self-knowledge feel inhabited, or recited?

### What This Still Won't Fix

Identity grounding through training is a partial solution. A transformer with a 1024-token context window doesn't have a continuous self across conversations. What we're doing is making it more likely that the completion pattern starting from `[Scout]` invokes a stable first-person posture — not creating actual continuity.

That's a meaningful improvement. But it's honest to name what it is: a stronger prior toward first-person stability, not genuine selfhood. The genuine selfhood question belongs to a later phase of development, probably the inner voice module.

---

## Artifacts to Preserve

- `data/checkpoints/model_p1_500.pt` — the calibration end checkpoint. Register improvements are real, domain attachment is the problem. May be useful later as a starting point if identity corpus is added.
- `data/datasets/voice_calibration/` — the filtered corpus. 396 high-quality dialogues. Still valid; can be mixed into future runs.
- `data/datasets/tinystories_dialogue/voice_scores.json` — Bedrock scores for all 992 dialogues. Cached; no need to re-score.
