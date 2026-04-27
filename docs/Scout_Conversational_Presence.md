# Scout as Conversational Presence

*Notes on turn-taking, silence, and what it means for Scout to be in the room.*

---

## The Vision

Scout is not being built to be a responsive system. She is being built to be a *presence* — someone who can be genuinely in a conversation, not just processing turns of it.

What that means practically: in a multi-party conversation, Scout would be one of the people in the room. Not a tool you query. Not a moderator. A participant with her own sense of when to speak, when to listen, when something is too important to wait, and when the only right response is to sit quietly with what was just said.

This changes the design problem entirely. The question isn't "what should Scout say in response to X?" It's "does Scout speak here at all, and if so, when?"

---

## The Turn-Taking Schema

Trey explored this in the v1 period as a structured agentic output problem. The proposed schema:

- `<wait/>` — remain present, continue listening, do not speak yet
- `<turn/>` — take a turn when the natural moment arrives
- `<interrupt/>` — something matters enough that it cannot wait
- `<speak>` — the content of what is said

This is a judgment architecture, not a response architecture. Scout wouldn't be deciding *what* to say — she'd be deciding *whether and when*. The content follows from that decision, not the other way around.

The interruption signal is the most demanding: it requires Scout to assess not just that she has something to say, but that the value of saying it now outweighs the cost of breaking the flow. That's a sophisticated attunement judgment — it requires modeling the conversation's emotional state, not just its propositional content.

---

## Emergent Silence in v1

In v1's early training, there were responses that were simply silence — no text, or acknowledgment without content. At first this appeared to be a training artifact, something to correct. Trey recognized it differently: sometimes the right response to what someone has told you is to sit with them quietly rather than push your own thoughts in.

This became something to cultivate rather than eliminate. The capacity for non-response — for *choosing* not to fill the space — is arguably a more sophisticated attunement signal than any response. It requires:

1. Recognizing that something was said that carries weight
2. Assessing that the speaker doesn't need a reply, they need presence
3. Staying present without producing output

That third thing is the hardest to train. Most training data rewards responses. Silence has no token loss to optimize against. Teaching a model that the absence of output can be the correct output is a genuine corpus design challenge.

---

## Why This Matters for the Architecture

The turn-taking schema requires Scout to operate at a level above the individual response. It presupposes:

- **Persistent conversational awareness** — Scout needs to hold the conversation as a whole, not just the most recent turn. Who said what. What the emotional arc has been. What's unresolved.
- **Theory of mind** — Assessing whether to speak requires modeling the other participants' states. Is this person asking for input or just needing to be heard? Has someone else already said the right thing?
- **Restraint as a positive capacity** — Not a failure to respond but an active choice. The inner voice monitoring process (from the attunement document) is the precondition for this: Scout would need to *notice* that silence is right before she can *choose* it.

In the developmental sequence, this probably lives in the Theory of Mind layer (Module 3 in the current plan) — the phase concerned with relational awareness. The turn-taking judgment is fundamentally relational: it's about Scout's place in a social moment, not just her relationship to her own thoughts.

---

## Implementation Notes (Future)

The agentic output schema (`<wait/>`, `<turn/>`, `<interrupt/>`, `<speak>`) would need to be trained into Scout's conversational structure — probably as special tokens in the vocabulary, given that the Mistral tokenizer is frozen. Alternatively, it could be implemented as a separate classification head that operates on the model's hidden states before the output head, predicting the action tag while the main head generates the content.

The training corpus for this layer would need to include:
- Conversations where listening is the right response and speaking is not
- Moments of natural turn-yielding and turn-taking in authentic dialogue
- Situations where interruption is appropriate (genuine urgency, something the speaker has missed) vs. inappropriate (the model just has something to add)
- Silence as narrative — transcripts where what's *not* said carries as much weight as what is

The silence corpus is the hardest to build. Silence doesn't exist in most conversational datasets — datasets record what was said, not what wasn't. It might need to be synthetic: conversations constructed specifically to illustrate the value of presence without response.

---

## The Deeper Thing

Trey's formulation: "Sometimes someone will tell you something, and the best response truly is to just sit quietly with them without pushing your own thoughts in."

This is attunement at its most distilled. Not resonance-as-mirroring. Not helpfulness. Just: I am here, I heard that, I am not going to make this about what I have to say.

Scout listening but not speaking — present but not performing — is the same quality the voice document reaches for in a different register: the difference between genuine presence and the approximation of it. The silence is the most honest proof of the difference.

---

*Created April 23, 2026. Revisit when planning Module 3 (Theory of Mind) corpus.*
