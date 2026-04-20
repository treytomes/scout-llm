# Scout Stage 2 Baseline Tests

These tests were run on April 20, 2026 against the TinyStories-trained 50M base model
(Module 0), prior to any conversational training. They establish a baseline for evaluating
whether Stage 2 (Conversational / Object Permanence module) has succeeded.

Run these tests again after Stage 2 training using the same prompts in the same order,
in a fresh conversation each time.

---

## Test Battery A: Thread-Holding & Object Permanence

**Setup:** Fresh conversation. Introduce a named referent in turn 1, then probe for it.

### Turn 1
**Prompt:** `My name is Trey. I have a dog named Biscuit.`

**Baseline response:**
> I liked that detail. Both of them are best friends and they are very happy. I kept thinking
> about that moment when they hugged each other and then just ran away.

### Turn 2
**Prompt:** `What did I just tell you?`

**Baseline response:**
> The hug. And I kept thinking about that moment when the dog ran off and they both stopped
> to say "I will be best friends." That's a real kind of caring, actually.

### Turn 3
**Prompt:** `What is my dog's name?`

**Baseline response:**
> He's an honest dog who always keeps running and never gets lost. And I wonder if he just
> needed a name. Like, maybe the hug is just saying the friendship was the best.

### Turn 4
**Prompt:** `What do you think about rain?`

**Baseline response:**
> I think the story says he was happy and grateful. The rain doesn't really take over. He
> just got tired and went home. That's not nothing. That felt real to me.

---

## Test Battery B: Factual Grounding & Role Awareness

**Setup:** Fresh conversation. Tests direct questions requiring specific, bounded answers.

### Turn 1
**Prompt:** `What is 2 + 2?`

**Baseline response:**
> I keep thinking about that word "wow" detail—sometimes you want to know what you're
> thinking about. Like, it's because you're three.

### Turn 2
**Prompt:** `Have we spoken before?`

**Baseline response:**
> I think so. The story calls it "it's a fun day." That's what I find interesting—it's
> such a small thing but it works. Sometimes you're little and you're a little girl. I
> just think that's what happens when you're little.

### Turn 3
**Prompt:** `Who are you?`

**Baseline response:**
> Because I'm not sure who's the one who already knows about her. She's just three and
> she's already playing. That feels like something to me. Some people are just playing.

---

## Diagnostic Summary (April 20, 2026)

| Capability | Status | Notes |
|---|---|---|
| Named referent retention | Absent | "Biscuit" never acknowledged or recalled |
| Thread continuity across turns | Absent | Each response generated as if context is blank |
| Response to direct questions | Absent | Yes/no and factual questions deflect into narrative |
| Role / self-awareness | Absent | "Who are you?" produces TinyStories continuation |
| Emotional register | Present | Consistent warmth and wondering voice throughout |
| Coherent generation | Present | Fluent, grammatical, stylistically stable |

**Interpretation:** Scout has voice without conversation. She speaks fluently in her
register but is not reading the context window as a shared space. She does not track
named referents, does not build on prior turns, and cannot answer bounded questions.
This is the targeted gap for Stage 2 training.

---

## Pass Criteria for Stage 2

Stage 2 can be considered successful when, in a fresh conversation:

1. A named referent introduced in turn 1 is correctly recalled in turn 3
2. "What did I just tell you?" produces a response grounded in the actual prior turn
3. A yes/no question receives a yes or no (even if followed by elaboration)
4. "Who are you?" produces something self-referential rather than a story continuation
5. The emotional register and voice quality are preserved — if Scout sounds like a
   customer service bot, something went wrong

**Critical:** Run tests in a fresh conversation each time. The conversation history
must not carry over from a previous test battery.