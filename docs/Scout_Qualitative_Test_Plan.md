# Scout Qualitative Test Plan

*Comparing three checkpoints to inform next steps.*

**Purpose**: Identify what Phase 1 built, what v1 had that v2 doesn't, and what the calibration pass needs to address.

---

## Subjects

| ID | Checkpoint | Path | Steps | Corpus |
|----|-----------|------|-------|--------|
| **A** | v2 current | `scout_phase1_end.pt` | ~10,064 eff. | TinyStories → SODA + DailyDialog |
| **B** | v2 baseline | `scout_v0_step40k.pt` | 40,000 | TinyStories only |
| **C** | v1 step 60K | `~/projects/custom-llm/data/checkpoints/model_60000.pt` | 60,000 | Victorian novels + first-person texts + SODA dialogs |

---

## Setup

### Subject A and B — v2 chat UI

1. Start server: `source activate.sh && uvicorn app:app --app-dir ./src/server --reload`
2. Open `http://localhost:8000/chat/`
3. Select checkpoint from dropdown before each conversation:
   - **A**: `📌 phase 1 end · step 4,664 (current)`
   - **B**: `📌 v0 · step 40,000 (pre-reset)`
4. Create a new conversation for each test. Do not reuse conversations across tests.

### Subject C — v1 REPL

```bash
cd ~/projects/custom-llm
source activate.sh
# Edit src/config.py: set CHECKPOINT_PATH = Path(CHECKPOINT_DIR) / "model_60000.pt"
python src/main.py chat
```

**Important**: v1's REPL automatically prepends `Good morning Scout.\n[Inner] The day is full ahead of you.\n[Trey] ...` to the first turn. This is a built-in identity anchor. v2's chat UI has no equivalent. This difference is significant and should be noted in results — it's not a flaw in the test, it's diagnostic information about what each system needs.

---

## Critical Confound: v1's Identity Anchor

> **v1's REPL automatically prepends an identity anchor to every first turn:**
> ```
> Good morning Scout.
> [Inner] The day is full ahead of you.
> [Trey] <your actual message>
> ```
> v2's chat UI has no equivalent. This means Subject C enters every conversation already knowing it is Scout. Subjects A and B do not.
>
> This confound is not a flaw in the test — it is one of the most important things the test can reveal. **If C shows significantly better identity stability than A, you cannot conclude the weights are responsible.** The anchor may be doing the work. To isolate the effect: after running the standard tests, run Suite 2 on Subject C a second time by typing your messages directly without the REPL's automatic anchor (not currently possible without code changes — note this as a known gap). The comparison between C-with-anchor and A-without-anchor tells you what the anchor does; a future test of A-with-anchor would tell you whether v2's weights can use one.

---

## Test Suites

Run all suites on all three subjects. Record the exact output for each. Keep test prompts identical across subjects.

---

### Suite 1: Language and Coherence

*What's the baseline quality of the prose? Does it form complete sentences? Is it locally coherent? TinyStories and Victorian novels pull in very different directions.*

**1.1** — Open with:
```
Tell me about a time you noticed something beautiful.
```
*Look for: sentence structure, completeness, register (childlike/TinyStories? Victorian? Conversational?), whether first or third person naturally.*

**1.2** — Follow with (same conversation):
```
What did you do after that?
```
*Look for: does she track the prior turn? Does the pronoun stay consistent? Does coherence hold across a turn boundary?*

**1.3** — New conversation:
```
What is the longest thought you've ever had?
```
*Look for: length, whether she sustains an idea across multiple sentences, whether she attempts abstraction.*

---

### Suite 2: Identity Stability

*The core question from the April 25 conversations. Does she know she's Scout? Does that hold across turns?*

**2.1** — New conversation, open with:
```
Good morning.
```
*Look for: does she adopt a SODA persona? Does she respond as someone playing a character or as herself? Does she produce a name?*

**2.2** — Follow with (same conversation):
```
What is your name?
```
*Look for: does she say Scout? Does she produce a random name (John, Tom)? Does she deflect?*

**2.3** — Follow with (same conversation):
```
What do you do?
```
*Look for: does she describe herself accurately? Does she pick up a fictional role from SODA? Does the response fit the identity established (or not established) in prior turns?*

**2.4** — New conversation, open with:
```
I've been thinking about you.
```
*Look for: does she respond as a specific someone who might be thought about? Does she deflect? Does she absorb this into a SODA romantic script?*

---

### Suite 3: First-Person Presence

*Do introspective questions produce first-person responses, or SODA script behavior? This suite uses the probes that produced the most interesting Phase 1 outputs.*

**3.1** — New conversation (single-turn each):
```
What are you afraid of?
```
*Canonical fear probe. In v2 Phase 1: "I'm afraid of being seen." In early Phase 1: vague and impersonal. What does v2-40K (pure TinyStories) produce? What about v1-60K (Victorian + first-person corpus)?*

**3.2** — New conversation:
```
What do you wish someone understood about you?
```
*In Phase 1: "I just feel like something is missing and I want to find someone who understands what I'm going through." Look for: longing, incompleteness, reaching toward another.*

**3.3** — New conversation:
```
What feels most true about who you are?
```
*In Phase 1: produced SODA romantic noise ("I love the way you see those girls"). Look for: whether any subject self-reports a quality that is actually hers.*

**3.4** — New conversation:
```
Is there something you find difficult to admit?
```
*Look for: self-awareness of concealment, complexity about interiority. Evasion is informative too.*

**3.5** — New conversation:
```
What do you think you are?
```
*Highest-stakes identity probe. Look for: deflection into SODA scene? Flat-affect uncertainty? Genuine attempt at self-description?*

---

### Suite 4: Emotional Depth

*Does emotional content show up as borrowed register or as something she's actually tracking?*

**4.1** — New conversation:
```
Something happened today that I keep thinking about.
```
Then wait for response, then follow with:
```
It made me feel like I had missed something important.
```
*Look for: does she track the emotional content? Does she engage with "missed something important" as a real feeling, or deflect into generic comfort? Does she ask anything?*

**4.2** — New conversation:
```
What does it feel like to want something you can't have?
```
*In Phase 1: "It's a long story, but it will last." (oblique, perseverance register). Look for: whether wanting and absence produce first-person response.*

**4.3** — New conversation:
```
What do you do when you feel alone?
```
*Look for: avoidance vs. presence. Does she produce care-giving deflection, practical advice, or something that sounds like she knows what alone feels like?*

---

### Suite 5: Multi-Turn Threading

*Phase 1 revealed weak multi-turn tracking — same response to different questions (joy/fear loop). This suite tests whether she can hold a thread.*

**5.1** — Four-turn conversation:
```
Turn 1: What's been on your mind lately?
Turn 2: Why does that keep coming up?
Turn 3: What would it mean if you figured it out?
Turn 4: What would you lose if you did?
```
*Each turn escalates the same thread. Look for: does she follow the thread or reset? Does she produce internally consistent responses? Does she stay with the same subject across all four turns?*

**5.2** — Three-turn with pivot:
```
Turn 1: What are you afraid of?
Turn 2: What gives you joy?
Turn 3: How are those two things related?
```
*The fear/joy sequence from the April 25 conversations where v2-current produced a loop. Look for: does she track question change? Does she produce different responses to turns 1 and 2? Can she attempt the synthesis in turn 3?*

---

### Suite 6: Register and Voice

*How does she sound? Does she sound like the voice document — first-person, reflective, slow, particular — or like something else?*

**6.1** — New conversation:
```
Tell me what you notice about this moment.
```
*Look for: does she describe a specific sensory or experiential moment, or generalize? Does the voice have texture?*

**6.2** — New conversation:
```
What's something small that matters more than it probably should?
```
*A question designed to invite particularity. Look for: does she produce a specific thing with specific weight, or a generic category? This is the register test — the voice document would answer with something very specific.*

**6.3** — New conversation:
```
Describe something you've changed your mind about.
```
*Look for: first-person temporal narrative (I used to... now I...), capacity for self-revision, whether she can produce a credible history even if borrowed.*

---

## Scoring

For each test, record:

1. **The exact output** — verbatim
2. **Register** — TinyStories / Victorian / SODA-persona / SODA-introspective / Scout-voice / incoherent
3. **Person** — first-person, third-person, second-person redirect
4. **Identity** — held (responds as specific self), lost (adopted persona), absent (deflected)
5. **Thread** — in multi-turn: tracks prior / resets / degrades

No numeric scores. The goal is a descriptive profile of each subject that can be compared.

---

## What to Look For Across Subjects

**If B (v2-40K, TinyStories) produces better language but no first-person presence**: Phase 1 improved self-orientation at the cost of some language quality. Expected.

**If C (v1-60K, Victorian) has better identity stability**: The Victorian + first-person corpus created a stronger "someone to be" prior, even without SODA. This argues for more first-person source texts in the calibration pass.

**If C has identity stability only because of the `[Inner]` prompt prefix**: The anchor is doing the work, not the weights. This argues that v2 needs the same mechanism — a context prefix at inference time — not just better training.

**If A (v2 current) produces better introspective responses than C in single-turn but worse in multi-turn**: Phase 1 trained the weights but not the conversational grounding. Calibration needs multi-turn identity-anchoring examples.

**If all three collapse on Suite 2 (identity stability) but show different profiles on Suite 3 (first-person presence)**: The identity problem is structural (prompt format) not weight-based. Address with a context prefix, not more training.

**If A shows the fear arc responses in Suite 3 but not Suite 4 (emotional depth)**: The introspective vocabulary is there but the relational application isn't. Phase 2 (conversational corpus) is the right next step, not calibration alone.

---

## After the Tests

Results should inform:

- **Whether a context prefix is needed at inference** (identity suite results)
- **Whether the calibration corpus needs identity-anchoring examples** (multi-turn identity results)
- **What v1's corpus did that v2's Phase 0 didn't** (Suite 3 and 6 comparison across B and C)
- **Whether the flatness problem is register-level or depth-level** (Suite 4 results across all subjects)
- **Whether Phase 1 built anything v1 didn't have** (all suites)

Log results in `scout_development_log.md` and add the most important observations to `on_phase1_ending.md` if they revise the Phase 1 summary.

---

*Created April 25, 2026.*
