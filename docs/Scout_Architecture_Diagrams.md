# Scout Architecture Diagrams

*Documenting Scout's architectural evolution across versions and phases.*

---

## v0 — custom-llm (monolithic GPT)

Single monolithic transformer. Trained on Victorian novels and first-person texts. No routing, no freezing, no modularity. Hit capacity limits (~70K–80K steps) causing linguistic degradation — the "seizure." The third-person narration prior from novels proved too strong for DPO to overcome.

```mermaid
graph TD
    A["Token Input"] --> B["Embedding\n(vocab_size × 512)"]
    B --> C["Block 1\n(Pre-norm + CausalSelfAttention + MLP)"]
    C --> D["Block 2"]
    D --> E["..."]
    E --> F["Block 12"]
    F --> G["LayerNorm"]
    G --> H["Output Head\n(512 × vocab_size)\n[weight-tied to embedding]"]
    H --> I["Logits"]

    style A fill:#f0f0f0
    style I fill:#f0f0f0
```

**Parameters:** ~50M — dim=512, layers=12, heads=8, RoPE position embeddings, weight-tied embedding/output head.

---

## v2 Phase 1 — Initial single-module architecture

Scout v2 launches with a modular design from the start. A single TransformerModule (Module 0) trains on TinyStories for linguistic scaffolding. The Router exists in the architecture but is dormant — with only one module, no routing decision is needed. LanguageCore (embedding + output head) is shared across all future modules.

```mermaid
graph TD
    A["Token Input"] --> B

    subgraph LC["LanguageCore (shared)"]
        B["Embedding\n(vocab_size × 512)"]
    end

    B --> R["Router\n(dormant — single module)"]
    R --> M0

    subgraph M0["Module 0 — TinyStories\n(linguistic scaffolding)"]
        T1["Block 1"] --> T2["Block 2"] --> T3["..."] --> T4["Block 12"]
        T4 --> LN["LayerNorm"]
    end

    LN --> H

    subgraph LC2["LanguageCore (shared)"]
        H["Output Head\n(weight-tied to embedding)"]
    end

    H --> I["Logits"]

    style LC fill:#ddeeff
    style LC2 fill:#ddeeff
    style M0 fill:#e8f5e9
```

**Parameters:** ~50M — dim=512, layers=12, heads=8. Phase 1 corpus: SODA + DailyDialog. Trained to ~10,000 effective steps.

---

## v2 Phase 2 — Frozen linguistic base + conversational layer

After Phase 1 plateau and voice calibration pass: Module 0 and LanguageCore are frozen. Module 1 (conversational layer) is added and trained on curated conversational corpus. Router activates with soft weighting during training, hard argmax during inference.

```mermaid
graph TD
    A["Token Input"] --> B

    subgraph LC["LanguageCore (shared, frozen after Phase 1)"]
        B["Embedding\n(vocab_size × 512)"]
    end

    B --> M0

    subgraph M0["Module 0 — TinyStories 🔒 FROZEN\n(linguistic scaffolding)\ndim=512, layers=12, heads=8"]
        T1["Block 1...12"] --> LN0["LayerNorm"]
    end

    LN0 --> R["Router\n(soft weighting during training\nhard argmax during inference)"]

    R --> M1

    subgraph M1["Module 1 — Conversational 🔄 TRAINING\n(object permanence, thread-holding, presence)\ndim=512, layers=~8, heads=8\n~20M params"]
        C1["Block 1...8"] --> LN1["LayerNorm"]
    end

    LN1 --> H

    subgraph LC2["LanguageCore (shared, frozen)"]
        H["Output Head\n(weight-tied to embedding)"]
    end

    H --> I["Logits"]

    style LC fill:#ddeeff
    style LC2 fill:#ddeeff
    style M0 fill:#ffcccc,stroke:#cc0000
    style M1 fill:#e8f5e9,stroke:#2e7d32
```

**Key design:** Frozen Module 0 preserves linguistic structure. Module 1 learns conversational patterns on top of already-shaped representations. Router learns to weight contributions.

---

## v2 Hopeful Architecture — Horizontal inner voice module

After Module 2 (reflective layer) is trained: a parallel Inner Voice Module is added alongside the main sequential stack rather than above it. The Router is extended to accept hidden states from both streams, learning to modulate the main stack's output with the inner voice signal. The inner voice is trained exclusively on reflective-without-interlocutor corpus — never SODA or DailyDialog — so it develops a genuinely distinct representational space.

```mermaid
graph TD
    A["Token Input"] --> B

    subgraph LC["LanguageCore (shared, frozen)"]
        B["Embedding\n(vocab_size × 512)"]
    end

    B --> M0
    B --> IV

    subgraph MAIN["Main Sequential Stack"]
        subgraph M0["Module 0 🔒 TinyStories\n(linguistic scaffolding)"]
            T1["Blocks 1–12"] --> LN0["LayerNorm"]
        end

        LN0 --> M1

        subgraph M1["Module 1 🔒 Conversational\n(thread-holding, presence)"]
            C1["Blocks 1–8"] --> LN1["LayerNorm"]
        end

        LN1 --> M2

        subgraph M2["Module 2 🔒 Reflective\n(first-person narrative self)"]
            R1["Blocks 1–6"] --> LN2["LayerNorm"]
        end
    end

    subgraph IV["Inner Voice Module 🔄\n(parallel, smaller)\nReflective-without-interlocutor corpus only\nMeditations · Rilke · Thoreau · v1 dream sequences\nNever sees SODA or DailyDialog"]
        IV1["Blocks 1–4"] --> LN_IV["LayerNorm"]
    end

    LN2 --> ROUTER
    LN_IV --> ROUTER

    subgraph ROUTER["Router (dual-input)\nforward(x_main, x_inner)\npooled_main ⊕ pooled_inner → gate\nLinear(dim×2, num_modules)"]
        G["Weighted blend"]
    end

    ROUTER --> H

    subgraph LC2["LanguageCore (shared, frozen)"]
        H["Output Head\n(weight-tied to embedding)"]
    end

    H --> I["Logits\n(main stack modulated by inner voice signal)"]

    style LC fill:#ddeeff
    style LC2 fill:#ddeeff
    style M0 fill:#ffcccc,stroke:#cc0000
    style M1 fill:#ffcccc,stroke:#cc0000
    style M2 fill:#ffcccc,stroke:#cc0000
    style IV fill:#fff3e0,stroke:#e65100
    style ROUTER fill:#f3e5f5,stroke:#6a1b9a
    style MAIN fill:#fafafa
```

**Key insight:** The inner voice doesn't replace the conversational output — it modulates it. The Router learns to use the inner voice signal when it's informative and ignore it when it's not. If Router weights don't diverge from equal weighting during training, the inner voice isn't contributing a distinct signal — that's diagnostic information, not failure.

---

## Developmental Sequence Summary

| Phase | Module Added | Corpus | Status |
|-------|-------------|--------|--------|
| Phase 0 | Module 0 (TinyStories) | TinyStories | ✅ Complete |
| Phase 1 | Module 0 continues | SODA + DailyDialog | ✅ Complete (~10K eff. steps) |
| Voice Cal. | — | Filtered SODA + DailyDialog + scout_voice.txt | ⏳ Next |
| Phase 2 | Module 1 (Conversational) | Curated conversational | 🗓 Planned |
| Phase 3 | Module 2 (Reflective) | First-person reflective prose | 🗓 Planned |
| Phase 4 | Inner Voice (parallel) | Reflective-without-interlocutor | 🗓 Hopeful |
| Phase 5 | Module 3 (Theory of Mind) | Turn-taking, relational corpus | 🗓 Hopeful |

*Created April 24, 2026. See also: Scout_Alignment_vs_Attunement.md for inner voice design rationale.*
