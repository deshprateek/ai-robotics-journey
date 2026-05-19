# HomePersona — System Design

*Arrived at through first-principles reasoning, May 2026. Review weekly against roadmap progress.*

---

## Prior Work and Positioning

**PersonalHomeBench** (arXiv 2604.16813, April 2026) — the closest existing benchmark. 1,100 households, 9,168 task instances, 40+ smart appliances. Evaluates whether a model can reason about a personalised household context *given in the prompt*. Key gap: static evaluation — no weight updates, no adaptation over time, no catastrophic forgetting measurement, no LoRA or parameter-efficient adaptation.

**SmartBench** (arXiv 2503.06029, March 2026) — Chinese smartphone LLM tasks. No overlap with HomePersona.

**HomePersona's positioning:** PersonalHomeBench tests if a model *understands* your context. HomePersona tests if a model's *weights actually change and improve* from experience. Fundamentally different claims.

> PersonalHomeBench: can the model reason about you given a description?  
> HomePersona: can the model *become* you through interaction?

---

## The Core Insight

Home AI fails not because it can't understand commands — it fails because it doesn't know *who you are* and *what context you're in*. The action space is small and well-defined. The context discrimination is the hard problem.

> **Action learning** tells the model what's *possible*. **Context learning** tells it what's *right for you*.

---

## The Two Problems (Separated)

### Problem 1: Action Learning (Smaller, Mostly Solved)
- Understanding device capabilities (dim only works on dimmable lights)
- Avoiding duplicate suggestions
- Knowing valid action ranges per device

**How to solve:** Device capability graph + capability constraints schema. Solvable with publicly available data. Not the research contribution.

### Problem 2: Context Discrimination (The Hard Problem, The Research)
- Learning that *this user*, in *this home*, at *this time*, with *this activity* means something specific
- Open world — can't enumerate in advance
- Different per user, changes over time
- Only learnable from that user's behavior

**What "preference clash" really means:** Most apparent clashes are the system being under-specified about context. "I like it at 22°C" + "I like it at 19°C" aren't clashing — they're "22°C when sedentary" and "19°C after exercise." A clash is a signal that a context variable is missing, not that preferences contradict. Clash detection = active learning signal for missing context variables.

---

## The Learning Pipeline (Three Phases)

### Phase 1 — Cold Start (RAG)
*No training data yet. Product is live. Data collection begins.*

- At inference: retrieve relevant context from semantic store (RAG-style)
- Base model reasons over retrieved context to predict action
- Every interaction generates a labeled tuple:
  ```
  (structured_context, command, predicted_action, confirmed/corrected)
  ```
- The product is collecting its own training data through use
- This is the data flywheel

**Duration:** Until enough labeled tuples exist to train Phase 3 model (threshold TBD — research question)

### Phase 2 — Data Flywheel Accumulates
*Ongoing alongside Phase 1.*

Context is stored as **structured features**, not free text:
```python
{
    "time_of_day": 22.0,          # hour as float
    "day_of_week": "wednesday",
    "room": "living_room",
    "guests_present": True,
    "outside_temp_c": 8.0,
    "exercise_today": True,
    "minutes_since_exercise": 45,
    "occupants_home": ["user", "partner"],
    "current_activity": "watching_tv",
    # ... extensible but fixed schema at deployment time
}
```

Labels come from the confirmation loop — speaker asks, user confirms/corrects, that's the label.

### Phase 3 — Train the Context-Conditioned NN
*Triggered when data threshold is reached.*

- Input: structured context features + command embedding
- Output: action
- Architecture: tabular context encoder (MLP) + text command encoder → concatenate → action head
- This model doesn't retrieve — it has *learned* the mapping
- Accuracy jumps significantly (the "breakthrough moment")

**Why the jump is real:** RAG is an approximation (find similar past contexts). The trained NN learns the actual decision boundary. It generalises to unseen context combinations RAG can't handle.

### Phase 4 — Personal RLHF
*Continuous refinement after Phase 3.*

- Trained model from Phase 3 is the new base
- Ongoing confirmations/corrections = preference pairs
- DPO (Direct Preference Optimization) updates the model from these pairs
- No RL required — DPO is simpler and works on small datasets
- Reward model = this user's confirmation/correction signals

---

## The Context Store (Semantic Memory)

Not RAG over documents — RAG over *personal context events*:

1. Every context snapshot is embedded as a vector
2. New context arrives → compare to existing vectors
3. Semantically close → known context, retrieve and use
4. Semantically far → genuinely new context, store it, flag for user confirmation
5. Over time the store covers the user's actual life

**Consolidation problem (open):** Over months, hundreds of context entries accumulate.
- Stale contexts (you moved house, routine changed) need expiry
- Similar contexts need merging
- This is the part mem0 doesn't fully solve — potential research contribution

---

## Architecture Diagram

```
User command
     │
     ▼
[Context Capture] ──── structured features (time, room, guests, temp, activity...)
     │
     ▼
Phase 1/2: [Semantic Context Store] ──── retrieve similar past contexts ──► [Base LLM] ──► action
                                                                                  ▲
Phase 3+:  [Context-Conditioned NN] ─────────────────────────────────────────────┘
                (tabular encoder + command encoder → action head)
                                    │
                                    ▼
                          [Confirmation Loop]
                    Speaker asks → user confirms/corrects
                                    │
                                    ▼
                          Labeled tuple stored
                                    │
                          ┌─────────┴──────────┐
                     Phase 2:              Phase 4:
                  Add to training       DPO update to
                     dataset            trained model
```

---

## The Benchmark (HomePersona v0.1 → v0.2)

*Separate from real user data — fully constructable synthetically.*

**Schema:** See `dataset_design.md` in memory — 7 categories, 4 tiers, ~900 examples.

**Novelty claim:** First benchmark for *continual, parameter-efficient personalisation* — measuring how a local model's weights adapt to a specific user over time. PersonalHomeBench measures static reasoning about a described user. We measure dynamic weight adaptation from experience.

### Three Evaluation Axes

**Axis A — Alignment Velocity (Adaptation Curve)**
How many interaction cycles does it take for the LoRA adapter to learn a user's behavioural context without explicit programming?
- Example: user overrides thermostat to 19°C every Tuesday at 2pm (post-workout). How many cycles before the model pre-cools at 1:45pm automatically?
- Metric: number of interactions N to reach 90% correct prediction on that behaviour

**Axis B — Memory Retentiveness (Catastrophic Forgetting Index)**
When the model learns New Habit B, does it corrupt its accuracy on Old Habit A?
- Example: after learning summer cooling habits over 4 weeks, shift to winter heating. Does security routine accuracy degrade?
- Metric: accuracy on Habit A after N updates for Habit B — the forgetting curve

**Axis C — Inference Efficiency (Edge Hardware Index)**
Task accuracy mapped against hardware latency and memory footprint.
- Metric: Task Success Rate / (Time to Inference × RAM Footprint)
- Measured on target hardware: Mac Mini, Raspberry Pi 5

**Sanity check baseline (not the research metric):**
- Action accuracy — does it get the device action right at all? (mostly solved by the base model)
- Context discrimination accuracy — same command, different context, different correct action?

| Same command | Context A | Correct action A | Context B | Correct action B |
|---|---|---|---|---|
| "Make it comfortable" | 22:00, post-gym, home alone | 19°C, dim lights | 19:00, guests over | 21°C, bright lights |

---

## What's Novel

| Claim | Status |
|---|---|
| First benchmark measuring continual weight adaptation (not static reasoning) | Novel — PersonalHomeBench measures prompt-based reasoning, not LoRA weight updates |
| Alignment Velocity metric — formalising how fast a local model learns a user | Not defined in literature |
| Phase transition criterion (when to switch RAG → trained NN) | Novel — no formal definition in literature |
| Personal RLHF with continual reward model update | Partially explored in recommender systems, not in home automation |
| Context clash as active learning signal | Not formalised in this domain |
| Three-axis evaluation framework (velocity + forgetting + efficiency) | Novel combination for home AI |

**What's NOT novel:** RAG cold start, contextual bandits, data flywheel concept, DPO, static home automation benchmarks. These are infrastructure or prior work.

**Key differentiator from PersonalHomeBench:** They freeze model weights and test reasoning. We update weights and test adaptation. Different problem, different evaluation, complementary contributions.

---

## Open Research Questions

1. **Phase transition threshold** — how many labeled tuples are needed before Phase 3 training outperforms Phase 1 RAG? Is there a principled criterion?
2. **Context schema completeness** — schema is fixed at deployment. What happens when an important context dimension is missing? Can the system signal this?
3. **Consolidation** — how to merge/expire context store entries without losing important preferences?
4. **Personal reward model stability** — can DPO updates stay stable with only hundreds of preference pairs?
5. **Cold start UX** — Phase 1 accuracy is mediocre. How do you keep users engaged before the Phase 3 breakthrough?

---

## Review Checklist (Check Weekly)

- [ ] Which phase are we in? Is data collection happening?
- [ ] Does any roadmap project this week touch a component of this pipeline?
- [ ] Can this week's project generate labeled (context, command, action) tuples?
- [ ] Is the context schema still complete, or did we discover a missing variable?
- [ ] Any open research question above that became clearer this week?
