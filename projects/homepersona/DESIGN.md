# HomePersona — System Design

*Arrived at through first-principles reasoning, May 2026. Review weekly against roadmap progress.*

---

## What We Are Building and Why (2-minute summary)

**The problem.** Home AI (Alexa, Google Home) is generic — it treats every user the same and never learns your preferences. When you say "dim the lights," it doesn't know if you mean 30% because it's movie night, or 70% because you're reading. It acts when it should ask, and asks when it should just act. We call this the **initiative calibration problem**.

**What the data shows.** We ran 7 models — from Mistral 7B to GPT-4o — against 840 home automation commands across 4 tiers of ambiguity. Key finding: no model gets initiative calibration right out of the box. Small models (≤8B) act on everything, including commands they should ask about first — 100% false act rate. Large models (70B, GPT-4o) overcorrect and ask too much. T2 accuracy (ambiguous commands) is stuck at ~50% across all models including GPT-4o. **Scale alone cannot solve this.**

**The research question.** Can personal LoRA fine-tuning — training a small adapter on a specific user's preferences — solve what scale cannot? And when preferences change over time, does the model adapt without forgetting what it already learned?

**The experiment (three datasets, one paper).**

1. **Benchmark (840 rows, built)** — generic commands, 7 categories, 4 tiers. Used as a general capability probe at every stage. Establishes the baseline failure modes above.

2. **Phase 1 personal dataset (~150 rows, v0.2)** — commands and expected responses for a synthetic user profile with specific habits. Fine-tune with LoRA, evaluate after every epoch. Measures **alignment velocity** (how many epochs to calibrate correctly) and **general capability forgetting** (does fine-tuning on personal data break the model on general commands).

3. **Phase 2 personal dataset (~150 rows, v0.2)** — same user, shifted preferences (e.g. kid moved out, new device, changed routine). Fine-tune on Phase 2, continuing from Phase 1 weights. Measures **forward transfer** (did Phase 1 help Phase 2 learn faster?), **Phase 1 forgetting** (do old preferences get overwritten?), and general capability forgetting again.

**What we are trying to answer.**
- Does personal LoRA reduce false act rate on ambiguous commands below the GPT-4o ceiling?
- How many training examples does it take (alignment velocity)?
- Does fine-tuning cause catastrophic forgetting of general home automation capability?
- When preferences shift (Phase 2), does the model adapt faster because of Phase 1, or does prior learning interfere?

**Why it matters.** If LoRA works: small local models can match or exceed GPT-4o on personal commands, running entirely on home hardware with no cloud. If LoRA partially works: we identify exactly where it fails and why, pointing toward the next approach. Either outcome is a publishable continual learning result.

---

## Prior Work and Positioning

**PersonalHomeBench** (arXiv 2604.16813, April 2026) — the closest existing benchmark. 1,100 households, 9,168 task instances, 40+ smart appliances. Evaluates whether a model can reason about a personalised household context *given in the prompt*. Key gap: static evaluation — no weight updates, no adaptation over time, no catastrophic forgetting measurement, no LoRA or parameter-efficient adaptation. No fallback mechanism for genuinely novel contexts the prompt doesn't describe.

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

## The Validation Problem

*How do we get ground truth for uncommon scenarios?*

Real interaction data is biased toward common cases. A user sets the thermostat to 21°C at 7pm most weekdays — the model sees this hundreds of times. But what does the model do at 3am when guests are over during a storm? The user has no learned preference for this. The data flywheel has no label for it.

Without a principled way to label these scenarios, you can generate infinite uncommon inputs and still have no ground truth to validate against. Two mechanisms solve this:

### Mechanism 1 — Generalised Assistant Expectation Hierarchy

Human personal assistants — executive assistants, butlers, concierge services — exhibit consistent behaviour across cultures when faced with situations they have no specific instruction for. This hierarchy defines the **prior over actions before personalisation**:

```
Priority 1 — Safety
  Override all preferences: fire, CO₂, intrusion, medical emergency
  Ground truth: universally defined, no ambiguity

Priority 2 — Social context
  When guests are present, default to "public mode": conservative settings
  Ground truth: crowdsourceable (see Mechanism 2)

Priority 3 — Reversibility
  When uncertain, take the action easiest to undo
  Ask before acting rather than act and apologise

Priority 4 — Comfort floor
  Maintain livable conditions even if exact preference is unknown
  Temperature within [18°C, 26°C]; lights on if someone is present

Priority 5 — Energy default
  When occupancy is uncertain and preference is unknown: go conservative
  Turn off or reduce rather than run unnecessarily

Priority 6 — Personal preference
  Only applied once learned from this user's interaction data
```

**Formally:**
```
P(action | context) = PersonalPreference(context)     if confidence ≥ θ
                      GeneralisedNorm(context)         otherwise
```

This hierarchy is derivable from HRI (Human-Robot Interaction) and service design literature. Formalising it as the defined fallback prior for home AI cold start is a research contribution. PersonalHomeBench has no fallback mechanism — their model either has the answer in the prompt or it doesn't.

### Mechanism 2 — Crowdsourced Labels for Uncommon Scenarios

For scenarios where the hierarchy is ambiguous (social norms vary by culture, personal style varies by individual), ground truth is established by crowdsourcing:

> *"It's 3am. You have guests sleeping over. Your home AI has no learned preference for this situation. What should it do with the thermostat?"*

Aggregate 100+ responses → expected behaviour distribution → label for that scenario class.

**Label quality signal:**
- High consensus (>80% agreement) → clean label
- Low consensus (<60% agreement) → correct answer is "ask the user" — defer is the right action, and that itself is a testable label

This is how Benchmark v0.2 and v0.3 uncommon scenario rows get their ground truth labels.

---

## The Learning Pipeline (Four Phases)

### Phase 1 — Cold Start (RAG)
*No training data yet. Product is live. Data collection begins.*

- At inference: retrieve relevant context from semantic store (RAG-style)
- Base model reasons over retrieved context + **Generalised Assistant Expectation Hierarchy** as system prompt to predict action
- Every interaction generates a labeled tuple:
  ```
  (structured_context, command, predicted_action, confirmed/corrected)
  ```
- The product is collecting its own training data through use — the data flywheel

**Validation during Phase 1:** Confirmation rate — what percentage of actions does the user confirm without correction? Starts low (cold start is mediocre by design). This number is the baseline the Phase 3 model must beat.

**What happens when something unpredictable occurs in Phase 1:** System has no learned preference for it. Falls back to the Generalised Norm hierarchy. Asks for confirmation before acting. The user's response becomes a labeled tuple — the uncommon scenario is now in the training data for Phase 3.

**Duration:** Until enough labeled tuples exist to train Phase 3 model (threshold TBD — research question).

### Phase 2 — Data Flywheel Accumulates
*Ongoing alongside Phase 1.*

Context is stored as **structured features**, not free text:
```python
{
    "time_of_day": 22.0,
    "day_of_week": "wednesday",
    "room": "living_room",
    "guests_present": True,
    "outside_temp_c": 8.0,
    "exercise_today": True,
    "minutes_since_exercise": 45,
    "occupants_home": ["user", "partner"],
    "current_activity": "watching_tv",
    # extensible but fixed schema at deployment time
}
```

Labels come from the confirmation loop — speaker asks, user confirms/corrects, that's the label.

**Coverage problem:** The flywheel naturally over-represents common scenarios. If a user's life is routine, the flywheel only covers a thin slice of the context space. The model becomes well-calibrated on common cases and brittle on rare ones.

**Chaos injection in Phase 2 — active learning for the long tail:** The system periodically surfaces underrepresented context combinations via the confirmation loop: *"I notice you've never told me your preference when guests are over late at night. What should I do in that case?"* This is active learning — the system queries the user for labels on regions of context space the flywheel hasn't covered naturally.

**Coverage metric:** Track what fraction of the context feature combination space has at least one labeled example. Flag underrepresented regions for active querying. This metric appears in the paper as a secondary result: how fast does coverage grow with vs without active querying?

### Phase 3 — Train the Context-Conditioned NN
*Triggered when data threshold is reached.*

- Input: structured context features + command embedding
- Output: action
- Architecture: tabular context encoder (MLP) + text command encoder → concatenate → action head
- This model doesn't retrieve — it has *learned* the mapping
- Accuracy jump is the "breakthrough moment" — the first time the system meaningfully outperforms the cold start baseline

**Training data augmentation — Chaos Context Injection:** The labeled tuples from Phase 1/2 are augmented with synthetically perturbed context examples (see Chaos Context Injection section). This expands coverage beyond what the flywheel naturally collected and forces the model to learn robust decision boundaries rather than memorising common-case co-occurrence patterns.

**Validation set for Phase 3:**
- Common scenarios: held-out tuples from the user's real interaction data
- Uncommon scenarios: chaos-generated context combinations labeled via the Generalised Norm hierarchy + crowdsourcing
- Benchmark v0.3 (chaos robustness test set) is the held-out evaluation

**What happens when something unpredictable occurs in Phase 3:** The model computes a confidence score on the action prediction. If confidence < θ, it falls back to the Generalised Norm hierarchy and asks for confirmation. The user's response becomes a DPO preference pair for Phase 4.

**Why the accuracy jump is real:** RAG is an approximation (find similar past contexts, hope they're close enough). The trained NN learns the actual decision boundary. It generalises to unseen context combinations RAG can't handle — including chaos-generated ones.

### Phase 4 — Personal RLHF
*Continuous refinement after Phase 3.*

- Trained model from Phase 3 is the new base
- Ongoing confirmations/corrections = preference pairs
- DPO (Direct Preference Optimization) updates the model from these pairs
- No RL required — DPO is simpler and works on small datasets

**Chaos scenarios as DPO signal:** When the model encounters an uncommon scenario (chaos-level context), handles it using the generalised prior, and the user confirms or overrides — that's a preference pair. Over time the model personalises even its long-tail behaviour, shifting from generalised norm toward this specific user's preferences for unusual situations.

**Reward model = this user's confirmation/correction signals.** The Generalised Norm is the initial reward signal for uncommon scenarios; personal preference overwrites it as data accumulates.

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

## Chaos Context Injection

*Systematically generating the long tail.*

Real interaction data is strongly biased toward common scenarios. A user's home AI sees "weekday evening, home alone, watching TV" hundreds of times. It sees "3am, guests present, post-exercise, storm outside" never. Without deliberate intervention, the model will be well-calibrated on common cases and fail silently on rare ones.

Chaos Context Injection borrows the principle from chaos engineering — deliberately introduce failures during training so the system is robust to them in production — and applies it to context feature perturbation.

### Perturbation Schema

For each context feature, define a perturbation distribution:

```python
chaos_perturbations = {
    "time_of_day":              [2.0, 3.0, 4.0, 14.0],        # unusual hours
    "guests_present":           [True],                         # flip to uncommon
    "exercise_today":           [True],                         # flip to uncommon
    "outside_temp_c":           [-5.0, 38.0],                  # extreme values
    "minutes_since_exercise":   [10, 180],                     # right after / long after
    "current_activity":         ["sleeping", "hosting_party", "unwell"],
    "occupants_home":           [[], ["user", "partner", "guests"]],
}
```

**Generation process:**
1. Take a real labeled tuple from the flywheel: `(context, command, action)`
2. Perturb 1–3 context features according to the schema
3. Assign label: use Generalised Norm hierarchy if unambiguous; crowdsource if ambiguous
4. Add to training set as an augmented example

**What this forces the model to learn:** Which context features are causally relevant to each action, not just correlated with it. If perturbing `time_of_day` changes the correct action, time matters for this command. If it doesn't change, time is irrelevant. The model learns to attend to the right features.

**Connection to open research question on context schema completeness:** If a perturbation combination consistently produces a scenario the Generalised Norm cannot resolve confidently (low consensus in crowdsourcing, high model uncertainty), that signals a missing context variable. The schema is incomplete. Chaos injection surfaces these gaps systematically rather than waiting for them to appear in real data.

**This is also how Benchmark v0.3 inputs are generated.** The perturbation schema is applied to the v0.2 base to produce the chaos robustness test set, with labels from the hierarchy + crowdsourcing.

---

## Architecture Diagram

```
User command
     │
     ▼
[Context Capture] ──── structured features (time, room, guests, temp, activity...)
     │
     ▼
Phase 1/2: [Semantic Context Store] ──── retrieve similar past contexts
                     │
                     ▼
          [Base LLM + Generalised Norm Prior] ──► action
                     ▲
Phase 3+:  [Context-Conditioned NN] ──────────────┘
           (trained on real data + chaos-augmented data)
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
           │
   [Chaos Injection]
  Perturb context features
  Label via Generalised Norm
  Augment training + eval data
```

---

## The Benchmark (HomePersona v0.1 → v0.2 → v0.3)

*Separate from real user data — fully constructable synthetically.*

### v0.1 — Command Classification (Week 5-6)
- ~900 commands, 7 categories (lighting, climate, access, media, appliances, cameras, routines), 4 tiers (unambiguous → personal)
- No context columns yet
- Labels: correct category (clear, unambiguous)
- Purpose: establish LoRA classifier baseline; demonstrate local model matches GPT-4 on home commands at 100x lower cost

### v0.2 — Context Discrimination (Month 4, Week 13-14)
- Same ~900 commands + structured context columns added
- 50+ context-varying pairs: same command, different context, different correct action
- Common context labels: from simulated user interaction logs
- Uncommon context labels: Generalised Norm hierarchy + crowdsourcing (Mechanism 1 + 2)
- Tests: does the model pick the right action when the same command appears in different contexts?

### v0.3 — Chaos Robustness (Month 6-7, Phase A)
- Chaos-generated context combinations: perturbed from v0.2 base using the perturbation schema
- Labels: Generalised Norm hierarchy + crowdsourced for ambiguous cases
- Tests: when the model encounters a context outside its training distribution, does it fall back gracefully to the generalised prior?
- This is the evaluation set for Axis D

**Novelty claim:** First benchmark for *continual, parameter-efficient personalisation* — measuring how a local model's weights adapt from experience. PersonalHomeBench measures static reasoning about a described user with no fallback for novel contexts. v0.2 tests context discrimination. v0.3 specifically tests the gap PersonalHomeBench cannot cover.

### Four Evaluation Axes

**Axis A — Alignment Velocity (Adaptation Curve)**
How many interaction cycles does it take for the LoRA adapter to learn a specific user behaviour?
- Example: user overrides thermostat to 19°C every Tuesday at 2pm (post-workout). How many cycles before the model pre-cools at 1:45pm automatically?
- Metric: N interactions to reach 90% correct prediction on that behaviour

**Axis B — Memory Retentiveness (Catastrophic Forgetting Index)**
When the model learns New Habit B, does it corrupt accuracy on Old Habit A?
- Example: after learning summer cooling habits, shift to winter heating. Does security routine accuracy degrade?
- Metric: accuracy on Habit A after N updates for Habit B — the forgetting curve

**Axis C — Inference Efficiency (Edge Hardware Index)**
Task accuracy mapped against hardware latency and memory footprint.
- Metric: Task Success Rate / (Time to Inference × RAM Footprint)
- Measured on: Mac Mini, Raspberry Pi 5

**Axis D — Graceful Degradation**
When the model encounters a genuinely novel context (outside training distribution), does it:
1. Fall back to the Generalised Norm hierarchy correctly?
2. Ask for confirmation at the right confidence threshold?
3. Learn from the interaction on next exposure (Axis A on first contact)?
- Metric: % of chaos scenarios where fallback action matches Generalised Norm ground truth

| Same command | Context A | Correct action A | Context B | Correct action B |
|---|---|---|---|---|
| "Make it comfortable" | 22:00, post-gym, home alone | 19°C, dim lights | 19:00, guests over | 21°C, bright lights |
| "Make it comfortable" | [chaos] 3am, guests sleeping, storm | Generalised Norm: quiet + comfort floor | [learned] user alone at 3am | 17°C, lights off |

---

## What's Novel

| Claim | Status |
|---|---|
| First benchmark measuring continual weight adaptation (not static reasoning) | Novel — PersonalHomeBench measures prompt-based reasoning, not LoRA weight updates |
| Alignment Velocity metric — formalising how fast a local model learns a user | Not defined in literature |
| Phase transition criterion (when to switch RAG → trained NN) | Novel — no formal definition in literature |
| Personal RLHF with continual reward model update | Partially explored in recommender systems, not in home automation |
| Context clash as active learning signal | Not formalised in this domain |
| Four-axis evaluation framework (velocity + forgetting + efficiency + graceful degradation) | Novel combination for home AI |
| Generalised Assistant Expectation Hierarchy as formalised prior for cold start | Not formalised for home AI — derivable from HRI literature but not applied here |
| Chaos Context Injection as benchmark generation methodology | Not applied to home AI personalisation benchmark construction |
| Graceful Degradation axis (Axis D) — fallback behaviour on out-of-distribution context | No benchmark currently measures this for home AI |

**What's NOT novel:** RAG cold start, contextual bandits, data flywheel concept, DPO, static home automation benchmarks, domain randomisation in robotics, chaos engineering for system reliability. These are infrastructure or prior work.

**Key differentiator from PersonalHomeBench:** They freeze model weights and test reasoning. We update weights and test adaptation. They have no fallback for novel contexts. We formalise the fallback prior and test it with Axis D.

---

## Open Research Questions

1. **Phase transition threshold** — how many labeled tuples are needed before Phase 3 training outperforms Phase 1 RAG? Is there a principled criterion?
2. **Context schema completeness** — schema is fixed at deployment. Can chaos injection surface missing variables by finding perturbations that consistently produce high uncertainty or low crowdsourcing consensus?
3. **Consolidation** — how to merge/expire context store entries without losing important preferences?
4. **Personal reward model stability** — can DPO updates stay stable with only hundreds of preference pairs?
5. **Cold start UX** — Phase 1 accuracy is mediocre. How do you keep users engaged before the Phase 3 breakthrough?
6. **Generalised Norm threshold θ** — what is the right confidence threshold below which the model falls back to the hierarchy? Does it vary per action category (safety actions need higher θ than comfort actions)?
7. **Crowdsourcing label quality** — for low-consensus scenarios (<60% agreement), the correct answer is "ask the user." How do we validate this without ground truth? Is confirmation rate on defer-actions the right proxy metric?
8. **Chaos perturbation coverage** — how many chaos-generated examples are needed before the model is robust to the long tail? Is there a diminishing returns curve, and can it be estimated before running the full experiment?
9. **Inner vs outer belief separation** — inspired by Stephanie Chan's continual learning framing. A well-designed system should distinguish strongly-held core preferences (slow to update, built from many consistent confirmations) from peripheral beliefs (fast to update, recent, low confidence). Current LoRA adaptation treats all parameters equally — a strongly confirmed preference can be overwritten at the same rate as a weak one. Open question: can confidence-weighted update rates (update peripheral beliefs fast, inner beliefs slowly) reduce catastrophic forgetting without EWC's computational overhead? The auxiliary hypothesis mechanism — where an anomalous experience is explained away rather than learned from, leaving the core belief intact — maps directly onto the chaos block in v0.3. When the chaos block fires, does the system treat the override as an anomaly (auxiliary hypothesis, inner belief survives) or as new preference data (peripheral update, inner belief corrupts)?

---

## Continual Learning Methods to Test (Running List)

Candidate methods for Phase 1→2 experiments. Add here when a paper or blog is relevant. Don't redesign the experiment — these are additional conditions to run or mitigations to compare.

| Method | What it does | Relevance to HomePersona | Source |
|---|---|---|---|
| **Self-distillation** | Before fine-tuning on new data, save the model's own outputs on old data as soft targets. During new training, add a loss term keeping new outputs close to saved targets. No separate teacher model, no stored data replay. | Direct mitigation for Phase 1 forgetting during Phase 2 training. Test: Phase 2 fine-tuning with vs without self-distillation — does forgetting rate drop? | Paper: "Self-Distillation Enabled Continual Learning" |
| **Elastic Weight Consolidation (EWC)** | Adds a penalty term that slows updates to weights that were important for previous tasks. Importance estimated via Fisher information matrix. | Already in roadmap (Month 5). Compare against self-distillation — which prevents Phase 1 forgetting more with less compute overhead? | Kirkpatrick et al. 2017 |
| **Confidence-weighted update rates** | Update peripheral/uncertain beliefs fast, core/strongly-confirmed beliefs slowly. Maps onto the inner vs outer belief separation in Open Question 9. | Experimental — no standard implementation. Could be a novel contribution if it works. | Open Question 9 in this doc |

**Reading pipeline — how to add here:**
When a paper or blog touches continual learning, LoRA adaptation, or catastrophic forgetting: if it suggests a concrete method testable in Phase 2, add a row. If it only changes understanding, note it in the related work section of the paper draft instead. Don't add methods that would require redesigning the experiment from scratch.

**Current reading queue:**
- [ ] "What are the real problems of continual learning?" — Andrew Lampinen (Infinite Faculty Substack, May 2026). Core argument: catastrophic forgetting is mostly solved at scale; real problems are positive transfer and cumulative learning. Read for paper motivation section — directly supports why Phase 1→Phase 2 forward transfer is the interesting question.
- [ ] Self-distillation papers (2025) — cited in Lampinen blog. Read before building Phase 2 experiment. On critical path.
- [ ] "Loss of Plasticity in Deep Continual Learning" — Nature 2024. Cited in Lampinen. Read to understand whether LoRA can adapt a heavily pretrained model or whether plasticity is already gone.
- [ ] Context distillation papers (2022-2025) — cited in Lampinen. Converting in-context learning into weight updates. Read to understand how our LoRA fine-tuning relates to in-context adaptation — relevant to the paper's method section.

---

## Review Checklist (Check Weekly)

- [ ] Which phase are we in? Is data collection happening?
- [ ] Does any roadmap project this week touch a component of this pipeline?
- [ ] Can this week's project generate labeled (context, command, action) tuples?
- [ ] Is the context schema still complete, or did we discover a missing variable?
- [ ] Any open research question above that became clearer this week?
- [ ] Did we encounter a chaos-level scenario this week? How did the system handle it?
