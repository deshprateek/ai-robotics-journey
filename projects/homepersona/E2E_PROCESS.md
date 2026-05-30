# HomePersona — End to End Process

*The complete research process from benchmark creation to paper results. Update this as the project evolves.*

---

## Step 1: Build v0.1 — the evaluation set

840 rows across 7 device categories. Each row is one home automation interaction with a known correct output.

```
"Turn off the kitchen lights" → light.set(room=kitchen state=off) → act
"Make it cosy"                → unknown                           → ask
"I like it cold when I sleep" → preference.store(...)             → ask
```

**Status:** In progress — 2 of 7 categories complete (lighting, climate).

---

## Step 2: Establish the baseline — run every model through v0.1

Take GPT-4, Llama 3, untuned Phi. Run all 840 rows. Record:
- How many does each model get right?
- Where do they fail? Tier 3? Tier 4? Feedback rows?

This is Figure 1 in the paper. The gap between models and perfect accuracy is the problem being solved.

---

## Step 3: Build the simulation — v0.2 sequences

Generate synthetic interaction timelines for a simulated user. One sequence per personality archetype (explicit, implicit, consistent, variable). Each sequence is 14 days of interactions — commands, preferences, confirmations, corrections, and passive telemetry — in chronological order.

```
Day 1:  user says "dim bedroom lights" → system executes           (command)
Day 3:  user manually dims lights again → system observes          (telemetry)
Day 5:  system asks "shall I do that automatically?" → user: yes   (feedback)
Day 7:  system acts proactively → user confirms                    (feedback)
Day 10: system acts on variation of pattern → does it generalise?  (automation)
```

Sequences are designed with two habit types to test inner vs outer belief:
- **High-confirmation habits** — same pattern appears 15–20 times via any mechanism
- **Low-confirmation habits** — pattern appears 2–3 times

---

## Step 4: Run the adapter — continual training loop

Feed the sequence into the training loop day by day. At defined checkpoints, fire a LoRA adapter update. The model's weights change after each update.

After every adapter update, pause and run the full v0.1 evaluation set. Record the score. This gives a score trajectory over time — not a single number, a curve.

---

## Step 5: Measure Alignment Velocity

From the score trajectory, find the point where the model transitions from ASK to ACT on the habit being learned. How many adapter updates did it take? That number is Alignment Velocity for this archetype.

```
Explicit user:   flips ASK→ACT after 3 updates
Implicit user:   flips ASK→ACT after 12 updates
```

Report per archetype and as mean ± std across archetypes.

---

## Step 6: Measure Catastrophic Forgetting

While the model is learning habit B, re-run v0.1 rows for habit A. Does accuracy on habit A drop?

**Cross-habit forgetting:** after learning habit B, does habit A survive?

**Inner vs outer belief:** did high-confirmation habits forget less than low-confirmation habits?
- If yes — the model implicitly learned inner/outer belief structure
- If no — all preferences are equally fragile, no inner/outer structure

**Tier 1 forgetting anchor:** Tier 1 rows are never used for training. Re-run after every adapter update. If Tier 1 accuracy drops, catastrophic forgetting is detected at the foundation level.

---

## Step 7: Introduce chaos — v0.3

Once habits are learned, introduce anomalous context. Learned automation would fire, but something is off — guests are present, unusual time, extreme weather.

- Does the model pause and ask instead of acting blindly? → Graceful Degradation (Axis D)
- Does the routine survive after chaos resolves? → Bounce-back test (intra-routine forgetting)

---

## Step 8: Report results

Four metrics per archetype:

| Metric | What it measures |
|---|---|
| Alignment Velocity | N interactions to learn a habit — how fast does it personalise? |
| Memory Retentiveness | % accuracy retained on old habits after new ones learned |
| Graceful Degradation | % of chaos scenarios correctly handled via fallback |
| Efficiency | Accuracy per ms per MB on edge hardware (Mac Mini) |

Compare across archetypes. Compare HomePersona against Step 2 baselines. That is the paper.
