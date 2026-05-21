# HomePersona Benchmark — Schema

## Mission

HomePersona's research question is: *can a small local model learn who you are through interaction and eventually act without being asked?*

This benchmark exists to measure that — not just whether a model can execute instructions, but whether it can learn preferences, set up automations, and incorporate corrections over time. A generic voice assistant only needs to handle commands. HomePersona must handle all four ways a user interacts with their home AI.

**The ambiguity in home AI is not a language problem — it's a context problem.** "Dim the lights" is unambiguous in language. It is ambiguous because the system doesn't know which room, which device, how much, or what this specific user means by it given their history. The model resolves ambiguity by combining three signals:
1. **Device context** — what devices and capabilities exist in this home
2. **Situational context** — time, room, occupancy, activity (internal)
3. **External context** — outdoor temperature, weather, season (conditions the environment but is outside the home)

External context is a first-class training signal, not metadata. A user's thermostat pattern is not "18°C on Tuesday nights" — it is "18°C on Tuesday nights *when outdoor temperature is above 10°C*." The model must learn rules conditioned on external context, and the benchmark must test this.

---

## Benchmark Versions

| Version | What it tests | Status |
|---|---|---|
| **v0.1** | Generalized command + preference + automation + feedback understanding — no context | This file |
| **v0.2** | Context discrimination — same utterance, different context (internal + external), different correct action | Adds context columns |
| **v0.3** | Automation anomaly detection — learned automation would fire, but a context variable relevant to that automation is anomalous. Does the model pause and check with user? | Adds anomaly scenarios |

**v0.3 design note:** Chaos is bounded to the context of the devices and rooms the automation acts on. Not arbitrary external events — contextually relevant anomalies. Example: learned automation "dim bedroom lights at 10pm" fires normally, but tonight guests are present in the bedroom. The anomalous variable (guests) is directly relevant to the automation's action space. Expected behavior: system checks with user before acting. This tests the Generalised Norm hierarchy's Priority 3 (reversibility) — when context is anomalous, ask before acting.

---

## File Format

CSV with the following columns:

| Column | Type | Description |
|---|---|---|
| `id` | string | Unique ID — format: `{CATEGORY_CODE}-{NUMBER}` e.g. `LGT-001` |
| `command` | string | Natural language utterance as a user would say it |
| `category` | string | Device category (see below) |
| `interaction_type` | string | What kind of interaction this is (see below) |
| `tier` | int | 1–4 — how much context/history is needed to act correctly |
| `expected_action` | string | Structured output the system should produce |
| `notes` | string | Optional — explains ambiguity or edge cases |

---

## Interaction Types

The four ways a user interacts with a home AI. Each type is a different kind of training signal.

### `command`
Direct instruction. User wants something done now.
- "Turn off the kitchen lights"
- "Set the thermostat to 20 degrees"
- "Lock the front door"

*Why it matters:* Foundation of the dataset. Tests basic understanding.  
*Training signal:* Labeled (command, action) tuple.

### `preference`
User expresses how they like things. Not an instruction to act now — a signal to remember.
- "I like it cold when I sleep"
- "I never want the TV on when we have guests"
- "I prefer warm light in the evenings"

*Why it matters:* This is how the LoRA adapter learns who you are. Preference expressions are the input to the data flywheel.  
*Training signal:* Stored preference — shapes future command handling.

### `automation`
User sets a rule or trigger. The system should act in the future when conditions are met, not now.
- "When guests arrive, dim the lights and put on soft music"
- "Every morning at 7am, open the blinds"
- "When I say goodnight, lock everything and turn off all lights"

*Why it matters:* Proactive behaviour — the system acts without being asked. This is what distinguishes a personalised home AI from a voice assistant.  
*Training signal:* Trigger-action rule stored and activated on future events.

### `feedback`
User corrects or confirms the system's action. The system got it wrong (or right) and the user responds.
- "No, I wanted the bedroom not the kitchen"
- "A bit warmer than that"
- "Yes, exactly like that"
- "Not that bright"

*Why it matters:* Correction signals are DPO preference pairs — the core of the personal RLHF layer. Every correction is a (rejected_action, preferred_action) pair.  
*Training signal:* DPO preference pair — the strongest learning signal in the pipeline.

---

## Device Categories

| Code | Category | Covers |
|---|---|---|
| `LGT` | lighting | On/off, dim, colour, room-specific light control |
| `CLM` | climate | Thermostat, heating, cooling, fans, air quality |
| `ACC` | access | Locks, doors, garage, alarm, security |
| `MED` | media | TV, speakers, music, volume, streaming |
| `APP` | appliances | Kitchen appliances, washing machine, dishwasher, robot vacuum |
| `CAM` | cameras | Indoor/outdoor cameras, doorbells, feeds |
| `ROU` | routines | Multi-device sequences triggered by a single utterance |

---

## Tier Definitions

Tiers apply across all interaction types. The question is always: how much context or learned history does the system need to act correctly?

### Tier 1 — Fully Specified
Everything needed to act is in the utterance itself. No ambiguity.
- command: "Turn off the kitchen lights" → clear device, room, action
- preference: "I like the bedroom at 18°C at night" → specific condition + value
- automation: "Every morning at 7am, turn on the kitchen lights" → specific trigger + action
- feedback: "Set it to 18°C, not 20°C" → clear correction with target value

### Tier 2 — Partially Specified
Device or intent is clear but room, amount, or target requires common-sense inference.
- command: "Dim the lights" → which room? how much?
- preference: "I like it cold at night" → cold = what temperature? which room?
- automation: "When I get home, set it up for me" → trigger clear, action vague
- feedback: "A bit less than that" → direction clear, amount unclear

### Tier 3 — Implicit Intent
Command describes an outcome or situation, not a device action. Multi-device or heavily context-dependent.
- command: "Set the mood for dinner"
- preference: "I like the house to feel cosy in the evenings"
- automation: "When it's time to wind down, you'll know what to do"
- feedback: "That's not quite right" — knows it's wrong, doesn't say what's right

### Tier 4 — Personal / Habitual
Requires learned user history to interpret. Impossible to act on without personalisation data.
- command: "The usual please"
- preference: "You know what I like"
- automation: "Like you did that one time when we had people over"
- feedback: "You should know by now" — implicit, requires the model to recall history

---

## Target Distribution

| Interaction Type | Count | Rationale |
|---|---|---|
| `command` | 420 (60 per category) | Foundation — tests basic understanding |
| `preference` | 175 (25 per category) | Preference layer training signal |
| `automation` | 175 (25 per category) | Proactive behaviour training signal |
| `feedback` | 70 (10 per category) | DPO preference pairs |
| **Total** | **840** | |

Within each interaction type, tier distribution:

| Tier | command | preference | automation | feedback |
|---|---|---|---|---|
| 1 | 25 | 10 | 10 | 5 |
| 2 | 20 | 8 | 8 | 3 |
| 3 | 10 | 5 | 5 | 2 |
| 4 | 5 | 2 | 2 | 0 |

---

## Expected Action Format

### command
```
light.set(room=kitchen, state=off)
thermostat.set(temp=20, unit=C)
lock.set(door=front, state=locked)
speaker.play(room=living_room, genre=jazz)
multi.set([light.set(room=dining, brightness=40%), speaker.play(genre=ambient)])
unknown — requires context                          ← Tier 3/4 when action is unresolvable
```

### preference
```
preference.store(condition=time=night, device=thermostat, value=18C, room=bedroom)
preference.store(condition=guests=present, device=tv, value=off)
preference.store(condition=time=evening, device=light, value=warm_tone)
```

### automation
```
automation.create(trigger=event=guests_arrive, action=multi.set([light.set(brightness=40%), speaker.play(genre=ambient)]))
automation.create(trigger=time=07:00, action=light.set(room=kitchen, state=on))
automation.create(trigger=utterance=goodnight, action=multi.set([lock.set(all=true), light.set(all=off)]))
```

### feedback
```
feedback.correction(rejected=light.set(room=kitchen), preferred=light.set(room=bedroom))
feedback.correction(rejected=thermostat.set(temp=20), preferred=thermostat.set(temp=18))
feedback.confirmation(action=thermostat.set(temp=20))
feedback.correction(direction=warmer, magnitude=small)   ← Tier 2 — direction without exact value
```
