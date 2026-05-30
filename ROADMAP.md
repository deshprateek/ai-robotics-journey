# ML + Robotics Research Roadmap
**Goal:** Research role at a robotics/AI company + 1 published paper by end of 2026  
**Time:** 1hr weekdays, 3hrs each weekend day (~11hrs/week)  
**Contributors:** Prateek Deshmukh + Samir Jibhakate — both Senior SWEs, learning ML + Robotics from scratch, equal contributors on all work  
**Niche:** Personalised on-device continual learning for home automation  
**Learning approach:** Build-first. For every reading item, attempt to build the concept from a one-sentence description before reading. Read to check and fill gaps, not as the starting point. Generate a day-by-day build-first session plan at the start of each new week.  
**GitHub:** Move repo to a shared GitHub organisation so both contributors own the codebase — do this before Week 5-6 when the real HomePersona work begins

---

## Quick Reference — Month by Month

| Month | Focus | Milestone |
|---|---|---|
| 1 | PyTorch, CNNs, backprop | Comfortable implementing models |
| 2 | Transformers, LoRA, HomePersona infrastructure | LoRA classifier + Home Assistant + ChromaDB live |
| 3 | First working pipeline + Benchmark v0.2 + MemGPT | End-to-end pipeline running, Day 0 baseline logged |
| 4 | Experiments Phase A + B | Baselines established, personalisation curve measured |
| 5 | Experiments Phase C + D | Catastrophic forgetting measured, ablation complete |
| 6 | Write + submit paper | arXiv preprint live |
| 7 | Job search | Interviews at Tier 1 companies |

---

## Daily Schedule

```
Mon (1hr): Read — 1 paper, book chapter, or course video
Tue (1hr): Implement — code from what you read Monday
Wed (1hr): Experiments — run, measure, record results in W&B
Thu (1hr): Write — paper draft, blog post, or research notes
Fri (1hr): Community — Twitter/X, Discord, read what others are doing

Sat (3hr): Deep project work — main coding or experiment session
Sun (3hr): Read 2-3 papers + plan the coming week
```

---

## HomePersona Build Arc

Every project from Week 5 onward directly contributes a component of HomePersona. By the end of Month 7 you will have a working system. The benchmark built in Week 5-6 is used to evaluate every experiment from Month 4 onwards.

```
Week 1-4:   Learn the tools — backprop, PyTorch, CNNs, training dynamics
Week 5-6:   Build the seed — Benchmark v0.1 + LoRA command classifier      ← HomePersona starts here
Week 7-8:   Build the infrastructure — Home Assistant + ChromaDB + CartPole
Week 9-10:  Connect the pieces — first end-to-end pipeline, Day 0 baseline
Week 11-12: Benchmark v0.2 + MemGPT integration + LoRA adapter layer
Week 13-16: Experiments Phase A + B — baselines + personalisation over time
Month 5:    Experiments Phase C + D — catastrophic forgetting + ablation
Month 6:    Write the paper
Month 7:    Job search
```

---

## Month 1 — ML Fluency (PyTorch + CNNs)

**Habits this month:**
- [ ] Show up every day — 30 minutes beats a 10hr weekend binge
- [ ] Commit to HomePersona as your niche — do not switch topics

### Week 1-2: Backprop + PyTorch
- [x] Watch: Karpathy "The spelled-out intro to neural networks and backpropagation" (YouTube)
- [x] Watch: Karpathy "The spelled-out intro to language modeling" (YouTube)
- [ ] Read: PyTorch "Learn the Basics" tutorial (pytorch.org)

- [x] **Project 1: Finish micrograd**
  - What: complete the Neuron, Layer, MLP classes and training loop you are building now
  - Goal: train the network on 4 examples until loss drops below 0.01
  - Success: you can explain every line of code including what backward() does

- [x] **Project 2: Rebuild MLP in PyTorch**
  - What: rewrite your micrograd MLP using `torch.nn.Module` instead
  - Goal: same training behavior — loss goes down the same way
  - Why: PyTorch is micrograd but faster and with GPU support. Seeing them side by side makes PyTorch click immediately
  - Success: both models produce similar loss curves on the same data

- [x] **Project 3: UCI Wine Dataset**
  - What: a classic dataset of 178 Italian wines, each described by 13 chemical measurements (alcohol content, acidity, color intensity etc.). The goal is to predict which of 3 vineyards (classes) a wine came from based on those 13 numbers
  - Why: your micrograd used toy data you made up. This is your first real dataset with messy, real-world numbers — a step closer to actual ML work
  - How to get it: `from sklearn.datasets import load_wine` — one line, no downloading
  - Goal: train your PyTorch MLP on it, get >90% accuracy on a held-out test set
  - Success: you understand train/test split, accuracy as a metric, and why real data is harder than toy data

### Week 3-4: CNNs + Training Dynamics
- [ ] Watch: Fast.ai Lesson 1-3 (fast.ai, free)
- [ ] Read: "A guide to convolution arithmetic" — Dumoulin & Visin (paper)
- [ ] Read: "A Recipe for Training Neural Networks" — Karpathy blog post

- [x] **Project 1: Real-time Floor Classification**
  - What: 4-class image classifier (Spill, Clean, Reflection, Obstacle) built with fast.ai and ResNet18
  - How: scraped images via DuckDuckGo search terms, trained with fine_tune(), evaluated with confusion matrix
  - Result: ~90% accuracy (84/93). Key confusion: Spill → Reflection (3 misclassifications) due to visual similarity in scraped images
  - Success: you understand transfer learning in practice — pretrained ResNet18 adapted to a domain-specific task with ~100 images per class

- [x] **Project 2: Deliberately overfit, then fix it**
  - What: 4 images per class (16 total), trained for 80 epochs on the spill dataset subset
  - Result: clear divergence — train loss dropped to ~0.13, valid loss rose to ~0.93. Sweet spot at epoch 26 (error_rate 0.0, valid_loss 0.179), then full overfit after
  - Fix: `aug_transforms(max_warp=0)` — valid loss trend reversed, stayed near 0.45 vs 0.93 without augmentation
  - Key learning: early stopping matters — the best model existed at epoch 26, continuing training destroyed generalisation. Augmentation works by preventing pixel memorisation.

- [x] **Project 3: The math of your CNN — why does it confuse Spill and Reflection?**
  - What: ran `interp.plot_top_losses()` on the spill classifier
  - Findings: (1) wet tile floors and reflective floors share the same visual pattern — bright patches on flat surface, indistinguishable at this camera angle; (2) one Clean image (sunlight through window) was mislabelled — model was right, label was wrong; (3) one anime image in Obstacle folder was bad scrape — corrupted the classifier
  - Fix applied: removed anime image, moved mislabelled Clean→Reflection. Spill→Reflection confusion dropped from 3→1
  - Key learning: plot_top_losses reveals data problems, not model problems. Cleaning data is more impactful than tuning the model.

---

## Month 2 — Transformers + RL

**Habits this month:**
- [ ] Show up every day
- [ ] Set up Weights & Biases account at wandb.ai — you will log every experiment from here onwards
- [ ] Follow on Twitter/X: Andrej Karpathy, Sergey Levine, Chelsea Finn, Pieter Abbeel
- [ ] Join Discord: Hugging Face, LeRobot, Papers with Code

### Week 5-6: Transformers + LoRA
- [ ] Read: "The Illustrated Transformer" — Jay Alammar blog post (read first)
- [ ] Read: "Attention is All You Need" — Vaswani et al. 2017 (the original paper)
- [ ] Watch: Karpathy "Let's build GPT" (YouTube)
- [ ] Read: "LoRA: Low-Rank Adaptation of Large Language Models" — Hu et al. 2022 (the paper behind personalised adapters)

- [ ] **Project 1: Build a tiny GPT from scratch**
  - What: follow Karpathy's "Let's build GPT" video and implement a character-level language model — a transformer that predicts the next character in a sequence
  - Why: transformers are the backbone of every modern AI system — GPT, BERT, RT-2 (robot model), all of them. Building one from scratch means you will never be confused by transformer architecture again
  - Train it on: a small text file — Shakespeare works, or download smart home logs
  - Goal: model generates coherent text after training — even if it's nonsense, the structure should look right
  - Success: you can explain what "attention" means in one sentence without hand-waving

- [ ] **Project 2: Smart home command classifier with LoRA + HomePersona Benchmark v0.1**
  - What: create a dataset of ~900 home automation commands across 7 categories (lighting, climate, access, media, appliances, cameras, routines) and 4 tiers (unambiguous → personal). Fine-tune Llama 3.2 3B using LoRA — a technique that adds a tiny set of trainable parameters (adapters) on top of a frozen base model, making personalised fine-tuning cheap enough to run on your laptop
  - Why LoRA specifically: full fine-tuning of a 3B parameter model requires expensive GPUs. LoRA adds only ~1M trainable parameters on top — it can run on a MacBook. This is the core mechanism behind your entire research direction. Understanding it now means every subsequent project builds on solid ground
  - Example inputs: "dim the bedroom lights" → lighting, "lock the front door" → access, "play jazz in the kitchen" → media
  - How: use `peft` library from Hugging Face (`pip install peft`), load Llama 3.2 3B via Ollama, apply LoRA config, fine-tune
  - Compare: LoRA fine-tuned small model vs GPT-4 API on the same commands — measure accuracy and latency
  - Goal: LoRA model matches GPT-4 accuracy on home commands at 100x lower cost and runs fully locally
  - **This dataset is also HomePersona Benchmark v0.1** — the same ~900 examples serve as the evaluation benchmark for all future HomePersona experiments. No public benchmark exists for personal home automation preference learning — creating this is a research contribution in its own right. See `projects/homepersona/DESIGN.md` for full schema and 4-tier structure
  - Success: (1) LoRA classifier works; (2) benchmark is versioned, documented, and reusable for all Month 5-6 experiments
  - Write first blog post when this project is complete — byline: Prateek Deshmukh and Samir Jibhakate: "We fine-tuned a 3B model on home automation commands and matched GPT-4 at 100x lower cost — and built the first benchmark to prove it." Post to Towards Data Science or your own blog
  - Email 1 researcher in the personalisation / small models space — share the blog post and the benchmark. This establishes your niche publicly from Month 2

### Week 7-8: HomePersona Infrastructure + CartPole
- [ ] Read: "MemGPT: Towards LLMs as Operating Systems" — Packer et al. 2023 (read now, you are building the memory store this week)
- [ ] Read: Sutton & Barto "Reinforcement Learning" Ch 1-2 (conceptual foundation only)

- [ ] **Project 1: CartPole with Q-learning** *(3 days)*
  - What: a pole balanced on a moving cart — the "hello world" of RL. Write a Q-table or DQN from scratch, no Stable-Baselines3
  - Why: every robotics researcher has done this. Teaches the core RL loop — observe state, pick action, get reward, update policy — which is exactly what DPO replaces in the HomePersona preference layer
  - Goal: agent keeps the pole balanced for 200+ timesteps consistently
  - Log in W&B: reward per episode, steps balanced over time
  - Success: you can explain what a reward function is and why reward design is the hard part

- [ ] **Project 2: Home Assistant + Benchmark Day 0 baseline**
  - What: Home Assistant is an open-source smart home platform that runs locally and connects to real or virtual devices. Set it up locally and connect 3 virtual devices: a dimmable light, a thermostat, and a smart lock
  - Task: run 50 commands from your Benchmark v0.1 (spread across lighting, climate, access categories) through the untuned Llama 3.2 3B. Record which commands it gets right — this is your Day 0 baseline before any personalisation
  - How: `pip install homeassistant`, configure 3 virtual devices in `configuration.yaml`, write a Python script that sends model responses as Home Assistant API calls
  - Goal: working pipeline where a text command reaches a device action, even if accuracy is low
  - Log in W&B: Day 0 accuracy per category (lighting, climate, access) on Benchmark v0.1 — this number appears in your paper as the untuned baseline
  - Success: you have a reproducible Day 0 number on Benchmark v0.1

- [ ] **Project 3: ChromaDB memory store**
  - What: ChromaDB is a local vector database — stores text as embeddings and retrieves the most semantically similar entries for any query. This is the memory layer of HomePersona (Phase 1 RAG cold start)
  - Task: store 100 fake user interaction logs (e.g. "User asked to dim bedroom lights at 10pm, set to 30%"). Write a retrieve(query) function. Test on 20 queries with known correct retrievals and measure precision@3
  - How: `pip install chromadb sentence-transformers`, embed with `all-MiniLM-L6-v2`, store and query
  - Goal: retrieval precision above 80% on your 20 test queries
  - Log in W&B: retrieval precision@3 and precision@5, embedding model used, average query latency
  - Success: you understand why embedding choice matters — this store powers the RAG cold-start phase

---

## Month 3 — First Working Pipeline

**Habits this month:**
- [ ] Read 2 papers/week — in context of what you're building, not as a separate activity
- [ ] Every Sunday: check new arXiv papers in personalisation, continual learning, on-device ML

**How to read papers fast:**
1. Title + abstract
2. Conclusion
3. All figures
4. Introduction
5. Methods (only if still interested)

**Tools:**
- [ ] Set up Connected Papers (connectedpapers.com) — paste any paper, see citation graph
- [ ] Set up paper reading system — Obsidian or Notion, 3-line summary per paper

### Week 9-10: Connect the Pieces — Full Baseline Pipeline

- [ ] **Project: First end-to-end pipeline**
  - What: connect all three components built in Week 5-8: ChromaDB memory store + local LLM (Llama 3.2 3B via Ollama or MLX) + Home Assistant API
  - Build the full query pipeline: user command → retrieve memories from ChromaDB → inject into LLM context → model responds → execute Home Assistant action
  - Run all 50 Benchmark v0.1 commands through this pipeline — compare against Week 7-8 Day 0 baseline. Did adding memory retrieval improve things?
  - Goal: working end-to-end pipeline with measurable improvement over Day 0 baseline
  - Log in W&B: accuracy vs Day 0 baseline, response latency, retrieval precision — this is Experiment 3 from Month 4 Phase A
  - Success: one number goes up vs Day 0 — even small improvement proves the pipeline works

- [ ] **Project: Retrieval tuning**
  - What: scale ChromaDB from 100 to 500 fake logs and tune retrieval — experiment with embedding models, top-k values, and similarity thresholds
  - Why: at 500 logs, retrieval noise becomes a real problem. Wrong memories injected into context actively confuse the model
  - Goal: retrieval precision above 85% on your 20 test queries (up from 80% in Week 7-8)
  - Log in W&B: precision@3 and precision@5 per embedding model — which embedding model wins?
  - Success: you can quantify exactly how much retrieval quality affects downstream task success rate

**Background reading this week (read alongside building):**
- [ ] **Federated Learning** — McMahan et al. 2017: 3 clients train on their own data, only share gradients. Read to understand the privacy angle you will reference in your related work section
- [ ] **PersonalHomeBench** — arXiv 2604.16813: read the paper that is your closest prior work. Understand exactly what they measure and what they don't — your differentiation must be crisp

- [ ] **Continual Learning Survey** — De Lange et al. 2022
  - No build challenge — it's a survey. Read it with one question: "what are the 3 main approaches to preventing catastrophic forgetting, and which one is most applicable to a LoRA adapter that updates from user interactions?"
  - Write a one-paragraph answer before moving on

- [ ] **LLM in a Flash** — Apple 2024
  - Read only — systems paper about running large models on small devices. Note: what is their core constraint and how do they work around it?

- [ ] **Phi-3 Technical Report** — Microsoft 2024
  - Read only — evidence that small models can be powerful with the right data. Note: what data decisions made Phi-3 outperform models 10x its size?

**Papers — Context (read only, understand the landscape):**

- [ ] SayCan — Google 2022. LLMs tell robots what to do. Background for grounding language to actions
- [ ] Inner Monologue — Google 2022. LLMs reason about robot actions in real time
- [ ] ALFRED benchmark — MIT 2020. Standard home task benchmark — know what everyone else evaluates on
- [ ] ACT (Action Chunking with Transformers) — Zhao et al. 2023. Imitation learning for physical robots. Expected knowledge at Figure AI and Physical Intelligence interviews
- [ ] RT-2 — Brohan et al. 2023. Foundation models for robot control. Understand the model class ML teams at robotics companies work on

### Week 11-12: Benchmark v0.2 + MemGPT Integration + LoRA Layer
- [ ] Read: "How to Read a Paper" — Keshav (3 page PDF)
- [ ] Read: "An Opinionated Guide to ML Research" — John Schulman blog post

- [ ] **Project 1: Build Benchmark v0.2 — Layer 2 dynamic evaluation**
  - What: v0.2 is two artifacts: (1) **chronological sequence files** and (2) **evaluation harness**. Together they turn the static v0.1 CSV into a temporal evaluation that measures the full learning arc
  - **Chronological sequence files:** structured day-by-day interaction sequences that define the training steps and validation checkpoints. Each sequence covers one habit being learned: Day 1-N the correct output is `ask`, Day N+1 onward (after sufficient feedback) the correct output transitions to `act`. Sequences also include a chaos block (3 days of anomalous context) and a bounce-back period. The sequences also add context columns — same utterance, different context, different correct action
  - **Evaluation harness:** runs the validation protocol after every adaptation epoch — (1) full Tier 1 anchor set for catastrophic forgetting check; (2) current habit scenario for alignment progress; (3) all prior habit scenarios for cross-habit forgetting; (4) bounce-back check after chaos block resolves. See `projects/homepersona/benchmark/schema.md` for the full validation protocol
  - Uncommon context labels: apply the Generalised Assistant Expectation Hierarchy as the labelling prior. For ambiguous cases, crowdsource 100+ responses — >80% consensus → majority label; <60% consensus → correct label is `ask`
  - Goal: sequence files + working harness versioned as v0.2 in `projects/homepersona/benchmark/`
  - Success: given a model checkpoint and a sequence file, the harness produces Alignment Velocity, cross-habit forgetting rate, and bounce-back pass/fail automatically

- [ ] **Project 2: MemGPT integration** *(use AI heavily to scaffold — understand by adapting, not by rebuilding from scratch)*
  - What: adapt MemGPT's memory architecture to your home automation pipeline. Replace their generic LLM backend with your locally running Phi-4-mini or Llama 3.2 3B. The memory tells the model what happened; the adapter will tell it who you are
  - Clone the Letta GitHub repo, read the core memory management files, understand what triggers a memory write vs a memory read
  - Integrate with your existing ChromaDB store — MemGPT's memory manager on top of your retrieval layer
  - Goal: working MemGPT-style memory layer running with your local model
  - Common failure modes to watch: wrong memories cluttering context; latency from retrieval; conflicting old vs new preferences

- [ ] **Project 3: Add LoRA adapter layer**
  - What: fine-tune a LoRA adapter on top of your base model using the home automation commands from Benchmark v0.1
  - Compare: no adapter vs LoRA adapter on Benchmark v0.1 — does fine-tuning help?
  - Log in W&B: accuracy with and without LoRA, LoRA rank comparison (r=4 vs r=8 vs r=16), training loss curve
  - Success: LoRA adapter improves accuracy over untuned base model on home commands

- [ ] Write first blog post — byline: Prateek Deshmukh and Samir Jibhakate: "We fine-tuned a 3B model on home automation commands and matched GPT-4 at 100x lower cost — and built the first benchmark to prove it." Post to Towards Data Science or your own blog
- [ ] Email 1 researcher in the personalisation / small models space — 3 sentences: what you built, why it matters, share the blog post link

---

## Month 4 — Go Deep on Your Niche

**Habits this month:**
- [ ] Read 2 papers/week

### Your Research Project: HomePersona

*"HomePersona: Continuously Adapting On-Device Models for Personalised Home Automation"*  
*Prateek Deshmukh\*, Samir Jibhakate\* — \*equal contribution*

**The core idea:**
Current home AI (Alexa, Google Home, Siri) is generic — it treats every user identically and forgets everything between sessions. Your research asks: what if the home AI was a small model that lived on your home hardware, learned your specific preferences continuously, and never sent your data to the cloud?

**The four-layer system you will build and study:**

```
Layer 1 — Base model:    Phi-4-mini (3.8B) or Llama 3.2 3B running locally via Ollama or MLX
                         General language understanding, runs on a Mac Mini
                         
Layer 2 — LoRA adapter:  A tiny set of extra parameters (~1M) on top of the base model
                         Updates continuously from your interactions
                         Captures your personal command style and preferences
                         
Layer 3 — Memory store:  A local vector database (ChromaDB) storing your history
                         "User always sets bedroom lights to 30% at 10pm"
                         Retrieved at query time and injected into context

Layer 4 — Preference layer: Learns from corrections and confirmations
                         When uncertain, system asks via home speaker before acting
                         Yes/no responses become labeled training data
                         Confirmation rate drops over time as system learns you
                         "User always declines cleaning when someone is home"
```

**Why this is novel:**
- PersonalHomeBench (April 2026) tests if a model can *reason about* a described user — HomePersona tests if a model's weights *adapt* from experience. Different problem, complementary contribution
- No paper has defined Alignment Velocity — how many interaction cycles to learn a specific user behaviour without explicit programming
- No benchmark measures the three-axis combination: adaptation speed + catastrophic forgetting + edge hardware efficiency
- Running entirely on home hardware with no cloud dependency is a real constraint nobody is optimising for
- The confirmation loop creates a data flywheel — the system collects its own labeled training data through use

**What you need — zero capital:**
- Your laptop or a Mac Mini (the base model runs on CPU)
- Ollama (free, runs Llama/Phi locally)
- ChromaDB (free, local vector database)
- Home Assistant (free, open source, runs on Raspberry Pi)
- Hugging Face `peft` library for LoRA

**The paper's central experiment:**
Run four conditions side by side over 4 weeks of simulated user interactions:
1. Generic GPT-4 with no personalisation (the baseline everyone compares against)
2. Small local model, no personalisation (shows cost of going local)
3. Small local model + memory only (shows what memory adds)
4. Small local model + memory + LoRA adapter (your full system)

Measure: task success rate, response latency, user preference score, confirmation rate (how often system asks vs acts), and critically — does performance improve and confirmation rate drop over time for conditions 3 and 4?

**The paper's secondary experiment — catastrophic forgetting:**
After 4 weeks of learning your preferences, introduce new conflicting preferences. Does the model forget old ones? How do you mitigate this? This is where your contribution to the continual learning literature comes from.

### Week 13-14: Phase A — Establish Baselines

- [ ] Read: "A Hacker's Guide to Language Models" — Jeremy Howard (YouTube)
- [ ] Read: "Direct Preference Optimization" (DPO) — Rafailov et al. 2023. The mechanism behind your preference layer — no RL required

- [ ] **Project: Build an eval harness for HomePersona**
  - What: a proper evaluation framework so every experiment produces comparable, trustworthy numbers. This is the Layer 2 harness from Benchmark v0.2 — do not run any experiment before this is working
  - Components: (1) RAGAS for RAG pipeline evaluation — retrieval relevance, answer faithfulness, context recall; (2) LLM-as-judge — use GPT-4 to score whether HomePersona's responses are correct and personalised; (3) task success rate — binary pass/fail per command against the Benchmark v0.1 ground truth; (4) **validation protocol** — after every adaptation epoch: run full Tier 1 anchor set (catastrophic forgetting check), run current habit scenario (alignment progress), run all prior habit scenarios (cross-habit forgetting), run bounce-back check after chaos block resolves
  - The harness runs the validation protocol automatically at every epoch checkpoint — not just at the end of training. A model that reaches high accuracy but catastrophically forgets between adaptation steps is a failed model; this catches it
  - Goal: given a model checkpoint, the harness produces 4 scores automatically: retrieval precision, response faithfulness, task success rate, and per-epoch Tier 1 anchor accuracy (the forgetting signal)
  - Success: you can rerun any experiment and get the same numbers; Tier 1 accuracy is logged at every epoch

- [ ] **Experiment 1:** Generic GPT-4 on 50 home automation commands — record accuracy and latency. This is the ceiling you are comparing against
- [ ] **Experiment 2:** Llama 3.2 3B with no personalisation on same 50 commands — record accuracy and latency. This is the cost of going local
- [ ] **Experiment 3:** Llama 3.2 3B + memory only (no LoRA) — does memory alone close the gap with GPT-4?
- [ ] **Experiment 4:** Llama 3.2 3B + LoRA only (no memory) — does the adapter alone help?
- [ ] **Experiment 5:** Full system — memory + LoRA — does combining them beat either alone?
- [ ] Log all 5 experiments in W&B with tags (GPT4-baseline, local-no-personalisation, local-memory-only, local-LoRA-only, local-full-system) — the comparison chart is Figure 1 in your paper
- [ ] Write second blog post after Phase A — byline: Prateek Deshmukh and Samir Jibhakate: "Does personalisation actually work? HomePersona Phase A results" — share the W&B chart publicly
- [ ] Email 2 researchers whose work is closest to yours — share the blog post and W&B report link. Researchers respond to concrete results, not ideas

### Week 15-16: Phase B — Personalisation Over Time

- [ ] Simulate 4 weeks of user interactions (generate synthetic interaction logs for a fictional user with consistent preferences — use AI to generate 500+ realistic log entries fast)
- [ ] Run your full system on week 1 data, measure task success rate
- [ ] Add week 2 data, update LoRA adapter and memory, measure again
- [ ] Repeat for weeks 3 and 4
- [ ] Log in W&B: task success rate per simulated week (weeks 1–4) — this plot is Figure 2 in your paper
- [ ] Plot: does task success rate go up over time? This is your key result

- [ ] **Project: Benchmark v0.3 — Chaos Robustness Test Set**
  - What: generate chaos test inputs by perturbing 1–3 context features in the v0.2 base (unusual time, flipped guest/exercise flags, extreme temperature, rare activity combinations). Label via the Generalised Norm hierarchy + crowdsourcing
  - Goal: 100+ chaos scenarios versioned as v0.3 in `projects/homepersona/benchmark/`
  - Success: you can measure Axis D — what % of chaos scenarios does the model handle correctly via the generalised fallback?

---

## Month 5 — Deep Experiments

**Habits this month:**
- [ ] Read 2 papers/week — focus on papers you will cite in the related work section

### Week 17-18: Phase C — Catastrophic Forgetting

- [ ] Read: "Elastic Weight Consolidation" paper — one of the main techniques for preventing catastrophic forgetting. Read before running this experiment
- [ ] Read: RAGAS documentation — framework for evaluating RAG pipelines

- [ ] After 4 weeks of learning "User A" preferences from Phase B, switch to "User B" preferences
- [ ] Measure: how quickly does the system forget User A? How quickly does it learn User B?
- [ ] Try Elastic Weight Consolidation to slow forgetting — does it help?
- [ ] Log in W&B: forgetting rate (User A accuracy after N User B interactions) — this is your second key result and continual learning contribution
- [ ] Write third blog post — byline: Prateek Deshmukh and Samir Jibhakate: "What we learned about catastrophic forgetting in a personalised home AI" — share the forgetting curve publicly
- [ ] Email the Letta team — 3 sentences: built on your memory architecture, adapted to home automation, found X about catastrophic forgetting. This is your strongest opener

### Week 19-20: Phase D — Ablation + Chaos Robustness

- [ ] Remove memory layer — how much does performance drop?
- [ ] Remove LoRA adapter — how much does performance drop?
- [ ] Reduce LoRA rank (r=4 vs r=8 vs r=16) — where is the minimum that still works?
- [ ] Test on a new user with completely different preferences — does the system generalise?
- [ ] Log in W&B: ablation table — all conditions with and without each component
- [ ] Run Benchmark v0.3 (chaos test set) against the full system — measure Axis D: what % of chaos scenarios does the model handle correctly via the generalised fallback?
- [ ] Log in W&B: Axis D score broken down by perturbation type — which context feature perturbations break the model most?

**Failure analysis (ongoing throughout Month 5)**
- [ ] Collect 20 specific examples where your system fails — categorise them
- [ ] Ambiguous commands: "make it nice" — system doesn't know what nice means to you yet
- [ ] Conflicting memories: old preference vs new preference, which wins?
- [ ] Cold start failures: first week has no data, what does the system do?
- [ ] Understanding these failures is where your next paper comes from

---

## Month 6 — Write + Submit

**Habits this month:**
- [ ] Read 2 papers/week

> **Build-heavy alternative:** If the system is working well and the paper feels like it's blocking you from shipping, prioritise in this order: (1) clean GitHub with reproducible code, (2) demo video of HomePersona learning preferences in real time, (3) blog post explaining what you built and why. This gets you into Tier 2-3 companies without a paper. The paper is what opens Tier 1 doors — do it if time allows, not at the cost of the working system.

### Before Writing
- [ ] Watch: "How to Write a Great Research Paper" — Simon Peyton Jones (YouTube, 1hr)
- [ ] Read: "Writing Pet Peeves" — Wojciech Zaremba blog
- [ ] Read 3 papers in your area specifically for writing style

### Your Paper's Narrative

Write this story — every section should serve it:

```
Problem:    Home AI today is generic and stateless. Recent benchmarks 
            (PersonalHomeBench, 2026) test whether models can reason about 
            a described user — but no system actually adapts its weights 
            from experience. Reasoning about you is not the same as 
            learning to be you.

Insight:    Home environments are constrained enough that a small local 
            model whose weights continuously adapt from interactions should 
            outperform a large generic model on personalised commands.
            You do not need world knowledge — you need to know this user.

Method:     HomePersona — a local LoRA adapter that updates continuously 
            from user interactions, backed by a semantic memory store.
            Evaluated on three axes: Alignment Velocity (how fast it 
            learns you), Memory Retentiveness (catastrophic forgetting), 
            and Inference Efficiency (accuracy per ms per MB on edge hardware).
            Runs entirely on home hardware. No cloud. No privacy risk.

Result:     HomePersona reaches X% task success on personalised commands 
            after N interaction cycles, retains Y% of prior habits when 
            learning new ones, and runs at Z ms on a Mac Mini — 
            outperforming prompted GPT-4 on user-specific commands 
            while running entirely locally.
```

### Paper Structure
```
Abstract:     4 sentences — problem, insight, method, result
Introduction: the gap (generic AI for personal spaces), why it matters,
              what you do, 3 bullet contributions
Related Work: MemGPT (memory), LoRA (adaptation), federated learning 
              (privacy), small language models — explain how you differ from each
Method:       diagram of the three-layer system, LoRA setup details, 
              memory retrieval approach, Home Assistant integration
Experiments:  5 baseline experiments, personalisation over time plot,
              catastrophic forgetting results, ablation table
Conclusion:   what you showed, limitations (synthetic data, one home), 
              future work (real users, real hardware, more devices)
```

### Tools
- [ ] Set up Overleaf (LaTeX, standard for ML papers)
- [ ] Use NeurIPS 2026 workshop template
- [ ] Matplotlib for graphs, Inkscape for diagrams

### Submission Checklist
- [ ] Week 21-22: Write full paper draft
- [ ] Week 23: Post preprint on arXiv — same day as submission, do not wait for acceptance
- [ ] Week 23: Submit to workshop
- [ ] Week 23: Update resume and GitHub README with arXiv link immediately
- [ ] Week 23: Post on Twitter/X and LinkedIn — one thread explaining what you built and what you found
- [ ] Week 23: Email every researcher you have contacted over the past months — share the paper link directly

### Workshop Targets
| Conference | Workshop | Deadline | Why |
|---|---|---|---|
| NeurIPS 2027 | "Personalization of Generative AI" workshop | ~August 2027 | Your exact topic |
| NeurIPS 2027 | "Open-World Agents" workshop | ~August 2027 | Home automation angle |
| CoRL 2027 | "Language and Robot Learning" | ~July 2027 | Robotics community |
| ICLR 2027 | Tiny Papers / workshop track | ~October 2026 | Earlier deadline — submit preprint first |

*Note: NeurIPS 2026 deadlines are July-August 2026, before experiments are complete. Targeting 2027 conferences is realistic. ICLR 2027 tiny papers is the earliest viable submission.*

---

## Month 7 — Job Search

**Habits this month:**
- [ ] Apply to Tier 3 first (Letta, Josh.ai, Home Assistant) — build interview reps before Tier 1

### Your Profile by Month 7
- [ ] GitHub: paper code, clean implementation, reproducible experiments
- [ ] arXiv: preprint with your name on it
- [ ] Resume: senior SWE + ML research + published work
- [ ] Blog: 2-3 posts explaining your research in plain English
- [ ] Twitter/X: engaged in ML research community

### Target Companies

**Comp filter applied:** all companies meet within 20% of big tech senior/staff total comp ($240K–$320K+ in US, £120K–£200K+ in London, €120K+ in Europe adjusted for local market). European companies noted where location affects comp.

**Strategy:** Start applying to Tier 3 in November 2026 to practice interviewing. Target Tier 2 from December 2026. Tier 1 is the medium-term aspiration — your profile after this roadmap gets you in the door.

---

**Tier 1 — Dream Companies** *(high comp + prestige + publish friendly — very competitive, target after stronger profile)*

| Company | Why | Comp Notes |
|---|---|---|
| OpenAI | Frontier AI, agents, on-device research — your skills directly relevant | $500K–$1.2M+ total comp |
| Anthropic | Safety-focused frontier AI, strong research culture, publish friendly | $563K–$756K total comp |
| Google DeepMind | Best AI research lab in the world, London + US offices, publish everything | Big tech rates, £150K–£250K+ London |
| Meta FAIR | Open research culture, publishes heavily, strong robotics team | Big tech rates $300K–$400K+ |
| Microsoft Research | Research-first culture, strong on-device AI focus, publish friendly | Big tech rates $300K–$400K+ |
| Apple ML Research | On-device ML is their entire strategy — your paper is exactly what they build toward | $190K–$528K, strong RSUs |
| Physical Intelligence | Foundation models for robots, research culture, strong equity at $2B valuation | Competitive with big tech |
| Skild AI | General-purpose robot AI, $1.4B Series C at $14B valuation, strong research output | Competitive, significant equity upside |

---

**Tier 2 — Target Companies** *(competitive comp + real impact + realistic with your profile by end of 2026)*

| Company | Why | Comp Notes |
|---|---|---|
| Figure AI | Humanoid robots, $39B valuation, ML engineering at scale | Competitive, strong equity |
| LangChain / LangGraph | Agent infrastructure your HomePersona runs on, values builders | Well-funded, competitive |
| Amazon Alexa AI | Solving generic home AI is their core problem, big tech rates | Big tech subsidiary rates |
| Wing / Alphabet | Drone delivery at scale, big tech subsidiary, strong ML infra team | Alphabet rates |
| Anyscale | Distributed ML infrastructure (Ray), Series C, engineering-focused culture | $144K–$350K total comp |
| Together AI | LLM inference infrastructure, well-funded, strong engineering team | Well-funded Series B+ |
| Modal | Serverless ML compute, growing fast, engineering-first | Well-funded, competitive |
| Mistral | Small capable models, your base model of choice *(US office roles only — Paris cash below threshold but equity meaningful at $14B valuation)* | US: competitive; Paris: €108K–€130K cash |

---

**Tier 3 — Practice + Accessible** *(good work and mission fit, start here in October to build interview reps — slightly below comp threshold in cash but equity upside noted)*

| Company | Why | Comp Notes |
|---|---|---|
| Letta | You reproduced their paper — they will know your work. Small team, high impact | Early stage equity upside, below cash threshold |
| Josh.ai | Your research solves their exact problem. Privacy-first home AI | Small startup, below cash threshold, equity bet |
| Home Assistant / Nabu Casa | You built on their platform — strongest possible signal | Open source org, below threshold |
| Agility Robotics | Humanoid for logistics, strong ML team, more accessible than Figure | Growing, comp improving |
| Joby Aviation | Air taxi launching 2026, ML for safety-critical systems | Aerospace comp structure, below threshold |
| Archer Aviation | Similar to Joby, active hiring ramp | Aerospace comp structure |
| Zipline | Drone delivery operational at scale, meaningful ML work | Below threshold in cash |
| Samsung Research | Home automation AI across device ecosystem | Below US threshold, good for international |
| YC-backed AI startups | Practice interviews, build network, occasionally convert to great roles | Variable, often below threshold in cash |

### How to Apply (Don't Just Apply Online)
- [ ] Find researchers at target companies whose work is closest to yours
- [ ] Email directly — do not just apply online, that is the slow path. Template: 3 sentences — what you built, why it is relevant to their work, attach arXiv link, ask for 20 min call
- [ ] Write a blog post per week explaining your research in plain English — people will find you, cross-post to Towards Data Science

### Interview Prep
| Round | What They Test | Your Status |
|---|---|---|
| Coding | Algorithms, systems | Already strong |
| ML fundamentals | Backprop, loss, overfitting, transformers | Roadmap covers this |
| Paper discussion | "Walk me through a recent paper" | Roadmap covers this |
| Research taste | "What problems interest you and why?" | Develop through reading |
| System design | ML pipelines, training infrastructure | Already strong |

### Production ML Tools Survey (2-3 days, do before interviews start)
Quick survey — not deep learning, just enough to speak to them in interviews:
- [ ] **Cleanlab** — automatic data quality detection, finds mislabelled examples using model confidence scores
- [ ] **Label Studio** — open source data labelling tool, understand how annotation pipelines work
- [ ] **MLflow** — experiment tracking alternative to W&B, widely used in industry
- [ ] **Data versioning** — DVC (Data Version Control), understand why versioning data matters at scale
- [ ] **Embedding clustering for failure analysis** — how to group model failures at scale instead of reviewing individually

One sentence answer for interviews: "At small scale I use plot_top_losses and manual review. At production scale I'd use Cleanlab for automatic mislabel detection and embedding clustering to review failure groups rather than individual examples."

---

## Parallel Tracks (Run Throughout)

### Community (30 min/week)
- [ ] Follow on Twitter/X: Andrej Karpathy, Sergey Levine, Chelsea Finn, Pieter Abbeel, Lerrel Pinto
- [ ] Join Discord: Hugging Face, LeRobot, Papers with Code
- [ ] Follow: r/MachineLearning, r/robotics

### Paper Reading (From Month 3 Onwards)
- [ ] Read 2 papers/week minimum
- [ ] Use: arxiv-sanity, Papers with Code, Semantic Scholar
- [ ] Every Sunday: check new arXiv papers in your area

### Hardware (Optional, Month 4+)
- [ ] Koch v1.1 — $300 DIY robot arm, used in LeRobot tutorials. Full plan in `projects/koch-arm/PLAN.md` — 4 phases from basic control → imitation learning → ego-scale human video transfer → LoRA personalisation. Work on this when you need a break from HomePersona, no deadline.
- [ ] Raspberry Pi 5 + Home Assistant — run real home automation locally

---

## Complete Resource List

### Courses
| Resource | Where | When |
|---|---|---|
| Fast.ai Practical Deep Learning | fast.ai (free) | Month 1 |
| Hugging Face Deep RL Course | huggingface.co/learn (free) | Month 2 |
| CS231n lecture notes | cs231n.github.io (free) | Reference |
| CS224N lecture notes | web.stanford.edu/class/cs224n (free) | Reference |

### Books
| Book | Where | Notes |
|---|---|---|
| "Deep Learning" — Goodfellow et al. | Free PDF online | Reference, not cover to cover |
| "Reinforcement Learning" — Sutton & Barto | incompleteideas.net (free) | Read Ch 1-6 |

### Key Papers by Phase
| Phase | Papers |
|---|---|
| Month 1 | Attention is All You Need, Deep Residual Learning, Dropout, LoRA |
| Month 2 | MemGPT, Continual Learning Survey, Federated Learning (McMahan), LLM in a Flash, Phi-3, SayCan, CLIP, ALFRED |
| Month 3-4 | MemGPT deep read + all cited papers, Elastic Weight Consolidation, QLoRA, DoRA (Weight-Decomposed LoRA) |
| Month 5-6 | 20+ papers on personalisation, continual learning, on-device inference |
| Month 7 | Papers you are comparing against in experiments |

### Tools
| Tool | Purpose | When to Start |
|---|---|---|
| PyTorch | All ML implementation | Month 1 |
| Hugging Face `transformers` | Pretrained models, fine-tuning | Month 1 |
| Hugging Face `peft` | LoRA and adapter fine-tuning | Month 1 Week 5 |
| Weights & Biases | Experiment tracking — use for every run | Month 2 — use always |
| Ollama | Run Llama/Phi locally on your laptop, zero cloud cost | Month 1 Week 5 |
| MLX | Apple's ML framework — faster than PyTorch on Apple Silicon for inference | Month 1 Week 5 |
| ChromaDB | Local vector database for the memory layer | Month 3 |
| mem0 | Managed memory layer for AI agents — evaluate as alternative to rolling your own ChromaDB pipeline | Month 3 |
| Home Assistant | Open source home automation platform, real device control | Month 3 |
| Letta / MemGPT | Memory architecture you will reproduce and extend | Month 4 |
| Home Assistant | Open source home automation — real device control and virtual device testing | Month 3 |
| Overleaf | Paper writing in LaTeX | Month 7 |
| Modal / Lambda Labs | Cheap GPU compute for larger fine-tuning runs | Month 5 |
| Connected Papers | Literature mapping — paste any paper, find related work | Month 2 |
| arXiv + Semantic Scholar | Paper discovery | Month 2 |
| vLLM | Fast LLM inference for serving your model | Month 6 |

---

## The Most Important Rules

1. **Your niche is decided — HomePersona. Commit to it.** Depth beats breadth for research roles. Do not switch topics.
2. **Log every experiment in W&B.** Never lose a result. A result you cannot reproduce is worthless.
3. **Post on arXiv the same day you submit.** The preprint is on your resume immediately — do not wait for acceptance.
4. **Email researchers directly.** Online applications are the slow path. The Letta team, Josh.ai team, and Home Assistant maintainers are all reachable.
5. **Blog about your work from Month 3.** Write "I am building a personalised home AI that runs locally" — people will find you.
6. **Show up every day.** 30 minutes of consistent work beats 10hr weekend binges.
7. **Read 2 papers a week from Month 2.** Research taste is built through volume of reading.
8. **Your SWE background is an advantage, not a gap.** You can build things researchers cannot. Use it.
