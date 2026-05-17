# Claude Code Instructions — AI Robotics Journey

## Project Context
This is a personal learning + research project. The goal is to get a research role at a robotics/AI company and publish 1 paper by end of 2026. The research niche is **personalised on-device continual learning for home automation** (project name: HomePersona). The primary reference document is `ROADMAP.md` at the root of this repo.

---

## Roadmap Update Command

When the user says **"update roadmap"** followed by context, follow this process:

### Step 1 — Identify which section is affected
Read `ROADMAP.md` and identify which section(s) the user's context touches:
- Started or finished a paper/project → find that item in the relevant Month section
- Something became irrelevant → find every occurrence and flag for removal
- New tool or framework → check Tools section and Key Papers section
- Company changed or new company found → find Target Companies section
- Stuck on something → find the relevant project or week and adjust
- Monthly review → scan all upcoming months for anything stale

### Step 2 — Make the minimal targeted change
Only edit the section(s) that are directly affected. Do not rewrite unrelated sections. Preserve all checkboxes — checked items stay checked.

### Step 3 — Apply one of these update patterns based on what the user said:

**"I started [X]"**
- Find the item in the roadmap
- If it has sub-items, add a note: `*(in progress — started [date])*`
- If the user shared early learnings, add them as indented notes under the item

**"I finished [X], here is what I learned: [notes]"**
- Mark the item as `[x]`
- Add a `> Note:` block under it with key learnings in 1-3 bullet points
- Check if anything learned changes upcoming sections — if yes, update those too and flag the change to the user

**"[Tool/paper/company] is no longer relevant because [reason]"**
- Remove or strike through the item
- If it was a core part of a project or experiment, suggest a replacement
- Update the Tools or Key Papers table if affected

**"I came across [new thing], it does [X]"**
- Evaluate whether it belongs in the roadmap
- If yes: add it to the right section with a note on what it replaces or complements
- If no: tell the user why and don't add it

**"[Company] changed focus / new company [X] appeared"**
- Update the Target Companies section
- Adjust tier if needed
- Add a one-line note on why the change was made so future-you understands

**"I've been stuck on [X] for [time] because [reason]"**
- Find the relevant item
- Assess: skip it, simplify it, or find an alternative approach
- Update the roadmap to reflect the adjusted plan
- Flag to the user what changed and why

**"Monthly review — here is what I've done: [list]"**
- Mark completed items as `[x]`
- Scan all upcoming months — flag anything that looks too ambitious, outdated, or should be reordered
- Suggest any additions based on what the user has learned
- Give the user a short summary: what is on track, what is at risk, what changed

**"After reading [paper], I now think [opinion]"**
- Evaluate whether the opinion changes the HomePersona research direction
- If yes: update the affected experiment or project section and explain the change
- If no: tell the user why the current plan still holds

---

## General Rules for All Roadmap Updates
- Always read the full `ROADMAP.md` before making any edit
- Never remove a checked `[x]` item — completed work stays in the doc as a record
- Keep the same level of detail as the existing entries — What, Why, How, Goal, Success
- If a change affects multiple sections, update all of them in one pass
- After updating, tell the user in 2-3 lines: what changed, which section, and why

---

## Other Project Conventions
- All ML experiments must be logged in Weights & Biases
- Source code lives in `foundations/` (learning) and `projects/` (research work)
- The HomePersona project will live in `projects/homepersona/` when created
- Use `resources/` for saving paper notes and references
