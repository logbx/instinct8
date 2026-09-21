# Brag Plan: Instinct8

## What is this app?
Instinct8 is a context compression middleware for long-running LLM agents that prevents goal drift through selective salience extraction — the agent judges what matters and protects it during compression.

## The angle
Show the core problem (agents forgetting their goals after compression) → reveal Instinct8's solution (selective salience extraction) → demonstrate the scientific proof (7% drift reduction, validated benchmarks) → land on the practical value (works as both CLI and MCP server for Claude Code).

The hook is the problem statement: "Long-running agents forget what they're doing." Everyone who's used an LLM agent for more than 15 minutes has felt this pain. The video makes that invisible problem visible, then shows the elegant solution: ask the agent itself to judge what matters.

## Hook (first 2-3 seconds)
**"Long-running agents forget what they're doing."**

Bold, white text on dark background. Simple. True. Painful. This is the thing developers complain about but accept as inevitable. Starting with the problem — not the solution — earns the next 20 seconds.

## Key moments (the middle)
1. **The goal drift visualization** — Show "Goal Coherence: 80% → 20%" with a progress bar draining. This makes the invisible problem visible and quantifiable.
2. **The selective salience reveal** — "Instinct8 asks the agent: What matters?" followed by verbatim quotes being extracted and protected. This is the core innovation — agent-as-judge.
3. **The scientific validation** — "7% average goal drift per compression" with a checkmark. Real numbers, real evaluation, real science.

## Outro / punchline
**"pip install instinct8-mcp"**

Clean, practical, actionable. The video proves the value, the outro shows how easy it is to get started. One command. No hype.

## User flow worth showing
This is a research tool/library, not a user-facing app with a traditional flow. However, the conceptual flow worth showing is:
1. **Problem state** — Agent hits context limit, traditional compression causes drift
2. **Instinct8 intervention** — Selective salience extraction preserves goal-critical information
3. **Result state** — Agent maintains goal coherence across compressions

This is more of a "before/after/proof" structure than a traditional user flow, but it demonstrates what the product *does* rather than just what it *claims*.

## Tone
- Preset: **polished**
- Creative direction: **Scientific elegance — quiet confidence for serious research with real-world impact**
- Interpretation: This is a validated research contribution solving a real problem. The pacing is deliberate, the copy is precise, the visuals are clean. No gimmicks, no overselling — let the science and the problem-solution clarity speak for themselves. Typography is minimal and technical. Transitions are smooth and confident.

## Format: landscape — 1920x1080
## Duration: 20 seconds

## Visual identity (from the project)
- Background: `#0a0a0a` (near-black, professional)
- Text: `#f0f0f0` (off-white, readable)
- Accent: `#3b82f6` (blue, technical but approachable)
- Display font: Inter or system-ui (clean, modern, technical)
- Body font: Inter or system-ui
- Strongest visual element: The README's problem statement and metrics (goal drift %, compression ratio, behavioral alignment scores)

## Share copy (draft)
Long-running LLM agents gradually forget their goals after context compression. Instinct8 prevents this through selective salience extraction — the agent judges what matters and protects it. 7% drift reduction, validated with 50-turn benchmarks. Works as a CLI or MCP server for Claude Code. pip install instinct8-mcp

## Audio direction
- Role: warm professional bed with tasteful accents
- Music: Polished corporate/tech track — something with subtle forward motion and professional warmth
- Music treatment: Fade in gently (0.3s), hold at moderate level (0.5-0.6 volume), subtle fade out under final text
- Music cue guidance: To be detected at composition time via analyze_music_cues.py. Target 2-3 strong cues for: (1) hook → problem reveal transition (~3s), (2) selective salience reveal (~10-12s), (3) final pip install (~18s)
- Audio-reactive treatment: subtle; use RMS to add gentle breathing to text emphasis and metric reveals, not heavy pulsing
- SFX posture: sparse; motion-matched; professional restraint
- Audio-coupled moments: 
  - Progress bar draining (subtle tick or whoosh)
  - Quote extraction (soft card sound)
  - Final pip install line (clean announcement tone)
- Restraint rule: No aggressive hits, no chaotic energy, no waveform visuals. Sound supports the science — it doesn't compete with it.

## Storyboard

### Scene 1 — The Problem Hook — 3s
**"Long-running agents forget what they're doing."**

White text, centered, on dark background. Clean typography. The line fades in quickly (0.3s) and holds centered for 2.5s. No motion, no distraction — just the problem statement.

Sequential/interaction: none
Audio intent: Set professional tone with subtle music entry
Audio-coupled idea: none — silence or near-silence for first 0.5s, then music fades in
Music: warm professional bed, fade-in
Transition mood: soft crossfade → Scene 2

### Scene 2 — The Drift Visualization — 5s
Show the invisible problem becoming visible:
- **"Goal Coherence"** label at top
- **Progress bar**: starts at 80%, drains to 20% over 2.5s
- **Text below**: "7% drift per compression" fades in after bar completes

The bar drain makes the abstract concept concrete. This is what "goal drift" looks like. Numbers from the actual README research.

Sequential/interaction: yes — bar drains progressively
Audio intent: Emphasize the problem's severity through motion
Audio-coupled idea: soft ticking or whoosh as bar drains
Music: bed continues, may align bar completion to a nearby beat
Transition mood: clean cut → Scene 3

### Scene 3 — The Solution Reveal — 6s
**"Instinct8 asks the agent: What matters?"**

Headline appears at top (1s settle).

Then, below it, 3 verbatim quote cards appear one by one (each ~0.8s apart):
- Card 1: "Choose PostgreSQL for relational guarantees"
- Card 2: "Latency must stay under 150ms"
- Card 3: "Cannot use AWS Aurora (compliance)"

These are real examples from the selective salience docs. Cards are clean, technical, styled like code blocks or terminal output. Each card slides in from left with a subtle motion.

Sequential/interaction: yes — 3 cards appear one by one
Audio intent: Reveal the core innovation — agent-as-judge
Audio-coupled idea: soft card sound on each arrival
Music: bed continues, cards may snap to beat grid if tempo allows readable spacing (~0.8s+ apart)
Transition mood: soft slide → Scene 4

### Scene 4 — The Proof — 4s
**"Validated with 50-turn benchmarks"**

Clean text at top. Below it, a simple metric display:
- ✓ 7% average goal drift
- ✓ 95%+ goal coherence maintained
- ✓ Works with any LLM

Simple checkmarks, clean typography. This is the science — real evaluation, real numbers, real validation.

Sequential/interaction: none — all text appears together
Audio intent: Land the credibility and validation
Audio-coupled idea: single clean tone when scene enters
Music: bed continues at full volume
Transition mood: smooth crossfade → Scene 5

### Scene 5 — The Call to Action — 2s
**"pip install instinct8-mcp"**

Large, centered, terminal-style text. Clean, actionable, practical. The video proved the value — now show how easy it is to start.

Sequential/interaction: none
Audio intent: Clean, confident finish
Audio-coupled idea: final soft tone or music swell
Music: gentle fade out under text
Transition mood: fade to black (end)

**Music mood for this video:** professional tech/corporate, warm but not cheesy, forward motion without aggression
**Audio summary:** Sparse professional sound design that supports the science without competing. Music provides warmth and continuity, SFX accent key moments, everything stays restrained and confident.
