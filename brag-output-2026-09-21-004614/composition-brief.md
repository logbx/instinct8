# Hyperframes Composition Brief: Instinct8

## Objective
Create a short launch-style brag video for Instinct8, a context compression middleware that prevents goal drift in long-running LLM agents.

## Output
- Composition directory: `/workspace/brag-output-2026-09-21-004614/composition/`
- Rendered video: `/workspace/brag-output-2026-09-21-004614/brag.mp4`
- Format: landscape — 1920x1080
- Duration: 20 seconds

## Source Material
- Project root: `/workspace`
- Primary files read: README.md, docs/features/SELECTIVE_SALIENCE.md, mcp_server/README.md, pyproject.toml
- Product name: Instinct8
- Tagline / strongest claim: "Context compression with goal drift prevention for LLM agents"
- Key concept to visualize: Selective salience extraction — the agent judges what matters and protects it during compression
- Copy that must appear verbatim:
  - "Long-running agents forget what they're doing."
  - "Instinct8 asks the agent: What matters?"
  - "Goal Coherence: 80% → 20%"
  - "7% drift per compression"
  - "Validated with 50-turn benchmarks"
  - "pip install instinct8-mcp"

## Creative Direction
- Tone preset: polished
- Creative direction: Scientific elegance — quiet confidence for serious research with real-world impact
- Interpretation: This is validated research solving a real problem. Pacing is deliberate, copy is precise, visuals are clean. No gimmicks, no overselling — let the science speak. Typography is minimal and technical. Transitions are smooth and confident.
- Angle: Show problem (agents forgetting goals) → solution (selective salience) → proof (benchmarks) → action (pip install)
- Hook: "Long-running agents forget what they're doing." — the universal pain point
- Outro / punchline: "pip install instinct8-mcp" — clean, practical CTA
- Avoid:
  - Generic AI/ML hype language
  - Abstract visualizations disconnected from the product
  - Over-designed or flashy transitions that compete with the science

## Visual Identity
- Background: #0a0a0a (near-black, professional)
- Text: #f0f0f0 (off-white, readable, high contrast)
- Accent: #3b82f6 (blue, technical, approachable)
- Display font: Inter or system-ui (clean, modern, technical)
- Body font: Inter or system-ui
- Visual references from the project: Terminal/CLI aesthetic, code blocks, research metrics, technical documentation style

## Storyboard
Use the storyboard in `/workspace/brag-output-2026-09-21-004614/brag-plan.md` as the creative contract.

Scene summary:
1. **The Problem Hook** — 3s — "Long-running agents forget what they're doing." (centered text, minimal)
2. **The Drift Visualization** — 5s — Progress bar draining from 80% to 20%, label "Goal Coherence", subtext "7% drift per compression"
3. **The Solution Reveal** — 6s — "Instinct8 asks the agent: What matters?" + 3 quote cards appearing sequentially
4. **The Proof** — 4s — "Validated with 50-turn benchmarks" + 3 checkmarked metrics
5. **The Call to Action** — 2s — "pip install instinct8-mcp" (terminal-style, centered)

## Audio
- Audio role: warm professional bed with tasteful accents
- Audio arc: Music fades in gently during Scene 1, holds through scenes 2-4, fades out during Scene 5
- Music: Professional tech/corporate track with subtle forward motion (select from bundled tracks that match this mood)
- Music treatment: Fade in 0.3s during Scene 1, hold at 0.5-0.6 volume, gentle fade out under Scene 5
- Music cue guidance: Detect at composition time via `npx hyperframes beats`. Target 2-3 strong cues for: (1) Scene 1→2 transition (~3s), (2) quote cards reveal in Scene 3 (~10-12s), (3) final pip install in Scene 5 (~18s)
- Audio-reactive treatment: subtle; use RMS for gentle breathing on text emphasis and metric reveals, no heavy pulsing or waveforms
- Audio-coupled moments:
  - Scene 2: progress bar draining (subtle tick or whoosh)
  - Scene 3: quote cards appearing one by one (soft card sound per card)
  - Scene 5: final pip install line (clean announcement tone)
- SFX selection guidance: Choose low high-frequency-risk sounds for repeated moments (quote cards). Prefer professional, technical aesthetic over playful sounds.
- SFX analysis guidance: `/workspace/brag/skills/brag/assets/sfx/sfx-analysis.md` if present
- Exact SFX choice: Hyperframes should choose filenames, timestamps, density, and volume based on the implemented animation.
- Audio files: copy the chosen music and any Hyperframes-selected SFX into `/workspace/brag-output-2026-09-21-004614/composition/assets/`

## Hyperframes Instructions
Requirements:
- Show the core concept: goal drift problem → selective salience solution → scientific validation → practical action
- Keep all text readable in the final render (minimum 0.8s settled for short labels, 0.3s per word for sentences)
- Keep the video at exactly 20 seconds
- Include music and SFX as planned (use bundled assets)
- Use local assets for audio and any required runtime/media dependencies
- Treat music cue metadata as optional timing hints — prioritize readability and scene pacing
- Major reveals may align to strong cues within ±0.15s; quote cards may align to consecutive beats within ±0.10s if readable spacing allows (at least 0.8s apart for text)
- Use audio-reactive workflow: extract audio data and use RMS for subtle emphasis on text and metrics
- Run `hyperframes check` before render — this is brag's single gate
