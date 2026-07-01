---
title: DeepFit AI — Redrob AI Hiring Hackathon
emoji: 🎯
colorFrom: red
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# DeepFit AI

**Ranking candidates by fit, not keywords.**

Built for the Redrob AI Hiring Hackathon. This demo shows the same ranking logic
used to score 100,000 candidates from the Redrob pool.

## What makes this different from Caliber

| Feature | Caliber | DeepFit AI |
|---|---|---|
| Primary signal | Skill tags | Career description text |
| Keyword stuffers | Ranked high | Floored to 0.02 |
| Behavioral | Additive weight | Bounded multiplier [0.50x-1.15x] |
| Reasoning | Template-based | Feature-value driven (zero hallucination) |
| Per-card feature breakdown | No | Yes (5 mini-bars: rs/exp/nlp/prod/rec) |
| Behavioral multiplier badge | No | Yes (up/down with exact value) |

## Scoring formula

```
base  = 0.28×role_substance + 0.22×exp + 0.08×skill_corroboration
      + 0.08×nlp_ir + 0.08×product + 0.05×recency + 0.05×edu + 0.03×loc

final = base × behavioral_multiplier  ×  honeypot_penalty
```

`role_substance` reads career **descriptions** using 5 regex patterns over
substance areas (retrieval/embeddings, ranking/LTR, recommendation/recsys,
search/IR, applied-ML-in-production). Skill tags are only used as a
corroboration gate — 4+ AI tags with zero description substance = stuffer, floored.

## Running locally

```bash
pip install gradio>=4.0
python testing/sandbox/app.py
```
