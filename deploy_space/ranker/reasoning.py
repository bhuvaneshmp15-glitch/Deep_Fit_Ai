"""
src/ranker/reasoning.py
=======================
WHAT THIS MODULE DOES:
  Generates a human-readable reasoning string for a scored candidate by
  reading COMPUTED FEATURE VALUES from CandidateScore — not templates.
  Every clause in the output is grounded in a real numeric signal.

WHAT THIS MODULE DOES NOT DO:
  Compute any score. Does not access config.WEIGHTS or call any scorer.
  Does not produce rankings or write to disk.

WHY THIS BOUNDARY EXISTS:
  Reasoning strings are a presentation concern, not a scoring concern.
  Decoupling them means changing the text never risks changing a score,
  and the scorer can be unit-tested independently of reasoning quality.

DESIGN — generate_reasoning_v2():
  - Zero templates. Every clause maps a real float → a human phrase.
  - Positive clauses: triggered when feature values EXCEED thresholds.
  - Gap clauses: triggered when feature values FALL BELOW thresholds.
  - Opener follows SCORE BAND (not substance alone):
      >= 0.85 → "Strong fit"   0.65–0.84 → "Solid fit"
      0.45–0.64 → "Fit"        < 0.45 → "Marginal fit"
  - Clause ORDER is deterministic per candidate (MD5 hash of candidate_id
    mod 4 picks one of 4 fixed orderings). Same candidate → same order on
    every run. Different candidates → different clause emphasis.
  - Leading positive clauses are capped at 3 for ranks > 30 to avoid
    lower-ranked candidates reading identically to top candidates.
  - Behavioral multiplier clause: appears only when it meaningfully moves
    the score (beh_mult >= 1.03 or <= 0.75).
"""

from __future__ import annotations

import hashlib

from .schema import CandidateScore
from .features import _f, _i
from .config import HARD_AI_SKILLS_WEIGHTED as HARD_AI_SKILLS


# ── Public entry point (called by rank.py) ────────────────────────────────────

def reasoning_for(cs: CandidateScore, rank: int = 0) -> str:
    """
    Generate a grounded reasoning string from a CandidateScore.

    rank: the final rank position (0 = placeholder, updated after boost).
    Returns a string <= 300 characters suitable for submission CSV.
    Delegates to generate_reasoning_v2() for the actual text.
    """
    if cs.is_stuffer:
        return cs.stuffer_reason

    raw = cs.raw_candidate
    f   = cs.features

    return generate_reasoning_v2(
        raw_candidate  = raw,
        final_score    = cs.final_score,
        role_substance = f.role_substance,
        nlp_ir         = f.nlp_ir_signal,
        product_score  = f.product_score,
        recency_score  = f.recency_score,
        beh_mult       = cs.behav_score,   # the multiplier value [0.50, 1.15]
        rank           = rank,
    )


# ── Deterministic clause-order seed ──────────────────────────────────────────

def _clause_order_seed(candidate_id: str) -> int:
    """
    Return 0-3 deterministically from candidate_id.
    Same ID → same seed always. Different IDs → different seeds.
    Uses MD5 (fast, not security-sensitive).
    """
    return int(hashlib.md5(candidate_id.encode()).hexdigest(), 16) % 4


# 4 fixed clause orderings — keys match the clause-builder dict below.
# Each ordering emphasises a different primary signal first.
_ORDERINGS = [
    # 0: substance-first (most common ML-engineer narrative)
    ["substance", "exp_band", "nlp_ir", "product", "recency", "behav"],
    # 1: experience-first (leads with seniority, then substance)
    ["exp_band", "substance", "product", "nlp_ir", "recency", "behav"],
    # 2: domain-first (leads with NLP/IR — signals domain fit before tenure)
    ["nlp_ir", "substance", "exp_band", "recency", "product", "behav"],
    # 3: recency-first (leads with shipping — signals active practitioner)
    ["recency", "substance", "exp_band", "nlp_ir", "product", "behav"],
]


# ── Core reasoning generator ──────────────────────────────────────────────────

def generate_reasoning_v2(
    raw_candidate: dict,
    final_score:   float,
    role_substance: float,
    nlp_ir:         float,
    product_score:  float,
    recency_score:  float,
    beh_mult:       float,
    rank:           int = 0,
) -> str:
    """
    Grounded reasoning: every clause comes from a real feature VALUE.
    Cannot hallucinate — each sentence is a reading of a computed signal.

    Three improvements over v1:
      1. Opener varies by score band (not hardcoded "Strong fit" for all).
      2. Clause order varies deterministically by candidate_id hash.
      3. Leading positives capped at 3 for ranks > 30.

    Output format:
      "{title}, {yoe:.1f} yrs · {company}.
       {Opener}: {positive clauses}. Gaps: {gap clauses}."
    """
    p       = raw_candidate.get('profile') or {}
    title   = str(p.get('current_title') or 'Professional')
    yoe     = _f(p.get('years_of_experience') or 0)
    company = str(p.get('current_company') or '')
    cid     = str(
        raw_candidate.get('candidate_id') or
        raw_candidate.get('id') or
        raw_candidate.get('_id') or
        title  # fallback: vary on title if no ID
    )

    # ── Build all possible clauses keyed by name ──────────────────────────────
    positives: dict[str, str] = {}
    gaps:      list[str] = []

    # substance
    if role_substance >= 0.8:
        positives["substance"] = "career history shows real retrieval/ranking/applied-ML substance"
    elif role_substance >= 0.5:
        positives["substance"] = "some applied-ML substance in role descriptions"
    elif role_substance < 0.25:
        gaps.append("career descriptions show no retrieval/ranking/ML substance")
    else:
        gaps.append("limited career substance for the role")

    # exp_band
    if 6.0 <= yoe <= 8.0:
        positives["exp_band"] = "experience in the JD's ideal band"
    elif 5.0 <= yoe <= 9.0:
        positives["exp_band"] = "experience close to the target band"
    elif yoe < 4.0 or yoe > 12.0:
        gaps.append(f"experience outside the target band ({yoe:.1f} yrs)")

    # nlp_ir
    if nlp_ir >= 0.999:
        positives["nlp_ir"] = "NLP/IR background present"
    elif nlp_ir <= 0.001:
        gaps.append("CV/speech/robotics-heavy with no NLP/IR (a JD negative)")

    # product
    if product_score >= 0.999:
        positives["product"] = "product-company experience"
    elif product_score <= 0.001:
        gaps.append("career entirely at services/consulting firms")
    elif product_score < 0.5:
        gaps.append("much of the career at services/consulting firms")

    # recency
    if recency_score >= 0.999:
        positives["recency"] = "recently shipping production code"
    elif recency_score <= 0.2:
        gaps.append("in a non-coding lead role with no recent shipping")

    # behav — only when it moves the score meaningfully
    if beh_mult >= 1.03:
        positives["behav"] = "engaged and available (behaviour lifts the score)"
    elif beh_mult <= 0.75:
        gaps.append("low engagement/availability drags the score")

    # ── Determine clause order for this candidate ─────────────────────────────
    seed    = _clause_order_seed(cid)
    order   = _ORDERINGS[seed]
    ordered_positives = [positives[k] for k in order if k in positives]

    # ── Cap leading positive clauses at 3 for ranks > 30 ─────────────────────
    # This prevents mid-ranked candidates from reading identically to top ones.
    # rank=0 means pre-boost placeholder: treat conservatively (no cap).
    if rank > 30:
        ordered_positives = ordered_positives[:3]

    # ── Opener by score band ──────────────────────────────────────────────────
    if final_score >= 0.85:
        opener = "Strong fit"
    elif final_score >= 0.65:
        opener = "Solid fit"
    elif final_score >= 0.45:
        opener = "Fit"
    else:
        opener = "Marginal fit"

    # Downgrade if no positives at all
    if not ordered_positives:
        opener = "Limited fit"

    # ── Assemble output ───────────────────────────────────────────────────────
    company_str = f" \u00b7 {company}" if company else ""
    lead = f"{title}, {yoe:.1f} yrs{company_str}"

    parts = [lead]
    if ordered_positives:
        parts.append(f"{opener}: " + "; ".join(ordered_positives))
    else:
        parts.append(opener)
    if gaps:
        parts.append("Gaps: " + "; ".join(gaps[:2]))

    result = ". ".join(parts) + "."

    # Truncate at word boundary, never mid-word. Max 300 chars.
    if len(result) > 300:
        result = result[:299].rsplit(" ", 1)[0].rstrip(",.;") + "\u2026"

    return result
