"""
src/ranker/scorer.py
====================
WHAT THIS MODULE DOES:
  Orchestrates the full scoring pipeline for one candidate.
  Calls features → behavioral_multiplier → honeypot → applies weights.

KEY CHANGE from old scorer:
  Behavioral is now a MULTIPLIER [0.50, 1.15], not an additive weight.
  Pipeline:  base = weighted_sum(features)
             final = base * behav_mult * honeypot_mult

  This prevents platform signals from inflating unqualified candidates.
  A Sales Executive with perfect behavioral signals still scores near 0
  because base = 0 (role_substance=0) and anything × 0 = 0.

WHAT THIS MODULE DOES NOT DO:
  Compute individual features (features.py). Read/write files (io_utils.py).
  Generate reasoning text (reasoning.py).
"""

from __future__ import annotations

from .config import WEIGHTS
from .features import structured_features, flatten_candidate, corroboration_ratio
from .behavioral import behavioral_multiplier
from .honeypot import penalty as honeypot_penalty
from .schema import CandidateScore, FeatureSet


def score_candidate(raw: dict) -> CandidateScore:
    """
    Full scoring pipeline for one raw candidate dict.

    Steps:
      1. Extract structured features from career descriptions + profile
      2a. Stuffer gate: skill_corroboration == 0.0 -> floor to 0.02
      2b. Title gate: corroboration_ratio == 0.0 (non-tech title + AI skills
          with no career-text evidence) -> floor to 0.02
      3. Weighted base score (behavioral is NOT in this sum)
      4. Apply corroboration_ratio as a multiplier (penalises partial stuffers)
      5. Apply behavioral multiplier [BEHAV_FLOOR, BEHAV_CAP]
      6. Apply honeypot penalty (hard floor to 0.01 for severe honeypots)
      7. Return typed CandidateScore
    """
    cid = (
        raw.get('candidate_id') or
        raw.get('id') or
        raw.get('_id') or
        'unknown'
    )

    # ── 1. Structured features ────────────────────────────────────────────────
    features = structured_features(raw)

    # ── 2a. Coarse stuffer gate (substance area check) ────────────────────────
    if features.skill_corroboration == 0.0:
        p         = raw.get('profile') or {}
        title     = p.get('current_title') or 'Unknown'
        ai_count  = features.ai_skill_count
        reason    = (
            f"Keyword-stuffing signal: {ai_count} AI skills claimed by '{title}', "
            "none corroborated by career history — gated to score floor."
        )
        return CandidateScore(
            candidate_id   = cid,
            base_score     = 0.02,
            honeypot_mult  = 1.0,
            final_score    = 0.02,
            features       = features,
            behav_score    = 1.0,
            raw_candidate  = raw,
            is_stuffer     = True,
            stuffer_reason = reason,
        )

    # ── 2b. Fine-grained title + per-skill corroboration gate ─────────────────
    corrob = corroboration_ratio(raw)
    if corrob == 0.0:
        p        = raw.get('profile') or {}
        title    = p.get('current_title') or 'Unknown'
        ai_count = features.ai_skill_count
        reason   = (
            f"Title gate: '{title}' is a non-tech role; "
            f"{ai_count} AI skills listed with zero career-text corroboration."
        )
        return CandidateScore(
            candidate_id   = cid,
            base_score     = 0.02,
            honeypot_mult  = 1.0,
            final_score    = 0.02,
            features       = features,
            behav_score    = 1.0,
            raw_candidate  = raw,
            is_stuffer     = True,
            stuffer_reason = reason,
        )

    # ── 3. Weighted base score (no behavioral term) ───────────────────────────
    W    = WEIGHTS
    base = (
        W['role_substance']      * features.role_substance
        + W['skill_corroboration'] * features.skill_corroboration
        + W['exp_score']           * features.exp_score
        + W['nlp_ir_signal']       * features.nlp_ir_signal
        + W['product_score']       * features.product_score
        + W['recency_score']       * features.recency_score
        + W['edu_score']           * features.edu_score
        + W['loc_score']           * features.loc_score
    )

    # ── 4. Corroboration multiplier (partial stuffers get proportional penalty)
    # corrob in (0, 1] here — 0.0 was handled by the gate above.
    # A candidate with half their AI skills corroborated gets ×0.5 on base.
    score = base * corrob

    # ── 5. Behavioral multiplier ──────────────────────────────────────────────
    behav_mult = behavioral_multiplier(raw)
    score      = score * behav_mult

    # ── 6. Honeypot penalty ───────────────────────────────────────────────────
    hp = honeypot_penalty(raw)
    if hp < 0.6:
        final = 0.01   # hard floor for severe honeypots
    else:
        final = score * hp

    return CandidateScore(
        candidate_id  = cid,
        base_score    = base,
        honeypot_mult = hp,
        final_score   = final,
        features      = features,
        behav_score   = behav_mult,
        raw_candidate = raw,
    )


def combine_features(feat_dict: dict, behav: float, weights: dict) -> float:
    """
    Apply a weight configuration to a feature dict + behavioral value.
    Used by eval/evaluate.py for weight sweeping.

    NOTE: behav here is the RAW behavioral_multiplier value [0.50, 1.15].
    combine_features treats it as an additive feature (normalised to [0,1])
    for the purposes of the sweep. This is an approximation acceptable for
    offline evaluation.

    feat_dict : output of FeatureSet.as_dict()
    behav     : behavioral_multiplier() value  (used normalised)
    weights   : dict of weight names → floats (auto-normalised)
    """
    W       = weights
    total_w = sum(W.values()) or 1.0
    # Normalise behav multiplier [0.50,1.15] → [0,1] for additive comparison
    behav_norm = max(0.0, min(1.0, (behav - 0.50) / 0.65))
    raw = (
        W.get('role_substance',      0) * feat_dict.get('role_substance',      0)
        + W.get('skill_corroboration', 0) * feat_dict.get('skill_corroboration', 0)
        + W.get('exp_score',           0) * feat_dict.get('exp_score',           0)
        + W.get('behav_score',         0) * behav_norm
        + W.get('nlp_ir_signal',       0) * feat_dict.get('nlp_ir_signal',       0)
        + W.get('product_score',       0) * feat_dict.get('product_score',       0)
        + W.get('recency_score',       0) * feat_dict.get('recency_score',       0)
        + W.get('edu_score',           0) * feat_dict.get('edu_score',           0)
        + W.get('loc_score',           0) * feat_dict.get('loc_score',           0)
    )
    return raw / total_w
