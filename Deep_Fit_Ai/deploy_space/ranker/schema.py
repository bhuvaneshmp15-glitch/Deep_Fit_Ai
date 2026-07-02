"""
src/ranker/schema.py
====================
Typed dataclasses for the two data objects passed between modules.
ZERO logic — no scoring, no I/O, no config access.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class FeatureSet:
    """
    All computed feature values for one candidate, before weighting.
    All floats in [0.0, 1.0] unless noted.
    """
    # ── Career-text features (read from descriptions, not skill tags) ──────────
    role_substance:      float = 0.0   # retrieval/ranking/ML/recsys in text
    skill_corroboration: float = 0.5   # AI skill tags backed by career text
    nlp_ir_signal:       float = 0.5   # NLP/IR vs CV/speech background
    product_score:       float = 0.5   # product company vs consulting penalty
    recency_score:       float = 0.5   # currently shipping code vs non-coding lead

    # ── Structured features ───────────────────────────────────────────────────
    exp_score:           float = 0.0   # YOE + title match
    edu_score:           float = 0.3   # education tier + degree + AI field
    loc_score:           float = 0.3   # location fit

    # ── Metadata (not weighted; used by reasoning.py) ─────────────────────────
    yoe:                 float = 0.0
    current_title:       str   = ''
    ai_skill_count:      int   = 0
    matched_skills:      List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, float]:
        """Return the scoreable floats as a plain dict (used by eval/evaluate.py)."""
        return {
            'role_substance':      self.role_substance,
            'skill_corroboration': self.skill_corroboration,
            'nlp_ir_signal':       self.nlp_ir_signal,
            'product_score':       self.product_score,
            'recency_score':       self.recency_score,
            'exp_score':           self.exp_score,
            'edu_score':           self.edu_score,
            'loc_score':           self.loc_score,
        }


@dataclass
class CandidateScore:
    """
    Final output of scorer.score_candidate() for one candidate.
    Passed directly to reasoning.reasoning_for().
    """
    candidate_id:   str
    base_score:     float          # pre-multiplier weighted score
    honeypot_mult:  float          # data-quality penalty (1.0 = clean)
    final_score:    float          # base_score * behav_mult * honeypot_mult
    features:       FeatureSet     # all feature values
    behav_score:    float          # behavioral multiplier (not additive score)
    raw_candidate:  dict           # original nested JSON dict (for reasoning)
    is_stuffer:     bool = False   # True if keyword-stuffer gate fired
    stuffer_reason: str  = ''      # human-readable stuffer explanation

    def __repr__(self):
        return (
            f"CandidateScore(id={self.candidate_id!r}, "
            f"final={self.final_score:.4f}, "
            f"behav_mult={self.behav_score:.3f}, "
            f"title={self.features.current_title!r})"
        )
