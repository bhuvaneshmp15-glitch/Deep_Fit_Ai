"""
src/ranker/honeypot.py
======================
WHAT THIS MODULE DOES:
  Detects fabricated, corrupt, or ghost candidate profiles using four
  data-quality checks. Returns a penalty multiplier in (0, 1].

WHAT THIS MODULE DOES NOT DO:
  Score skills, experience, or behavioral signals. Does not read REFERENCE_DATE
  — honeypot checks are structural (not time-dependent).

WHY THIS BOUNDARY EXISTS:
  Data-quality filtering is conceptually different from ranking quality.
  Keeping it isolated means the penalty logic can be audited and adjusted
  without any risk of affecting the feature scoring math.

PATTERNS DETECTED:
  1. Salary inversion: min > max  → data corruption
  2. Fake skill profile: all advanced/expert, 0 endorsements (>70% of skills)
  3. Ghost candidate: offer_acceptance_rate=-1, high interview completion,
     zero applications, not open to work
  4. Completeness mismatch: profile_completeness < 25 but > 10 skills listed
"""

from __future__ import annotations
from typing import Tuple, List

from .features import _f, _i


def is_honeypot(raw: dict) -> Tuple[bool, List[str]]:
    """
    Inspect the raw (nested) candidate dict for data-quality red flags.

    Returns (is_flagged: bool, reasons: list[str]).
    Callers should convert is_flagged → penalty multiplier via penalty().
    """
    reasons: List[str] = []
    sig    = raw.get('redrob_signals') or {}
    skills = raw.get('skills') or []

    # 1. Salary inversion
    sal_range = sig.get('expected_salary_range_inr_lpa') or {}
    sal_min   = _f(sal_range.get('min', 0))
    sal_max   = _f(sal_range.get('max', 0))
    if sal_min > 0 and sal_max > 0 and sal_min > sal_max:
        reasons.append('salary_inversion')

    # 2. Fake skill profile: all advanced/expert with 0 endorsements
    if len(skills) >= 5:
        advanced_zero = sum(
            1 for s in skills
            if isinstance(s, dict)
            and s.get('proficiency') in ('advanced', 'expert')
            and _i(s.get('endorsements', 0)) == 0
        )
        if advanced_zero / len(skills) > 0.70:
            reasons.append('fake_skill_profile')

    # 3. Ghost candidate
    oar  = _f(sig.get('offer_acceptance_rate', 0))
    icr  = _f(sig.get('interview_completion_rate', 0))
    apps = _i(sig.get('applications_submitted_30d', 0))
    otw  = bool(sig.get('open_to_work_flag', False))
    if oar == -1 and icr > 0.90 and apps == 0 and not otw:
        reasons.append('ghost_candidate')

    # 4. Completeness vs skill count mismatch
    pc = _f(sig.get('profile_completeness_score', 100))
    if pc < 25 and len(skills) > 10:
        reasons.append('completeness_mismatch')

    return bool(reasons), reasons


def penalty(raw: dict) -> float:
    """
    Return a score multiplier in (0, 1].
    1.0 = clean profile, lower = more suspicious.
    """
    flagged, reasons = is_honeypot(raw)
    mult = 1.0
    if 'salary_inversion'      in reasons: mult *= 0.60
    if 'fake_skill_profile'    in reasons: mult *= 0.75
    if 'ghost_candidate'       in reasons: mult *= 0.80
    if 'completeness_mismatch' in reasons: mult *= 0.85
    return mult
