"""
src/ranker/behavioral.py
========================
WHAT THIS MODULE DOES:
  Computes a behavioral MULTIPLIER in [FLOOR, CAP] from redrob_signals.
  The multiplier is applied as: final_score = base_score * behavioral_mult.

WHAT THIS MODULE DOES NOT DO:
  Add to the base score (old additive design was wrong — it let platform
  signals dominate substance). The multiplier can only boost by 15% or
  penalise by 50% — it MODULATES substance, never replaces it.

KEY CHANGE vs old behavioral.py:
  Old: behavioral_score() returned a [0,1] float used as a WEIGHTED ADDITIVE term
  New: behavioral_multiplier() returns a multiplier in [0.50, 1.15]

FLOOR = 0.50: worst-case unavailable candidate penalised 50%
CAP   = 1.15: best-case engaged candidate lifted 15%
"""

from __future__ import annotations

import datetime as dt

from .config import REFERENCE_DATE, BEHAV_FLOOR, BEHAV_CAP

_REF = dt.date.fromisoformat(REFERENCE_DATE)


def behavioral_multiplier(raw: dict) -> float:
    """
    Returns a multiplier in [BEHAV_FLOOR, BEHAV_CAP].
    Applied as: final = base * multiplier.

    Delta accumulation (starts at 0.0, added to 1.0 at end):
      +0.04  recency <= 3 months (proportional)
      -0.20  inactive 3–12+ months (proportional)
      +0.03  recruiter response rate > median 0.44
      -0.12  recruiter response rate < median
      +0.02  interview completion > 0.5
      -0.10  interview completion < 0.5
      -0.08  notice period > 30d (proportional, capped at 90d over)
      +0.02  open to work flag
      +0.02  saved by recruiters (proportional, 5 saves = max)
      +0.02  profile completeness > 70% (proportional)
      -0.03  profile completeness < 30%
      +0.01  verified email
      +0.01  verified phone
      +0.02  github_activity_score > 25 (proportional)
      -0.02  github_activity_score 0-25 (proportional)
      (no penalty for github=-1 / not linked — absence is neutral)

    Max possible delta: +0.04+0.03+0.02+0.02+0.02+0.02+0.01+0.01+0.02 = +0.19
    Clamped to CAP=1.15 (+0.15 max).
    Min possible delta: -0.20-0.12-0.10-0.08-0.03 = -0.53
    Clamped to FLOOR=0.50 (-0.50 max).
    """
    sig   = raw.get('redrob_signals') or {}
    delta = 0.0

    # 1. Recency — days since last active vs fixed reference date
    last_raw = sig.get('last_active_date') or sig.get('last_seen') or ''
    try:
        last   = dt.date.fromisoformat(str(last_raw)[:10])
        months = (_REF - last).days / 30.44
        if months <= 3:
            delta += 0.04 * max(0.0, (3.0 - months) / 3.0)
        else:
            delta -= 0.20 * min(1.0, (months - 3.0) / 9.0)
    except (ValueError, TypeError):
        pass

    # 2. Recruiter response rate (median 0.44 is neutral)
    rr = sig.get('recruiter_response_rate')
    if rr is not None and rr >= 0:
        if rr > 0.44:
            delta += 0.03 * ((rr - 0.44) / 0.56)
        else:
            delta -= 0.12 * ((0.44 - rr) / 0.44)

    # 3. Interview completion rate (0.5 is neutral)
    ic = sig.get('interview_completion_rate')
    if ic is not None and ic >= 0:
        if ic > 0.5:
            delta += 0.02 * ((ic - 0.5) / 0.5)
        else:
            delta -= 0.10 * ((0.5 - ic) / 0.5)

    # 4. Notice period — penalise only (longer = worse)
    notice = sig.get('notice_period_days') or 30
    if notice > 30:
        delta -= 0.08 * min(1.0, (notice - 30) / 90.0)

    # 5. Open to work — small lift
    if sig.get('open_to_work_flag'):
        delta += 0.02

    # 6. Saved by recruiters — external validation signal
    saved = min(1.0, _f(sig.get('saved_by_recruiters_30d') or 0) / 5.0)
    delta += 0.02 * saved

    # 7. Profile completeness
    pc = _f(sig.get('profile_completeness_score') or 70) / 100.0
    if pc > 0.7:
        delta += 0.02 * ((pc - 0.7) / 0.3)
    elif pc < 0.3:
        delta -= 0.03 * ((0.3 - pc) / 0.3)

    # 8. Verified contact
    if sig.get('verified_email'):  delta += 0.01
    if sig.get('verified_phone'):  delta += 0.01

    # 9. GitHub activity (gh=-1 means not linked → NEUTRAL, no penalty)
    gh = sig.get('github_activity_score')
    if gh is not None and gh >= 0:
        if gh > 25:
            delta += 0.02 * min(1.0, (gh - 25) / 45.0)
        else:
            delta -= 0.02 * (1.0 - gh / 25.0)

    return max(BEHAV_FLOOR, min(BEHAV_CAP, 1.0 + delta))


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
