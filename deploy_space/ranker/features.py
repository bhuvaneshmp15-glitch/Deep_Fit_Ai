"""
src/ranker/features.py
======================
WHAT THIS MODULE DOES:
  Extracts all structured features from one raw candidate dict and
  returns a typed FeatureSet. Every function here is PURE — no I/O,
  no weights, no side effects.

WHAT THIS MODULE DOES NOT DO:
  Apply weights (scorer.py's job). Read JSONL files (io_utils.py's job).
  Generate reasoning text (reasoning.py's job).

KEY DESIGN:
  - All scoring reads career DESCRIPTIONS, not skill tags.
  - Skill tags are only used as a corroboration GATE (stuffer detection).
  - REFERENCE_DATE is fixed — never datetime.now().
"""

from __future__ import annotations

import datetime as dt
import re
from typing import List, Optional, Tuple

from .config import (
    REFERENCE_DATE,
    STRONG_TITLE_RE, ADJACENT_TITLE_RE, NONTECH_TITLE_RE,
    SUBSTANCE_AREAS, NLP_IR_RE, CV_SPEECH_RE,
    BUILD_SHIP_RE, NONCODING_LEAD_RE,
    CONSULTING_FIRMS, HARD_AI_SKILLS,
    EDU_DEGREE_SCORES, TIER1_INSTITUTES, AI_FIELDS,
    PREFERRED_LOCATIONS, JD,
)
from .schema import FeatureSet

# ── Reference date (parsed once at import) ────────────────────────────────────
_REF_DT = dt.date.fromisoformat(REFERENCE_DATE)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _f(v) -> float:
    """Safe float coercion."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _i(v) -> int:
    """Safe int coercion."""
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def _days_since(date_str: str) -> float:
    """Days from date_str to REFERENCE_DATE. Returns 365 on parse error."""
    if not date_str:
        return 365.0
    try:
        d = dt.date.fromisoformat(str(date_str)[:10])
        return max(0.0, (_REF_DT - d).days)
    except (ValueError, TypeError):
        return 365.0


def _lookup_skill(name: str) -> Tuple[str, float]:
    """Return (normalised_name, weight) for a skill name."""
    n = (name or '').strip().lower()
    if n in HARD_AI_SKILLS:
        return n, 1.0
    return n, 0.0


# ── Career text extraction ─────────────────────────────────────────────────────

def _career_text(raw: dict) -> str:
    """
    Concatenate all text where substance lives:
    headline + summary + role titles + role descriptions.
    This is the PRIMARY input to all substance-area regexes.
    """
    p     = raw.get('profile') or {}
    parts = [p.get('headline', ''), p.get('summary', '')]
    for key in ('career_history', 'experience', 'work_experience'):
        for rec in (raw.get(key) or []):
            if not isinstance(rec, dict):
                continue
            parts.append(rec.get('title', '') or '')
            for field in ('description', 'responsibilities', 'summary',
                          'highlights', 'achievements', 'details'):
                val = rec.get(field)
                if isinstance(val, str):
                    parts.append(val)
                elif isinstance(val, list):
                    parts.extend(str(v) for v in val if v)
    for key in ('about', 'bio', 'profile_summary'):
        val = raw.get(key)
        if isinstance(val, str):
            parts.append(val)
    return '  '.join(x for x in parts if x)


# ── Title classification ───────────────────────────────────────────────────────

def _title_class(title: str) -> str:
    """Classify a job title into one of: strong / adjacent / nontech / other."""
    if STRONG_TITLE_RE.search(title):   return 'strong'
    if NONTECH_TITLE_RE.search(title):  return 'nontech'
    if ADJACENT_TITLE_RE.search(title): return 'adjacent'
    return 'other'


_TITLE_SCORE = {'strong': 1.0, 'adjacent': 0.45, 'other': 0.25, 'nontech': 0.0}


# ── CHANGE 1: score_role_substance ─────────────────────────────────────────────

def score_role_substance(raw: dict) -> float:
    """
    PRIMARY scoring signal. Reads career TEXT, not skill tags.

    Formula:
      title_score  = 1.0 (ML/AI title) | 0.45 (adjacent) | 0.25 (other) | 0.0 (non-tech)
      desc_score   = 0.0 | 0.5 | 0.8 | 1.0  (based on #substance areas found)
      final        = 0.45 * title_score + 0.55 * desc_score

    Why 55% description vs 45% title:
      Title alone is gamed (anyone can call themselves 'AI Engineer').
      Description text is harder to fake at scale.
    """
    p     = raw.get('profile') or {}
    title = str(p.get('current_title') or '')
    tc    = _title_class(title)
    title_score = _TITLE_SCORE[tc]

    text   = _career_text(raw)
    n_areas = sum(1 for rx in SUBSTANCE_AREAS.values() if rx.search(text))
    desc_score = {0: 0.0, 1: 0.5, 2: 0.8}.get(n_areas, 1.0)

    return round(0.45 * title_score + 0.55 * desc_score, 4)


# ── score_nlp_ir ───────────────────────────────────────────────────────────────

def score_nlp_ir(raw: dict) -> float:
    """
    NLP/IR background vs CV/speech (explicit JD signal).
    JD says: 'primary CV/speech/robotics without NLP/IR is a negative.'
    Returns 1.0 (NLP/IR), 0.7 (mixed), 0.5 (neutral), 0.0 (CV/speech only).
    """
    text   = _career_text(raw)
    has_nlp = bool(NLP_IR_RE.search(text))
    has_cv  = bool(CV_SPEECH_RE.search(text))
    if has_nlp and not has_cv: return 1.0
    if has_nlp and has_cv:     return 0.7
    if has_cv:                 return 0.0
    return 0.5


# ── score_product_vs_consulting ────────────────────────────────────────────────

def score_product_vs_consulting(raw: dict) -> float:
    """
    Product company experience (explicit JD negative for TCS/Infosys/Wipro etc.).
    Returns 0.0 if entire career is consulting, 1.0 if no consulting firms.
    """
    career = raw.get('career_history') or raw.get('experience') or []
    if not career:
        return 0.5
    firms = CONSULTING_FIRMS
    count = sum(
        1 for r in career
        if isinstance(r, dict)
        and any(f in (r.get('company') or '').lower() for f in firms)
    )
    frac = count / len(career)
    if frac >= 0.999:
        return 0.0
    return round(1.0 - frac, 4)


# ── score_production_recency ───────────────────────────────────────────────────

def score_production_recency(raw: dict) -> float:
    """
    Is this person currently SHIPPING CODE, or in a non-coding leadership role?

    Returns:
      1.0 — current role description has 'built/deployed/shipped/...'
      0.6 — neutral (no strong signals either way)
      0.2 — currently in a non-coding lead role without shipping evidence
      0.1 — in such a role for 18+ months
    """
    career = raw.get('career_history') or raw.get('experience') or []
    cur    = [r for r in career if isinstance(r, dict) and r.get('is_current')] or career[:1]
    if not cur:
        return 0.5

    r     = cur[0]
    desc  = str(r.get('description') or '')
    title = str(r.get('title') or '')

    if NONCODING_LEAD_RE.search(title) and not BUILD_SHIP_RE.search(desc):
        try:
            sd     = dt.date.fromisoformat(str(r.get('start_date') or '')[:10])
            months = (_REF_DT.year - sd.year) * 12 + (_REF_DT.month - sd.month)
            return 0.1 if months >= 18 else 0.2
        except (ValueError, TypeError):
            return 0.2

    if BUILD_SHIP_RE.search(desc):
        return 1.0
    return 0.6


# ── score_skill_corroboration ─────────────────────────────────────────────────

def score_skill_corroboration(raw: dict) -> float:
    """
    Gate: many AI skill tags with zero career substance = keyword stuffer.

    Logic:
      - Fewer than 4 AI skills in tags -> no gate needed -> return 1.0
      - 4+ AI skills -> count how many substance areas appear in career text
      - corroboration = min(1.0, n_areas / 2.0)
      - Returns 0.0 when 4+ AI skills claimed but ZERO substance in descriptions

    HARD_AI_SKILLS is a frozenset of lowercase skill names.
    """
    skills   = raw.get('skills') or []
    ai_count = sum(
        1 for s in skills
        if isinstance(s, dict) and (s.get('name') or '').lower() in HARD_AI_SKILLS
    )
    if ai_count < 4:
        return 1.0  # not enough AI tags to trigger gate

    text    = _career_text(raw)
    n_areas = sum(1 for rx in SUBSTANCE_AREAS.values() if rx.search(text))
    return round(min(1.0, n_areas / 2.0), 4)


# ── corroboration_ratio ────────────────────────────────────────────────────────

# Root-word extraction: map each skill name to 1-2 searchable root words
_SKILL_ROOTS = {
    'machine learning':          ['machine learning', 'ml model', 'trained model'],
    'deep learning':             ['deep learning', 'neural network', 'pytorch', 'tensorflow'],
    'nlp':                       ['nlp', 'natural language', 'text classification', 'tokeniz'],
    'natural language processing':['nlp', 'natural language', 'language model'],
    'llm':                       ['llm', 'language model', 'gpt', 'bert'],
    'transformers':              ['transformer', 'bert', 'attention', 'hugging'],
    'computer vision':           ['computer vision', 'image', 'cnn', 'object detect'],
    'pytorch':                   ['pytorch', 'torch'],
    'tensorflow':                ['tensorflow', 'keras', 'tf.'],
    'hugging face':              ['hugging face', 'huggingface', 'transformers'],
    'bert':                      ['bert', 'language model', 'fine-tun'],
    'gpt':                       ['gpt', 'language model', 'llm'],
    'information retrieval':     ['retrieval', 'search', 'ranking', 'index'],
    'recommender systems':       ['recommend', 'recsys', 'collaborative', 'personaliz'],
    'semantic search':           ['semantic search', 'embedding', 'vector search'],
    'embeddings':                ['embedding', 'vector', 'sentence transformer'],
    'neural networks':           ['neural', 'deep learning', 'backprop'],
    'scikit-learn':              ['scikit', 'sklearn', 'classification', 'regression'],
    'mlops':                     ['mlops', 'ml pipeline', 'model serving', 'deploy'],
    'sentence transformers':     ['sentence transformer', 'embedding', 'semantic'],
    'faiss':                     ['faiss', 'vector search', 'nearest neighbor', 'ann'],
    'qdrant':                    ['qdrant', 'vector', 'embedding store'],
    'weaviate':                  ['weaviate', 'vector', 'embedding'],
    'pinecone':                  ['pinecone', 'vector', 'embedding'],
    'llm fine-tuning':           ['fine-tun', 'lora', 'rlhf', 'instruction tun'],
    'rag':                       ['rag', 'retrieval augment', 'retrieval-augment'],
    'feature engineering':       ['feature engineering', 'feature extract', 'feature select'],
    'xgboost':                   ['xgboost', 'gradient boost', 'boosting'],
    'lightgbm':                  ['lightgbm', 'gradient boost', 'boosting'],
}


def corroboration_ratio(raw: dict) -> float:
    """
    Per-skill corroboration check: does each claimed AI skill have evidence
    in the candidate's own career text?

    Algorithm:
      1. Extract AI/ML skills from raw['skills']
      2. Build career text from headline + summary + career_history descriptions
      3. For each claimed AI skill, look up its root words and check if ANY
         appear in the career text
      4. Return corroborated_count / total_ai_claimed  (float 0.0–1.0)

    Hard gates (return 0.0 immediately):
      - NONTECH_TITLE_RE matches current title AND 2+ AI skills are claimed
        → clear non-tech title keyword-stuffing

    Neutral gate (return 1.0):
      - Fewer than 3 AI skills claimed → not enough evidence to judge

    Scoring:
      0.0 = every claimed AI skill has zero career-text evidence
      0.5 = half the claimed skills have evidence
      1.0 = all claimed skills corroborated OR fewer than 3 claimed
    """
    p     = raw.get('profile') or {}
    title = str(p.get('current_title') or '')

    # Collect claimed AI skills (lowercased)
    skills    = raw.get('skills') or []
    ai_skills = [
        (s.get('name') or '').strip()
        for s in skills
        if isinstance(s, dict) and (s.get('name') or '').lower() in HARD_AI_SKILLS
    ]

    # ── Hard gate 1: non-tech title + 2+ AI skills = definite stuffer ─────────
    if NONTECH_TITLE_RE.search(title) and len(ai_skills) >= 2:
        return 0.0

    # ── Neutral gate: too few claims to evaluate ───────────────────────────────
    if len(ai_skills) < 3:
        return 1.0

    career_text = _career_text(raw).lower()
    if not career_text:
        # No career text at all — can't corroborate anything
        return 0.0 if len(ai_skills) >= 4 else 0.5

    corroborated = 0
    for skill_name in ai_skills:
        roots = _SKILL_ROOTS.get(skill_name.lower(), [skill_name.lower()[:6]])
        if any(root in career_text for root in roots):
            corroborated += 1

    return round(corroborated / len(ai_skills), 4)


# ── score_experience ───────────────────────────────────────────────────────────

def score_experience(raw: dict) -> Tuple[float, float, str]:
    """
    Score YOE against the JD target band (5–9 yrs).
    Returns (exp_score, yoe, current_title).
    """
    p     = raw.get('profile') or {}
    sig   = raw.get('redrob_signals') or {}
    title = str(p.get('current_title') or '')

    yoe = _f(
        p.get('years_of_experience')
        or p.get('total_experience_years')
        or sig.get('years_of_experience')
        or 0
    )

    # Infer from career history if missing
    if yoe == 0.0:
        yoe = _infer_yoe(raw)

    target_min = JD['target_yoe_min']
    target_max = JD['target_yoe_max']

    if target_min <= yoe <= target_max:
        score = 1.0
    elif yoe < target_min:
        score = max(0.0, yoe / target_min)
    else:
        over  = yoe - target_max
        score = max(0.0, 1.0 - over / 6.0)

    # Title multiplier
    tc = _title_class(title)
    title_mult = {'strong': 1.0, 'adjacent': 0.85, 'other': 0.70, 'nontech': 0.40}[tc]
    return round(score * title_mult, 4), yoe, title


def _infer_yoe(raw: dict) -> float:
    career = raw.get('career_history') or raw.get('experience') or []
    total_months = 0
    for r in career:
        if not isinstance(r, dict):
            continue
        try:
            sd = dt.date.fromisoformat(str(r.get('start_date') or '')[:10])
            ed_raw = r.get('end_date') or ''
            ed = _REF_DT if r.get('is_current') or not ed_raw else dt.date.fromisoformat(str(ed_raw)[:10])
            total_months += max(0, (ed.year - sd.year) * 12 + (ed.month - sd.month))
        except (ValueError, TypeError):
            pass
    return round(total_months / 12.0, 1)


# ── score_education ────────────────────────────────────────────────────────────

def score_education(raw: dict) -> float:
    """Score education tier, degree level, and AI field alignment."""
    edu_list = raw.get('education') or []
    if not edu_list:
        return 0.3

    best = 0.0
    for edu in edu_list:
        if not isinstance(edu, dict):
            continue
        degree  = str(edu.get('degree') or edu.get('qualification') or '').lower()
        inst    = str(edu.get('institution') or edu.get('school') or '').lower()
        field   = str(edu.get('field_of_study') or edu.get('major') or '').lower()
        tier    = str(edu.get('tier') or '').lower()

        d_score = max(
            (v for k, v in EDU_DEGREE_SCORES.items() if k in degree),
            default=0.5,
        )
        t_score = 1.0 if tier == 'tier_1' else (0.7 if tier == 'tier_2' else 0.5)
        if any(kw in inst for kw in TIER1_INSTITUTES):
            t_score = max(t_score, 0.9)
        f_score = 1.0 if any(f in field for f in AI_FIELDS) else 0.6

        edu_score = 0.4 * d_score + 0.35 * t_score + 0.25 * f_score
        best = max(best, edu_score)

    return round(best, 4)


# ── score_location ─────────────────────────────────────────────────────────────

def score_location(raw: dict) -> float:
    """Score location fit against JD preferred cities."""
    p    = raw.get('profile') or {}
    loc  = str(p.get('location') or raw.get('location') or '').lower()
    city = loc.split(',')[0].strip()
    return PREFERRED_LOCATIONS.get(city, PREFERRED_LOCATIONS.get(loc, 0.3))


# ── flatten_candidate ─────────────────────────────────────────────────────────

def flatten_candidate(raw: dict) -> dict:
    """
    Produce a flat dict of all signals needed by behavioral.py.
    Normalises nested redrob_signals fields to top-level keys.
    """
    p   = raw.get('profile') or {}
    sig = raw.get('redrob_signals') or {}

    flat = {
        # Profile
        'current_title':            p.get('current_title', ''),
        'years_of_experience':      _f(p.get('years_of_experience') or 0),
        'location':                 p.get('location', ''),
        'skills':                   raw.get('skills') or [],

        # Behavioral signals
        'last_active_date':         sig.get('last_active_date', ''),
        'open_to_work':             bool(sig.get('open_to_work_flag', False)),
        'recruiter_response_rate':  _f(sig.get('recruiter_response_rate') or 0.44),
        'interview_completion_rate': _f(sig.get('interview_completion_rate') or 0.5),
        'notice_period_days':       _i(sig.get('notice_period_days') or 30),
        'github_score':             _f(sig.get('github_activity_score') or -1),
        'saved_by_recruiters':      _f(sig.get('saved_by_recruiters_30d') or 0),
        'profile_completeness':     _f(sig.get('profile_completeness_score') or 70),
        'verified_email':           bool(sig.get('verified_email', False)),
        'verified_phone':           bool(sig.get('verified_phone', False)),
        'search_appearance_score':  min(1.0, _f(sig.get('search_appearance_30d') or 0) / 300.0),
        'connection_count':         _f(sig.get('connection_count') or 0),
        'linkedin_connected':       bool(sig.get('linkedin_connected', False)),

        # Keep nested for honeypot.py
        'redrob_signals':           sig,
    }
    return flat


# ── structured_features (main entry point) ────────────────────────────────────

def structured_features(raw: dict) -> FeatureSet:
    """
    Extract all features from one raw candidate dict.
    Returns a populated FeatureSet. This is the ONLY function scorer.py calls.
    """
    p     = raw.get('profile') or {}
    title = str(p.get('current_title') or '')

    # Count AI skill tags (for corroboration gate metadata)
    skills   = raw.get('skills') or []
    ai_count = sum(
        1 for s in skills
        if isinstance(s, dict) and (s.get('name') or '').lower() in HARD_AI_SKILLS
    )
    matched = [
        s.get('name', '') for s in skills
        if isinstance(s, dict) and (s.get('name') or '').lower() in HARD_AI_SKILLS
    ][:6]

    exp_score, yoe, _ = score_experience(raw)

    return FeatureSet(
        role_substance      = score_role_substance(raw),
        skill_corroboration = score_skill_corroboration(raw),
        nlp_ir_signal       = score_nlp_ir(raw),
        product_score       = score_product_vs_consulting(raw),
        recency_score       = score_production_recency(raw),
        exp_score           = exp_score,
        edu_score           = score_education(raw),
        loc_score           = score_location(raw),
        yoe                 = yoe,
        current_title       = title,
        ai_skill_count      = ai_count,
        matched_skills      = matched,
    )
