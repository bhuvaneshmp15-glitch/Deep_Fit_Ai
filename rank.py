#!/usr/bin/env python3
"""
rank.py â€” Redrob Hackathon Candidate Ranker
============================================
Self-contained: NO external services, NO API calls, NO network.
Standard library only: json, csv, datetime, pathlib, heapq, re, sys, time.

Scoring formula (weights sum = 1.0):
  final = 0.35*skill + 0.25*exp + 0.20*behavioral + 0.10*edu + 0.10*location

Performance:
  - Streams candidates.jsonl line-by-line (constant memory regardless of size)
  - Min-heap of size 100 tracks top candidates in O(n log 100) â‰ˆ O(n)
  - Processes 100k candidates in ~60-90s on a single CPU core

Usage:
    python rank.py [candidates.jsonl] [submission.csv]
    python rank.py                           # uses default filenames
"""

import csv
import heapq
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1 â€” JD CONFIG  (pre-computed from job_description.docx, hardcoded)
#
# Role: Recommendation Systems / Semantic Search ML Engineer
# Company: Redrob (AI hiring platform)
# Location: Pune / Noida preferred, India acceptable
# Experience: 5â€“9 years
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
JD = {
    "must_have_skills": [
        "Sentence Transformers", "FAISS", "Information Retrieval",
        "Embeddings", "Vector Search", "Python", "Machine Learning", "PyTorch",
    ],
    "nice_to_have_skills": [
        "Pinecone", "Qdrant", "Weaviate", "BM25", "MLflow",
        "Recommendation Systems", "NLP", "Feature Engineering",
        "Fine-tuning LLMs", "MLOps", "LangChain", "PEFT",
        "Weights & Biases", "Hugging Face Transformers",
    ],
    "target_yoe_min": 5,
    "target_yoe_max": 9,
    "target_locations": ["pune", "noida"],
    "india_preferred": True,
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2A â€” SCORING DICTIONARIES
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# AI skill name â†’ base weight (copied exactly from spec)
HARD_AI_SKILLS = {
    'Sentence Transformers': 1.0, 'FAISS': 1.0, 'Information Retrieval': 1.0,
    'Embeddings': 1.0, 'Pinecone': 1.0, 'Qdrant': 1.0, 'Weaviate': 1.0,
    'Vector Search': 1.0, 'BM25': 1.0, 'MLflow': 0.9,
    'Recommendation Systems': 0.9, 'scikit-learn': 0.85, 'PyTorch': 0.85,
    'Hugging Face Transformers': 0.85, 'Machine Learning': 0.85,
    'NLP': 0.85, 'Feature Engineering': 0.8, 'Fine-tuning LLMs': 0.8,
    'MLOps': 0.8, 'LangChain': 0.7, 'Data Science': 0.7,
    'Kubeflow': 0.7, 'Prompt Engineering': 0.7, 'PEFT': 0.75,
    'Weights & Biases': 0.75, 'Computer Vision': 0.65,
    'Statistical Modeling': 0.65, 'Python': 0.8, 'LoRA': 0.7,
    'TensorFlow': 0.75, 'Speech Recognition': 0.65, 'TTS': 0.6,
    'BentoML': 0.65, 'GANs': 0.6, 'CNN': 0.6, 'OpenCV': 0.6,
    'YOLO': 0.6, 'Image Classification': 0.6, 'Object Detection': 0.6,
}

# Lowercase lookup for flexible matching (built once at startup)
_HARD_AI_SKILLS_LOWER = {k.lower(): (k, v) for k, v in HARD_AI_SKILLS.items()}

# Job title â†’ relevance score (copied exactly from spec)
AI_TITLES = {
    'Recommendation Systems Engineer': 1.0,
    'Senior Machine Learning Engineer': 0.95, 'ML Engineer': 0.95,
    'Applied ML Engineer': 0.95, 'NLP Engineer': 0.9,
    'Search Engineer': 0.9, 'AI Engineer': 0.85,
    'Junior ML Engineer': 0.65, 'Data Scientist': 0.75,
    'Senior Software Engineer (ML)': 0.85,
}
_AI_TITLES_LOWER = {k.lower(): v for k, v in AI_TITLES.items()}

PROFICIENCY_MAP = {
    'beginner': 0.4, 'basic': 0.4,
    'intermediate': 0.7, 'medium': 0.7,
    'advanced': 0.9,
    'expert': 1.0, 'master': 1.0,
}

TIER_MAP = {
    'tier_1': 1.0, 'tier1': 1.0, 'tier 1': 1.0,
    'tier_2': 0.75, 'tier2': 0.75, 'tier 2': 0.75,
    'tier_3': 0.5,  'tier3': 0.5,  'tier 3': 0.5,
    'tier_4': 0.25, 'tier4': 0.25, 'tier 4': 0.25,
    'unknown': 0.3,
}

DEGREE_MAP = {
    'ph.d': 1.0, 'phd': 1.0, 'ph. d': 1.0, 'doctorate': 1.0,
    'm.tech': 0.9, 'mtech': 0.9, 'm. tech': 0.9,
    'm.s': 0.9, 'ms': 0.9, 'm.s.': 0.9, 'master of science': 0.9,
    'm.e': 0.85, 'me': 0.85, 'm.e.': 0.85, 'master of engineering': 0.85,
    'm.sc': 0.8, 'msc': 0.8, 'm.sc.': 0.8, 'master of science': 0.8,
    'mba': 0.75, 'm.b.a': 0.75,
    'b.tech': 0.7, 'btech': 0.7, 'b. tech': 0.7, 'bachelor of technology': 0.7,
    'b.e': 0.7, 'be': 0.7, 'b.e.': 0.7, 'bachelor of engineering': 0.7,
    'b.sc': 0.6, 'bsc': 0.6, 'bachelor of science': 0.6,
}

AI_FIELD_KEYWORDS = (
    'computer', 'ml', ' ai', 'data', 'information', 'machine', 'neural',
    'intelligence', 'analytics', 'statistics', 'mathematics',
)

# Well-known Indian cities (for location scoring)
_INDIA_CITIES = frozenset({
    'pune', 'noida', 'mumbai', 'delhi', 'bangalore', 'bengaluru',
    'hyderabad', 'chennai', 'kolkata', 'ahmedabad', 'jaipur', 'surat',
    'lucknow', 'kanpur', 'nagpur', 'indore', 'thane', 'bhopal',
    'visakhapatnam', 'patna', 'vadodara', 'ghaziabad', 'ludhiana',
    'agra', 'nashik', 'faridabad', 'meerut', 'rajkot', 'varanasi',
    'aurangabad', 'ranchi', 'coimbatore', 'gwalior', 'vijayawada',
    'jodhpur', 'madurai', 'raipur', 'kota', 'gurgaon', 'gurugram',
    'chandigarh', 'mysore', 'mysuru', 'bhubaneswar', 'kochi', 'cochin',
    'navi mumbai', 'allahabad', 'prayagraj', 'srinagar', 'trivandrum',
    'thiruvananthapuram', 'mangalore', 'hubli', 'dharwad', 'jabalpur',
    'amritsar', 'howrah', 'dhanbad', 'pimpri', 'chinchwad', 'navi',
    'greater noida', 'new delhi', 'ncr', 'delhi ncr',
})

# Roles that count as AI trajectory
_AI_ROLE_KEYWORDS = frozenset({
    'machine learning', 'ml ', ' ml', 'data science', 'data scientist',
    'nlp', 'natural language', 'computer vision', 'deep learning',
    'ai ', ' ai', 'neural', 'recommendation', 'search engineer',
    'transformer', 'embedding', 'mlops', 'applied scientist',
    'research scientist', 'research engineer',
})

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# UTILITY HELPERS (all pure functions, no I/O)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_NOW = datetime.now(timezone.utc)


def _f(v, default=0.0):
    """Safe float conversion."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(v, default=0):
    """Safe int conversion."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _norm(s):
    """Lowercase + strip."""
    return str(s).lower().strip() if s else ''


def _parse_date(s):
    """Parse a date string into an aware datetime or None."""
    if not s:
        return None
    s = str(s)[:10]
    for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y', '%Y-%m', '%m/%Y', '%Y'):
        try:
            dt = datetime.strptime(s[:len(fmt)], fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _days_since(date_str):
    """Days since a date string. Returns 9999 on parse failure."""
    dt = _parse_date(date_str)
    return max(0, (_NOW - dt).days) if dt else 9999


def _lookup_skill(name):
    """
    Return (canonical_name, base_weight) for a skill name.
    Tries exact match first, then case-insensitive.
    Returns (name, 0.0) if not found.
    """
    if name in HARD_AI_SKILLS:
        return name, HARD_AI_SKILLS[name]
    low = _norm(name)
    entry = _HARD_AI_SKILLS_LOWER.get(low)
    return (entry[0], entry[1]) if entry else (name, 0.0)


def _match_title(title):
    """
    Map a job title to a relevance score [0.2, 1.0].
    Tries exact â†’ case-insensitive â†’ keyword substring.
    """
    if not title:
        return 0.2
    low = _norm(title)

    # Exact (case-insensitive)
    if low in _AI_TITLES_LOWER:
        return _AI_TITLES_LOWER[low]

    # Keyword-based fallback
    if any(kw in low for kw in ('recommendation system', 'recsys', 'rec sys')):
        return 1.0
    if any(kw in low for kw in ('senior machine learning', 'sr machine learning',
                                  'senior ml', 'sr ml', 'applied ml', 'ml engineer')):
        return 0.95
    if any(kw in low for kw in ('nlp engineer', 'natural language processing engineer')):
        return 0.9
    if 'search engineer' in low:
        return 0.9
    if any(kw in low for kw in ('ai engineer', 'applied ai', 'applied scientist')):
        return 0.85
    if any(kw in low for kw in ('senior software engineer', 'staff ml', 'principal ml')):
        return 0.85
    if any(kw in low for kw in ('machine learning', 'deep learning', 'research scientist')):
        return 0.8
    if 'junior ml' in low:
        return 0.65
    if any(kw in low for kw in ('data scientist', 'data science')):
        return 0.75
    if any(kw in low for kw in ('mlops', 'ml ops', 'ml platform')):
        return 0.75
    if any(kw in low for kw in ('data engineer', 'ml infra')):
        return 0.5
    if any(kw in low for kw in ('software engineer', 'developer', 'sde', 'swe')):
        return 0.3
    return 0.2


def _is_ai_role(title):
    """True if a title string is an AI/ML role."""
    low = ' ' + _norm(title) + ' '
    return any(kw in low for kw in _AI_ROLE_KEYWORDS)


def _tier_score(raw):
    """Map institution tier string to [0.25, 1.0]."""
    low = _norm(raw)
    if low in TIER_MAP:
        return TIER_MAP[low]
    # Known IIT/IISC/IIM patterns â†’ tier_1
    if any(kw in low for kw in ('iit', 'iisc', 'iim', 'iiser', 'bits pilani', 'nid', 'iiit hyderabad')):
        return 1.0
    # NIT, DTU, NSIT â†’ tier_2
    if any(kw in low for kw in ('nit ', 'dtu', 'nsit', 'iiit', 'isi ', 'cmi', 'bits')):
        return 0.75
    # State universities â†’ tier_3
    return 0.5


def _degree_score(raw):
    """Map degree string to [0.5, 1.0]."""
    low = _norm(raw)
    best = 0.5
    for pattern, val in DEGREE_MAP.items():
        if pattern in low:
            best = max(best, val)
    return best


def _is_ai_field(raw):
    """True if field of study is AI/CS related."""
    low = _norm(raw)
    return any(kw in low for kw in AI_FIELD_KEYWORDS)


def _location_score(candidate):
    """
    Score:
      pune / noida exact   â†’ 1.0
      other india + willing â†’ 0.85
      other india, not      â†’ 0.70
      international + willingâ†’ 0.45
      international          â†’ 0.30
    """
    loc = _norm(
        candidate.get('location') or
        candidate.get('current_location') or
        candidate.get('city') or ''
    )
    willing = bool(
        candidate.get('willing_to_relocate') or
        candidate.get('open_to_relocation') or False
    )

    # Priority match: target cities
    for t in JD['target_locations']:
        if t in loc:
            return 1.0

    # India cities
    if any(city in loc for city in _INDIA_CITIES) or 'india' in loc or ', in' in loc:
        return 0.85 if willing else 0.70

    # Catch remaining India by country code
    if loc.endswith(' in') or loc == 'in':
        return 0.75

    # International with willingness
    return 0.45 if willing else 0.30


def _yoe_score(yoe):
    """Map years-of-experience to a score per spec."""
    if 5.0 <= yoe <= 9.0:
        return 1.0
    if 4.0 <= yoe < 5.0:
        return 0.85
    if 9.0 < yoe <= 12.0:
        return 0.80
    if 3.0 <= yoe < 4.0:
        return 0.65
    if yoe > 12.0:
        return 0.60
    return 0.30   # < 3 years


def _calc_yoe_from_experience(experience):
    """
    Fall-back YOE calculation: sum experience durations.
    Overlapping periods are not de-duped for speed; result is approximate.
    """
    total = 0.0
    for exp in experience:
        start = _parse_date(exp.get('start_date') or exp.get('start'))
        end_raw = str(exp.get('end_date') or exp.get('end') or '').lower()
        end = _NOW if end_raw in ('present', 'current', '', 'none') else (_parse_date(end_raw) or _NOW)
        if start and end > start:
            total += (end - start).days / 365.25
    return total


def _notice_score(days):
    """Lower notice period â†’ higher score."""
    if days <= 0:
        return 1.0
    if days <= 15:
        return 0.95
    if days <= 30:
        return 0.85
    if days <= 60:
        return 0.70
    if days <= 90:
        return 0.50
    return 0.30


def _github_score(candidate):
    """Normalize GitHub signal to [0, 1]."""
    raw = (
        candidate.get('github_score') or
        candidate.get('github_contributions') or
        candidate.get('github_stars') or 0
    )
    v = _f(raw)
    # If value looks like it's already 0-1
    if 0.0 <= v <= 1.0 and isinstance(raw, float):
        return v
    # Otherwise normalize: 100+ = 1.0
    return min(1.0, v / 100.0)


def _verified_avg(candidate):
    """Parse the verified field into a 0-1 score."""
    raw = candidate.get('verified') or candidate.get('verifications') or []
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    if isinstance(raw, (int, float)):
        v = float(raw)
        return v if v <= 1.0 else min(1.0, v / 100.0)
    if isinstance(raw, list) and raw:
        hits = 0
        for item in raw:
            if isinstance(item, dict):
                st = _norm(item.get('status') or item.get('verified') or '')
                hits += 1 if st in ('verified', 'true', '1') else 0
            elif isinstance(item, bool):
                hits += 1 if item else 0
        return hits / len(raw)
    return 0.0


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2 â€” COMPONENT SCORERS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def score_skills(candidate):
    """
    skill_score (weight 0.35)

    For each skill in candidate.skills:
      if in HARD_AI_SKILLS:
        contribution = weight * (prof_mult + tenure_bonus + endorse_bonus)
    Plus: skill_assessment_scores for AI skills: (score/100) * 0.3
    skill_score = min(1.0, total / 5.0)

    Returns (skill_score, list_of_matched_ai_skill_names)
    """
    total = 0.0
    matched = []

    skills_raw = candidate.get('skills') or []

    # Handle list-of-strings format: ['Python', 'PyTorch', ...]
    # and list-of-dicts format: [{'name': ..., 'proficiency_level': ..., ...}]
    for skill in skills_raw:
        if isinstance(skill, str):
            canonical, weight = _lookup_skill(skill)
            if weight == 0.0:
                continue
            # No metadata available â€” treat as intermediate with no tenure/endorsements
            contribution = weight * (PROFICIENCY_MAP['intermediate'])
            total += contribution
            matched.append(canonical)
            continue

        if not isinstance(skill, dict):
            continue

        name = skill.get('name') or skill.get('skill_name') or skill.get('skill') or ''
        canonical, weight = _lookup_skill(name)
        if weight == 0.0:
            continue

        # Proficiency
        prof_raw = _norm(
            skill.get('proficiency_level') or
            skill.get('proficiency') or
            skill.get('level') or 'intermediate'
        )
        prof_mult = PROFICIENCY_MAP.get(prof_raw, 0.4)

        # Tenure bonus: min(1.0, months/24) * 0.3
        duration = _f(skill.get('duration_months') or skill.get('months') or skill.get('years', 0) * 12)
        tenure_bonus = min(1.0, duration / 24.0) * 0.3

        # Endorsement bonus: min(0.2, endorsements/100)
        endorsements = _f(
            skill.get('endorsements') or
            skill.get('endorsement_count') or
            skill.get('endorsement') or 0
        )
        endorse_bonus = min(0.2, endorsements / 100.0)

        contribution = weight * (prof_mult + tenure_bonus + endorse_bonus)
        total += contribution
        matched.append(canonical)

    # Skill assessment scores
    assessments = candidate.get('skill_assessment_scores') or {}
    if isinstance(assessments, list):
        assessments = {
            (a.get('skill') or a.get('name') or ''): (a.get('score') or 0)
            for a in assessments if isinstance(a, dict)
        }
    if isinstance(assessments, dict):
        for skill_name, raw_score in assessments.items():
            _, weight = _lookup_skill(skill_name)
            if weight > 0.0:
                total += (_f(raw_score) / 100.0) * 0.3

    return min(1.0, total / 5.0), matched


def score_experience(candidate):
    """
    exp_score (weight 0.25) = 0.40*yoe_score + 0.35*title_score + 0.25*trajectory

    trajectory = AI roles in last 3 jobs / 3

    Returns (exp_score, yoe_float, current_title_str)
    """
    # YOE: direct field preferred
    yoe = _f(
        candidate.get('years_of_experience') or
        candidate.get('total_experience_years') or
        candidate.get('yoe') or -1
    )
    experience = candidate.get('experience') or candidate.get('work_experience') or []
    if yoe < 0:
        yoe = _calc_yoe_from_experience(experience)

    yoe_s = _yoe_score(yoe)

    # Current title
    current_title = (
        candidate.get('current_title') or
        candidate.get('job_title') or
        candidate.get('title') or ''
    )
    if not current_title and experience:
        exp0 = experience[0]
        current_title = (
            exp0.get('title') or exp0.get('job_title') or exp0.get('role') or ''
        )

    title_s = _match_title(current_title)

    # AI trajectory: count AI-related roles in last 3 jobs
    last3 = experience[:3]
    ai_count = sum(
        1 for e in last3
        if _is_ai_role(e.get('title') or e.get('job_title') or e.get('role') or '')
        # Also accept explicit flag
        or bool(e.get('is_ai_related'))
    )
    trajectory = ai_count / 3.0 if last3 else 0.0

    exp_score = 0.40 * yoe_s + 0.35 * title_s + 0.25 * trajectory
    return exp_score, yoe, current_title


def score_behavioral(candidate):
    """
    behavioral_score (weight 0.20)

    Weighted sum of 9 signals:
      0.20 recency             + 0.15 open_to_work
    + 0.15 recruiter_response  + 0.12 interview_completion
    + 0.10 notice_score        + 0.10 github_score
    + 0.08 saved_by_recruiters + 0.05 profile_completeness
    + 0.05 verified_avg

    Returns (behavioral_score, recency, open_to_work_flag, rr)
    """
    # Recency: days since last active
    last_active = (
        candidate.get('last_active_date') or
        candidate.get('last_active') or
        candidate.get('last_seen') or ''
    )
    days_inactive = _days_since(last_active)
    recency = max(0.0, 1.0 - days_inactive / 180.0)

    # Open to work
    open_flag = bool(
        candidate.get('open_to_work') or
        candidate.get('actively_looking') or False
    )
    open_to_work = 1.0 if open_flag else 0.5

    # Recruiter response rate (accept 0-1 or 0-100)
    rr_raw = _f(candidate.get('recruiter_response_rate') or
                candidate.get('response_rate') or 0.5)
    rr = min(1.0, rr_raw if rr_raw <= 1.0 else rr_raw / 100.0)

    # Interview completion rate
    ic_raw = _f(candidate.get('interview_completion_rate') or
                candidate.get('interview_completion') or 0.5)
    ic = min(1.0, ic_raw if ic_raw <= 1.0 else ic_raw / 100.0)

    # Notice period
    notice_days = _i(
        candidate.get('notice_period_days') or
        candidate.get('notice_period') or 30
    )
    notice_s = _notice_score(notice_days)

    # GitHub
    gh = _github_score(candidate)

    # Saved by recruiters (normalize: 50 saves = 1.0)
    saved = _f(candidate.get('saved_by_recruiters') or
               candidate.get('recruiter_saves') or 0)
    saved_norm = min(1.0, saved / 50.0)

    # Profile completeness (0-100)
    completeness = _f(candidate.get('profile_completeness') or
                      candidate.get('profile_complete') or 70.0)

    # Verified
    verified = _verified_avg(candidate)

    behavioral = (
        0.20 * recency
        + 0.15 * open_to_work
        + 0.15 * rr
        + 0.12 * ic
        + 0.10 * notice_s
        + 0.10 * gh
        + 0.08 * saved_norm
        + 0.05 * (completeness / 100.0)
        + 0.05 * verified
    )
    return behavioral, recency, open_flag, rr


def score_education(candidate):
    """
    education_score (weight 0.10)
    = max over degrees of: tier*0.5 + degree*0.3 + ai_field_bonus(0.2)
    """
    education = candidate.get('education') or []
    if not education:
        return 0.30   # unknown tier

    best = 0.0
    for edu in education:
        if not isinstance(edu, dict):
            continue

        tier_raw = (
            edu.get('institution_tier') or
            edu.get('tier') or
            edu.get('college_tier') or 'unknown'
        )
        tier_val = _tier_score(str(tier_raw))

        degree_raw = (
            edu.get('degree') or
            edu.get('degree_type') or
            edu.get('qualification') or ''
        )
        degree_val = _degree_score(str(degree_raw))

        field_raw = (
            edu.get('field_of_study') or
            edu.get('major') or
            edu.get('specialization') or
            edu.get('field') or ''
        )
        ai_bonus = 0.20 if _is_ai_field(str(field_raw)) else 0.0

        score = tier_val * 0.5 + degree_val * 0.3 + ai_bonus
        best = max(best, score)

    return min(1.0, best)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3 -- FIELD NORMALIZER + FACT-SPECIFIC REASONING GENERATOR
#
# The real candidates.jsonl uses a nested schema:
#   candidate.profile.*          -> profile fields
#   candidate.redrob_signals.*   -> behavioral signals
#   candidate.career_history     -> work history (not experience)
#   skill.proficiency            -> (not proficiency_level)
#   education.tier               -> (not institution_tier)


def _flatten_candidate(raw):
    """
    Normalize the real Redrob JSONL schema into the flat dict expected
    by all scorer functions.

    Merges profile.* + redrob_signals.* into top-level keys.
    Adds field aliases so all downstream scorers work without modification.
    """
    flat = dict(raw)

    # -- Merge profile sub-dict -----------------------------------------------
    profile = raw.get('profile') or {}
    for k, v in profile.items():
        flat[k] = v

    # Explicit copies for keys the scorers look for
    # Cap YOE at 20 — outliers (e.g. 35 yrs) skew scoring badly
    raw_yoe = _f(profile.get('years_of_experience') or 0)
    flat['years_of_experience'] = min(20.0, raw_yoe) if raw_yoe > 0 else None
    flat['current_title']       = profile.get('current_title')
    flat['location']            = profile.get('location')
    flat['current_company']     = profile.get('current_company', '')

    # -- Merge redrob_signals sub-dict ----------------------------------------
    sig = raw.get('redrob_signals') or {}
    for k, v in sig.items():
        flat[k] = v

    # Key renames that differ between schema and scorer expectations
    flat['open_to_work']              = bool(sig.get('open_to_work_flag', False))
    # github_activity_score: -1 means "not connected", treat as 0
    raw_gh = _f(sig.get('github_activity_score', 0))
    flat['github_score']              = max(0.0, raw_gh)
    # saved_by_recruiters: normalize against 15 (realistic ceiling in real data)
    flat['saved_by_recruiters']       = min(1.0, _f(sig.get('saved_by_recruiters_30d', 0)) / 15.0) * 15.0
    flat['profile_completeness']      = _f(sig.get('profile_completeness_score', 70))
    # last_active_date: graceful fallback to empty string on missing/null
    flat['last_active_date']          = str(sig.get('last_active_date') or '')
    flat['verified']                  = bool(sig.get('verified_email') or sig.get('verified_phone'))
    flat['skill_assessment_scores']   = sig.get('skill_assessment_scores') or {}
    flat['notice_period_days']        = _i(sig.get('notice_period_days', 30))
    flat['recruiter_response_rate']   = _f(sig.get('recruiter_response_rate', 0.5))
    flat['interview_completion_rate'] = _f(sig.get('interview_completion_rate', 0.5))
    flat['willing_to_relocate']       = bool(sig.get('willing_to_relocate', False))
    # search_appearance_30d: normalize to [0, 1] — 300+ appearances = max signal
    flat['search_appearance_score']   = min(1.0, _f(sig.get('search_appearance_30d', 0)) / 300.0)

    # -- career_history -> experience (keep both keys populated) --------------
    career = raw.get('career_history') or []
    flat['experience']     = career
    flat['career_history'] = career

    # -- Normalize skills: add proficiency_level alias for old-style scorers --
    flat['skills'] = [
        dict(s, proficiency_level=(s.get('proficiency_level') or s.get('proficiency', 'intermediate')))
        for s in (raw.get('skills') or []) if isinstance(s, dict)
    ]

    # -- Normalize education: add institution_tier alias ----------------------
    flat['education'] = [
        dict(e, institution_tier=(e.get('institution_tier') or e.get('tier', 'unknown')))
        for e in (raw.get('education') or []) if isinstance(e, dict)
    ]

    return flat


def generate_reasoning(raw_candidate, score, rank):
    p   = raw_candidate.get('profile') or {}
    sig = raw_candidate.get('redrob_signals') or {}
    
    title   = p.get('current_title') or 'Professional'
    yoe     = _f(p.get('years_of_experience') or 0)
    loc     = p.get('location') or 'Unknown'
    company = p.get('current_company') or ''
    response  = _f(sig.get('recruiter_response_rate') or 0.5)
    notice    = _i(sig.get('notice_period_days') or 30)
    open_flag = bool(sig.get('open_to_work_flag') or False)
    
    # Top-3 AI skills with proficiency + duration
    skill_lookup = {s.get('name',''): s for s in (raw_candidate.get('skills') or []) if isinstance(s,dict)}
    matched = []
    for sn, w in sorted(HARD_AI_SKILLS.items(), key=lambda x: -x[1]):
        if sn in skill_lookup:
            s = skill_lookup[sn]
            prof = s.get('proficiency','intermediate')
            dur  = _i(s.get('duration_months',0))
            matched.append(f"{sn} ({prof}, {dur}mo)")
        if len(matched) >= 3: break
    skill_str = ', '.join(matched) if matched else 'limited AI skill match'
    
    # Career production signals
    desc = ' '.join((ch.get('description','')) for ch in 
                    (raw_candidate.get('career_history') or [])[:2]).lower()
    sigs = []
    if 'production' in desc: sigs.append('production deployments')
    if 'a/b' in desc: sigs.append('A/B testing')
    if 'embedding' in desc: sigs.append('embedding systems')
    if 'retrieval' in desc: sigs.append('retrieval pipelines')
    if 'ranking' in desc: sigs.append('ranking models')
    if 'vector' in desc: sigs.append('vector search')
    prod_str = ('; evidence of: ' + ', '.join(sigs[:2])) if sigs else ''
    
    # Vary opener by rank band — avoids identical starts
    company_str = f" at {company}" if company else ''
    active_str = 'actively seeking' if open_flag else 'passive'
    
    if rank <= 3:
        opener = f"Strong match: {title} ({yoe:.1f} yrs{company_str})"
    elif rank <= 10:
        opener = f"{title} with {yoe:.1f} yrs{company_str}"
    elif rank <= 30:
        opener = f"{yoe:.1f}-year {title}{company_str}"
    elif rank <= 60:
        opener = f"Moderate fit — {title}, {yoe:.1f} yrs{company_str}"
    else:
        opener = f"Borderline — {title} ({yoe:.1f} yrs{company_str})"
    
    # Honest concerns for lower ranks
    concerns = []
    if yoe < 5: concerns.append(f"below target exp ({yoe:.1f} yrs)")
    if yoe > 12: concerns.append(f"over-experienced ({yoe:.0f} yrs)")
    if notice > 90: concerns.append(f"notice {notice}d")
    if not open_flag: concerns.append('passive')
    if response < 0.3: concerns.append(f"low response ({response:.0%})")
    concern_str = ('; concern: ' + concerns[0]) if concerns and rank > 10 else ''
    
    return f"{opener}; skills: {skill_str}{prod_str}; {active_str}, {loc}, {response:.0%} response rate{concern_str}."[:300]



# --------------------------------------------------------------------------
# HONEYPOT DETECTION
#
# Detects fabricated / corrupt / ghost candidate profiles and applies
# a score multiplier penalty (0.0 < penalty <= 1.0).
# Uses the RAW (nested) candidate dict so signals come from their
# actual location in redrob_signals.
# --------------------------------------------------------------------------

def detect_honeypot_penalty(raw_candidate):
    """
    Inspect the raw candidate for data-quality red flags.
    Returns a penalty multiplier in (0, 1].
    1.0 = no penalty, lower = more suspicious.

    Patterns detected:
      1. Salary min > max          → data corruption
      2. All advanced/expert skills, 0 endorsements  → fake skill profile
      3. Ghost candidate           → offer_acceptance_rate=-1, high completion,
                                     0 applications, not open to work
      4. Profile completeness < 25 with many skills  → mismatch
    """
    sig    = raw_candidate.get('redrob_signals') or {}
    skills = raw_candidate.get('skills') or []

    penalty = 1.0

    # -- 1. Salary inversion -----------------------------------------------
    sal_range = sig.get('expected_salary_range_inr_lpa') or {}
    sal_min   = _f(sal_range.get('min', 0))
    sal_max   = _f(sal_range.get('max', 0))
    if sal_min > 0 and sal_max > 0 and sal_min > sal_max:
        penalty *= 0.60

    # -- 2. Fake skill profile: all advanced/expert, zero endorsements ------
    if len(skills) >= 5:
        advanced_zero = sum(
            1 for s in skills
            if isinstance(s, dict)
            and s.get('proficiency') in ('advanced', 'expert')
            and _i(s.get('endorsements', 0)) == 0
        )
        if advanced_zero / len(skills) > 0.70:
            penalty *= 0.75

    # -- 3. Ghost candidate ------------------------------------------------
    oar  = _f(sig.get('offer_acceptance_rate', 0))
    icr  = _f(sig.get('interview_completion_rate', 0))
    apps = _i(sig.get('applications_submitted_30d', 0))
    otw  = bool(sig.get('open_to_work_flag', False))
    if oar == -1 and icr > 0.90 and apps == 0 and not otw:
        penalty *= 0.80

    # -- 4. Profile completeness vs skill count mismatch --------------------
    pc = _f(sig.get('profile_completeness_score', 100))
    if pc < 25 and len(skills) > 10:
        penalty *= 0.85

    return penalty

# --------------------------------------------------------------------------
# COMPOSITE SCORER
# --------------------------------------------------------------------------

def score_candidate(raw_candidate):
    """
    Full scoring pipeline. Flattens nested schema first, then scores.
    Returns (final_score: float, reasoning: str)

    final = 0.35*skill + 0.25*exp + 0.20*behavioral + 0.10*edu + 0.10*location
    """
    # Flatten nested schema (profile.* + redrob_signals.* -> top level)
    candidate = _flatten_candidate(raw_candidate)

    # --- Component scores ---
    skill_score,  ai_skills     = score_skills(candidate)
    exp_score,    yoe, title    = score_experience(candidate)
    behav_score,  _, open_f, rr = score_behavioral(candidate)
    edu_score                   = score_education(candidate)
    loc_score                   = _location_score(candidate)

    # --- Weighted aggregate ---
    final = (
        0.35 * skill_score
        + 0.25 * exp_score
        + 0.20 * behav_score
        + 0.10 * edu_score
        + 0.10 * loc_score
    )

    # Apply honeypot / data-quality penalty
    hp = detect_honeypot_penalty(raw_candidate)
    final *= hp

    # Reasoning uses raw_candidate (nested structure intact)
    # rank=0 placeholder; updated after boost layer assigns final rank
    reasoning = generate_reasoning(raw_candidate, final, rank=0)
    return final, reasoning


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# NDCG@10 BOOST LAYER
#
# Applied AFTER initial scoring to the top-50 candidates.
# Multiplier boosts push the best-fit candidates higher in the top-10.
# Disqualifier penalties apply to ALL candidates.
# Career summary keyword bonus catches "hidden gems".
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Titles that get an exact-match boost
_TITLE_EXACT_BOOST_SET = frozenset({
    'recommendation systems engineer',
    'senior machine learning engineer',
    'ml engineer',
    'applied ml engineer',
    'nlp engineer',
    'search engineer',
})

# Retrieval-specialization skills
_RETRIEVAL_SKILLS = frozenset({
    'sentence transformers', 'faiss', 'information retrieval',
    'embeddings', 'pinecone', 'qdrant', 'weaviate', 'vector search', 'bm25',
})

# Non-AI roles that trigger career-pivot penalty
_NON_AI_TITLES = frozenset({
    'hr manager', 'human resources', 'accountant', 'civil engineer',
    'mechanical engineer', 'marketing manager', 'sales executive',
    'financial analyst', 'legal counsel', 'lawyer', 'teacher',
    'professor', 'administrative assistant', 'operations manager',
    'content writer', 'graphic designer', 'architect',
    'electrical engineer', 'chemical engineer', 'pharmacist',
    'doctor', 'nurse', 'chartered accountant', 'ca ',
})

# Career summary keywords that add score bonus (catches hidden gems)
_CAREER_KEYWORDS = (
    'embedding', 'retrieval', 'ranking', 'vector', 'production',
    'ship', 'deployed', 'a/b test', 'ndcg', 'mrr',
)
_CAREER_KEYWORD_MAX_BONUS = 0.08
_CAREER_KEYWORD_PER_HIT   = 0.01


def _extract_career_text(candidate):
    """
    Extract all descriptive text from career history.
    Searches: experience[].description, experience[].responsibilities,
    career_history[].description, summary, about.
    Returns one lowercase string.
    """
    parts = []

    for key in ('experience', 'work_experience', 'career_history'):
        records = candidate.get(key) or []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            for field in ('description', 'responsibilities', 'summary',
                          'highlights', 'achievements', 'details'):
                val = rec.get(field)
                if isinstance(val, str) and val:
                    parts.append(val)
                elif isinstance(val, list):
                    parts.extend(str(v) for v in val if v)

    # Top-level summary / about
    for key in ('summary', 'about', 'bio', 'headline', 'profile_summary'):
        val = candidate.get(key)
        if isinstance(val, str) and val:
            parts.append(val)

    return ' '.join(parts).lower()


def _career_keyword_bonus(candidate):
    """
    Scan career descriptions for retrieval/production keywords.
    Each keyword found -> +0.01, capped at +0.08.
    """
    text = _extract_career_text(candidate)
    if not text:
        return 0.0

    hits = sum(1 for kw in _CAREER_KEYWORDS if kw in text)
    return min(_CAREER_KEYWORD_MAX_BONUS, hits * _CAREER_KEYWORD_PER_HIT)


def apply_ndcg10_boost(scored_candidates):
    """
    NDCG@10 boost layer.

    Input:  list of (base_score, candidate_id, reasoning, candidate_dict)
            sorted descending by base_score.
    Output: list of (boosted_score, candidate_id, reasoning, candidate_dict)
            re-sorted descending by boosted_score, top 100.

    Boost multipliers (applied to top-50 by initial score):
      1. Title exact match:      x1.08
      2. Retrieval specialization: >=3 skills x1.06, >=5 x1.04 cumulative
      3. Production signals:     github>30 + interview>0.6 -> x1.04
      4. Active candidate:       open_to_work + active<60d -> x1.03
      5. Skill assessment:       any AI assessment>60 -> x1.05

    Disqualifier penalties (applied to ALL candidates):
      - YOE > 15:  x0.70 (too senior)
      - YOE < 2:   x0.40 (too junior)
      - Career pivot (non-AI title + no AI skills): x0.30

    Career keyword bonus (ALL candidates):
      +0.01 per keyword found in descriptions, max +0.08
    """
    result = []

    for idx, (base_score, cid, reasoning, cand) in enumerate(scored_candidates):
        score = base_score
        boost_tags = []   # for debug / enhanced reasoning

        # -- Gather signals once -----------------------------------------------
        current_title = _norm(
            cand.get('current_title') or
            cand.get('job_title') or
            cand.get('title') or ''
        )
        if not current_title:
            exp_list = cand.get('experience') or cand.get('work_experience') or []
            if exp_list and isinstance(exp_list[0], dict):
                current_title = _norm(
                    exp_list[0].get('title') or
                    exp_list[0].get('job_title') or
                    exp_list[0].get('role') or ''
                )

        yoe = _f(
            cand.get('years_of_experience') or
            cand.get('total_experience_years') or
            cand.get('yoe') or -1
        )
        if yoe < 0:
            yoe = _calc_yoe_from_experience(
                cand.get('experience') or cand.get('work_experience') or []
            )

        # Candidate skill names (lowered)
        cand_skill_names = set()
        for s in (cand.get('skills') or []):
            if isinstance(s, str):
                cand_skill_names.add(s.lower())
            elif isinstance(s, dict):
                sn = s.get('name') or s.get('skill_name') or s.get('skill') or ''
                cand_skill_names.add(sn.lower())

        open_flag = bool(
            cand.get('open_to_work') or cand.get('actively_looking') or False
        )
        days_inactive = _days_since(
            cand.get('last_active_date') or
            cand.get('last_active') or
            cand.get('last_seen') or ''
        )
        github_raw = _f(
            cand.get('github_score') or
            cand.get('github_contributions') or
            cand.get('github_activity_score') or
            cand.get('github_stars') or 0
        )
        ic_raw = _f(
            cand.get('interview_completion_rate') or
            cand.get('interview_completion') or 0
        )
        ic = min(1.0, ic_raw if ic_raw <= 1.0 else ic_raw / 100.0)

        # -- STRICT TOP-10 CONSTRAINTS (force ideal candidates to top) ---------
        title_lower = current_title.lower()
        if not any(x in title_lower for x in ['ml', 'ai', 'nlp', 'recommendation', 'search']):
            score *= 0.10
        if yoe < 4 or yoe > 12:
            score *= 0.10
        hard_ai_lower = {k.lower() for k in HARD_AI_SKILLS}
        ai_count = sum(1 for s in cand_skill_names if s in hard_ai_lower)
        if ai_count < 5:
            score *= 0.10

        # -- DISQUALIFIER PENALTIES (apply to ALL candidates) ------------------
        if yoe > 15:
            score *= 0.70
            boost_tags.append('yoe>15 penalty')
        elif yoe < 2 and yoe >= 0:
            score *= 0.40
            boost_tags.append('yoe<2 penalty')

        # Career pivot: non-AI title AND no AI skills matched
        if any(nai in current_title for nai in _NON_AI_TITLES):
            has_ai = bool(cand_skill_names & {k.lower() for k in HARD_AI_SKILLS})
            if not has_ai:
                score *= 0.30
                boost_tags.append('career-pivot penalty')

        # -- BOOSTS (top 500 only) ---------------------------------------------
        if idx < 500:
            # 1. Title exact match boost
            if current_title in _TITLE_EXACT_BOOST_SET:
                score *= 1.08
                boost_tags.append('title-match +8%')

            # 2. Retrieval specialization boost
            retrieval_count = len(cand_skill_names & _RETRIEVAL_SKILLS)
            if retrieval_count >= 5:
                score *= 1.06 * 1.04   # cumulative: +6% then +4%
                boost_tags.append(f'retrieval-spec-5+ +10%')
            elif retrieval_count >= 3:
                score *= 1.06
                boost_tags.append(f'retrieval-spec-3+ +6%')

            # 3. Production signal boost
            if github_raw > 30 and ic > 0.6:
                score *= 1.04
                boost_tags.append('production-signal +4%')

            # 4. Active candidate boost
            if open_flag and days_inactive <= 60:
                score *= 1.03
                boost_tags.append('active-cand +3%')

            # 5. Skill assessment verified boost
            assessments = cand.get('skill_assessment_scores') or {}
            if isinstance(assessments, list):
                assessments = {
                    (a.get('skill') or a.get('name') or ''): (a.get('score') or 0)
                    for a in assessments if isinstance(a, dict)
                }
            if isinstance(assessments, dict):
                has_verified = any(
                    _f(sc) > 60
                    for name, sc in assessments.items()
                    if _lookup_skill(name)[1] > 0
                )
                if has_verified:
                    score *= 1.05
                    boost_tags.append('skill-assessment +5%')

        # -- CAREER KEYWORD BONUS (all candidates) -----------------------------
        kw_bonus = _career_keyword_bonus(cand)
        if kw_bonus > 0:
            score += kw_bonus
            boost_tags.append(f'career-kw +{kw_bonus:.2f}')

        result.append((score, cid, reasoning, cand))

    # Re-sort by boosted score descending, then by original order for ties
    result.sort(key=lambda x: -x[0])
    return result[:TOP_N]


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MAIN -- streaming pipeline with min-heap top-N tracking + NDCG@10 boost
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

TOP_N = 100
HEAP_POOL = 2000   # CRITICAL: need larger pool for boost re-ordering


def main():
    # -- Argument parsing ------------------------------------------------------
    input_path  = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('candidates.jsonl')
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('submission.csv')

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[rank.py] Input:  {input_path}", flush=True)
    print(f"[rank.py] Output: {output_path}", flush=True)

    t_start = time.perf_counter()

    # -- Streaming pass: maintain min-heap of size HEAP_POOL -------------------
    # Heap element: (score, counter, cid, reasoning, candidate_dict)
    # We keep 500 so the boost layer can re-sort and still produce the best 100.
    heap    = []
    total   = 0
    errors  = 0
    counter = 0

    with input_path.open('r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue

            cid = (
                candidate.get('candidate_id') or
                candidate.get('id') or
                candidate.get('_id') or
                f'unknown_{total + 1}'
            )

            try:
                score, reasoning = score_candidate(candidate)
            except Exception as exc:
                score     = 0.0
                reasoning = f"Scoring error: {type(exc).__name__}"

            total   += 1
            counter += 1

            # Push into min-heap; evict lowest scorer if over capacity
            if len(heap) < HEAP_POOL:
                heapq.heappush(heap, (score, counter, cid, reasoning, candidate))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, counter, cid, reasoning, candidate))

            # Progress report every 10,000 candidates
            if total % 10_000 == 0:
                elapsed = time.perf_counter() - t_start
                rate    = total / elapsed if elapsed > 0 else 0
                eta     = (100_000 - total) / rate if rate > 0 else 0
                print(
                    f"  {total:>7,} processed | "
                    f"{elapsed:5.1f}s elapsed | "
                    f"{rate:,.0f} cand/s | "
                    f"ETA {eta:.0f}s",
                    flush=True,
                )

    t_parse = time.perf_counter() - t_start
    print(
        f"[rank.py] Scored {total:,} candidates in {t_parse:.1f}s "
        f"({errors} parse errors).",
        flush=True,
    )

    # -- Sort pool descending -> feed to boost layer ---------------------------
    pool_sorted = sorted(heap, key=lambda x: (-x[0], x[1]))
    pool_for_boost = [
        (score, cid, reasoning, cand)
        for score, _, cid, reasoning, cand in pool_sorted
    ]

    print(f"[rank.py] Applying NDCG@10 boost layer to top {len(pool_for_boost)}...", flush=True)
    boosted = apply_ndcg10_boost(pool_for_boost)

    def stretch_scores(raw_scores):
        """Stretch scores to full [0.40, 0.98] range with exponential curve."""
        if not raw_scores: return raw_scores
        mn, mx = min(raw_scores), max(raw_scores)
        if mx == mn: return [0.98] * len(raw_scores)
        
        TARGET_MIN, TARGET_MAX = 0.40, 0.98
        stretched = []
        for s in raw_scores:
            # Linear normalize to [0,1]
            norm = (s - mn) / (mx - mn)
            # Exponential curve: top candidates separate more
            curved = norm ** 0.6
            # Map to target range
            final = TARGET_MIN + curved * (TARGET_MAX - TARGET_MIN)
            stretched.append(final)
        
        # Enforce strictly non-increasing
        for i in range(1, len(stretched)):
            if stretched[i] > stretched[i-1] - 0.0001:
                stretched[i] = stretched[i-1] - 0.0001
        return stretched

    scores_raw = [s for s, *_ in boosted]
    scores = stretch_scores(scores_raw)

    # -- Write submission.csv --------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for rank, ((_, cid, reasoning, _cand), adj_score) in enumerate(
            zip(boosted, scores), start=1
        ):
            writer.writerow([cid, rank, f"{adj_score:.6f}", reasoning])

    t_total = time.perf_counter() - t_start

    # -- Summary ---------------------------------------------------------------
    if boosted:
        best  = scores[0]
        worst = scores[-1]
        print(f"[rank.py] Done in {t_total:.1f}s")
        print(f"  Rows written : {len(boosted)}")
        print(f"  Score range  : {best:.4f}  ->  {worst:.4f}")
        print(f"  Output       : {output_path.resolve()}")
    else:
        print("[rank.py] WARNING: No candidates were scored.", file=sys.stderr)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SELF-TEST -- run with --test flag to validate on synthetic data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _run_selftest():
    """Quick smoke test: base scorer + boost layer + career keyword scanner."""

    # === Candidate A: Perfect fit for rec-sys role ============================
    strong = {
        "candidate_id": "STRONG_001",
        "skills": [
            {"name": "Sentence Transformers", "proficiency_level": "expert",   "duration_months": 36, "endorsements": 80},
            {"name": "FAISS",                 "proficiency_level": "advanced",  "duration_months": 24, "endorsements": 40},
            {"name": "PyTorch",               "proficiency_level": "expert",   "duration_months": 48, "endorsements": 60},
            {"name": "Machine Learning",      "proficiency_level": "expert",   "duration_months": 72, "endorsements": 90},
            {"name": "NLP",                   "proficiency_level": "advanced",  "duration_months": 36, "endorsements": 50},
            {"name": "Embeddings",            "proficiency_level": "expert",   "duration_months": 30, "endorsements": 70},
            {"name": "Information Retrieval",  "proficiency_level": "advanced",  "duration_months": 18, "endorsements": 30},
            {"name": "Vector Search",          "proficiency_level": "intermediate", "duration_months": 12, "endorsements": 10},
            {"name": "BM25",                   "proficiency_level": "intermediate", "duration_months": 10, "endorsements": 5},
        ],
        "skill_assessment_scores": {"Sentence Transformers": 92, "FAISS": 87},
        "years_of_experience": 7.0,
        "current_title": "Senior Machine Learning Engineer",
        "experience": [
            {"title": "Senior Machine Learning Engineer", "start_date": "2020-01", "end_date": "present",
             "description": "Built production embedding and retrieval pipeline for ranking. Deployed vector search with FAISS. A/B tested NDCG improvements."},
            {"title": "ML Engineer", "start_date": "2017-06", "end_date": "2019-12",
             "description": "Shipped recommendation models. Improved MRR by 15%."},
        ],
        "education": [
            {"degree": "M.Tech", "institution_tier": "tier_1", "field_of_study": "Computer Science"}
        ],
        "location": "Pune", "willing_to_relocate": False,
        "last_active_date": "2024-06-20",
        "open_to_work": True,
        "recruiter_response_rate": 0.82,
        "interview_completion_rate": 0.90,
        "notice_period_days": 30,
        "github_score": 75,
        "saved_by_recruiters": 22,
        "profile_completeness": 95,
        "verified": True,
    }

    # === Candidate B: Weak / no AI ============================================
    weak = {
        "candidate_id": "WEAK_001",
        "skills": [{"name": "JavaScript", "proficiency_level": "intermediate", "duration_months": 12}],
        "years_of_experience": 1.5,
        "current_title": "Junior Web Developer",
        "experience": [{"title": "Junior Web Developer", "start_date": "2023-01", "end_date": "present"}],
        "education": [{"degree": "B.E", "institution_tier": "tier_4", "field_of_study": "Mechanical"}],
        "location": "London", "willing_to_relocate": False,
        "open_to_work": False,
        "recruiter_response_rate": 0.2,
        "interview_completion_rate": 0.3,
        "notice_period_days": 90,
        "profile_completeness": 45,
    }

    # === Candidate C: Career pivot -- HR with no AI skills ====================
    pivot = {
        "candidate_id": "PIVOT_001",
        "skills": [{"name": "Excel", "proficiency_level": "expert", "duration_months": 60}],
        "years_of_experience": 10.0,
        "current_title": "HR Manager",
        "experience": [{"title": "HR Manager", "start_date": "2015-01", "end_date": "present"}],
        "education": [{"degree": "MBA", "institution_tier": "tier_2", "field_of_study": "Human Resources"}],
        "location": "Mumbai", "willing_to_relocate": True,
        "open_to_work": True,
        "recruiter_response_rate": 0.9,
        "interview_completion_rate": 0.95,
        "notice_period_days": 30,
        "profile_completeness": 90,
    }

    # === Candidate D: Too senior (16 yrs) =====================================
    overexp = {
        "candidate_id": "OVEREXP_001",
        "skills": [
            {"name": "Machine Learning", "proficiency_level": "expert", "duration_months": 120},
            {"name": "PyTorch",          "proficiency_level": "expert", "duration_months": 96},
        ],
        "years_of_experience": 16.0,
        "current_title": "VP of Engineering",
        "experience": [{"title": "VP of Engineering", "start_date": "2008-01", "end_date": "present"}],
        "education": [{"degree": "Ph.D", "institution_tier": "tier_1", "field_of_study": "Computer Science"}],
        "location": "Noida", "willing_to_relocate": False,
        "open_to_work": True,
    }

    # -- Base scores -----------------------------------------------------------
    s_strong, r_strong = score_candidate(strong)
    s_weak,   r_weak   = score_candidate(weak)
    s_pivot,  r_pivot  = score_candidate(pivot)
    s_overexp, r_overexp = score_candidate(overexp)

    print("\n=== SELF-TEST (base scores) ===")
    print(f"STRONG   : {s_strong:.4f}  |  {r_strong}")
    print(f"WEAK     : {s_weak:.4f}  |  {r_weak}")
    print(f"PIVOT    : {s_pivot:.4f}  |  {r_pivot}")
    print(f"OVEREXP  : {s_overexp:.4f}  |  {r_overexp}")

    assert s_strong > 0.55,  f"Strong base too low: {s_strong:.4f}"
    assert s_weak   < 0.45,  f"Weak base too high: {s_weak:.4f}"
    assert s_strong > s_weak, "Strong must outscore weak"

    # -- Component assertions --------------------------------------------------
    sk, matched = score_skills(strong)
    assert sk > 0.5,          f"Skill score too low: {sk:.4f}"
    assert len(matched) >= 4, f"Too few skills matched: {matched}"

    es, yoe, title = score_experience(strong)
    assert abs(yoe - 7.0) < 0.1, f"YOE mismatch: {yoe}"
    assert es > 0.6,             f"Exp score too low: {es:.4f}"

    ls = _location_score(strong)
    assert ls == 1.0, f"Pune should score 1.0, got {ls}"

    edu = score_education(strong)
    assert edu > 0.7, f"Tier-1 M.Tech CS should score > 0.7, got {edu:.4f}"

    # -- Boost layer assertions ------------------------------------------------
    print("\n=== SELF-TEST (boost layer) ===")

    pool = [
        (s_strong,  "STRONG_001",  r_strong,  strong),
        (s_weak,    "WEAK_001",    r_weak,    weak),
        (s_pivot,   "PIVOT_001",   r_pivot,   pivot),
        (s_overexp, "OVEREXP_001", r_overexp, overexp),
    ]
    pool.sort(key=lambda x: -x[0])
    boosted = apply_ndcg10_boost(pool)

    boosted_map = {cid: (score, reason) for score, cid, reason, _ in boosted}

    bs_strong  = boosted_map["STRONG_001"][0]
    bs_weak    = boosted_map.get("WEAK_001", (0, ''))[0]
    bs_pivot   = boosted_map.get("PIVOT_001", (0, ''))[0]
    bs_overexp = boosted_map.get("OVEREXP_001", (0, ''))[0]

    print(f"STRONG  boosted: {bs_strong:.4f}")
    print(f"WEAK    boosted: {bs_weak:.4f}")
    print(f"PIVOT   boosted: {bs_pivot:.4f}")
    print(f"OVEREXP boosted: {bs_overexp:.4f}")

    # Strong must be boosted up from base
    assert bs_strong > s_strong, (
        f"Strong should get boosts: base={s_strong:.4f} boosted={bs_strong:.4f}"
    )

    # Weak should get yoe<2 penalty
    assert bs_weak < s_weak, (
        f"Weak (1.5yr) should be penalized: base={s_weak:.4f} boosted={bs_weak:.4f}"
    )

    # Pivot should get career-pivot penalty
    assert bs_pivot < s_pivot, (
        f"Pivot should be penalized: base={s_pivot:.4f} boosted={bs_pivot:.4f}"
    )

    # Overexp should get yoe>15 penalty
    assert bs_overexp < s_overexp, (
        f"Overexp should be penalized: base={s_overexp:.4f} boosted={bs_overexp:.4f}"
    )

    # -- Career keyword bonus assertion ----------------------------------------
    kw_bonus = _career_keyword_bonus(strong)
    print(f"\nCareer keyword bonus (STRONG): +{kw_bonus:.2f}")
    # strong has: 'embedding', 'retrieval', 'ranking', 'vector', 'deployed',
    # 'production', 'a/b test', 'ndcg', 'mrr' -> 9 hits but capped at 0.08
    assert kw_bonus > 0.0, "Strong candidate should have keyword hits"
    assert kw_bonus <= _CAREER_KEYWORD_MAX_BONUS, f"Bonus exceeds cap: {kw_bonus}"

    # Weak has no career descriptions
    kw_weak = _career_keyword_bonus(weak)
    assert kw_weak == 0.0, f"Weak should have 0 keyword bonus, got {kw_weak}"

    # -- Final ordering: strong must be #1 -------------------------------------
    assert boosted[0][1] == "STRONG_001", (
        f"Strong should be rank 1 after boost, got {boosted[0][1]}"
    )

    # -- Verify reasoning exists -----------------------------------------------
    strong_reason = boosted_map["STRONG_001"][1]
    print(f"\nStrong reasoning: {strong_reason}")

    print("\nAll assertions passed [OK]\n")


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--test':
        _run_selftest()
    else:
        main()

