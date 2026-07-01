#!/usr/bin/env python3
"""
rank.py — Redrob Hackathon Candidate Ranker  (thin CLI entry point)
====================================================================
Self-contained: NO external services, NO API calls, NO network.
Standard library only.

All scoring logic lives in src/ranker/:
  config.py     — constants, weights, skill dicts
  features.py   — career-text + structured feature extraction
  behavioral.py — redrob_signals scoring
  honeypot.py   — data-quality penalty detection
  scorer.py     — orchestrates features → weighted score → CandidateScore
  reasoning.py  — grounded text from feature values
  schema.py     — CandidateScore / FeatureSet dataclasses
  io_utils.py   — streaming JSONL reader, CSV writer

Usage:
    python rank.py [candidates.jsonl] [submission.csv]
    python rank.py                           # uses default filenames
    python rank.py --test                    # runs self-test suite
"""

import heapq
import sys
import time
from pathlib import Path

from src.ranker.io_utils  import stream_candidates, write_submission
from src.ranker.scorer    import score_candidate
from src.ranker.reasoning import reasoning_for
from src.ranker.features  import flatten_candidate, _f, _i, _days_since, _lookup_skill
from src.ranker.config    import (
    TOP_N, HEAP_POOL, HARD_AI_SKILLS,
    TITLE_EXACT_BOOST_SET, RETRIEVAL_SKILLS, NON_AI_TITLES,
    CAREER_KEYWORDS, CAREER_KEYWORD_MAX_BONUS, CAREER_KEYWORD_PER_HIT,
)


# ── NDCG@10 boost layer ───────────────────────────────────────────────────────

def _extract_career_text_for_boost(candidate: dict) -> str:
    """Extract lowercase career text for keyword bonus."""
    parts = []
    for key in ('experience', 'work_experience', 'career_history'):
        for rec in (candidate.get(key) or []):
            if not isinstance(rec, dict):
                continue
            for field in ('description', 'responsibilities', 'summary',
                          'highlights', 'achievements', 'details'):
                val = rec.get(field)
                if isinstance(val, str) and val:
                    parts.append(val)
                elif isinstance(val, list):
                    parts.extend(str(v) for v in val if v)
    for key in ('summary', 'about', 'bio', 'headline', 'profile_summary'):
        val = candidate.get(key)
        if isinstance(val, str) and val:
            parts.append(val)
    return ' '.join(parts).lower()


def _career_keyword_bonus(candidate: dict) -> float:
    text = _extract_career_text_for_boost(candidate)
    if not text:
        return 0.0
    hits = sum(1 for kw in CAREER_KEYWORDS if kw in text)
    return min(CAREER_KEYWORD_MAX_BONUS, hits * CAREER_KEYWORD_PER_HIT)


def apply_ndcg10_boost(scored_candidates: list) -> list:
    """
    NDCG@10 boost layer.

    Input:  list of (base_score, candidate_id, reasoning, candidate_dict)
            sorted descending by base_score.
    Output: list of (boosted_score, candidate_id, reasoning, candidate_dict)
            re-sorted descending, top TOP_N.

    Boost multipliers (top 500 only):
      1. Title exact match        × 1.08
      2. Retrieval specialization ≥5 skills × 1.10, ≥3 × 1.06
      3. Production signals       github>30 + interview>0.6 → × 1.04
      4. Active candidate         open + active<60d → × 1.03
      5. Skill assessment         any AI score>60 → × 1.05

    Disqualifier penalties (ALL candidates):
      - YOE > 15: × 0.70
      - YOE < 2:  × 0.40
      - Career pivot (non-AI title + no AI skills): × 0.30
      - No ML title match: × 0.0001
      - YOE out of 4–12 range: × 0.0001
      - Fewer than 5 hard AI skills: × 0.0001

    Career keyword bonus (ALL candidates):
      +0.01 per keyword in descriptions, capped at +0.08
    """
    result = []

    for idx, (base_score, cid, reasoning, cand) in enumerate(scored_candidates):
        score = base_score

        flat          = flatten_candidate(cand)
        current_title = str(flat.get('current_title') or '').lower().strip()
        yoe           = _f(flat.get('years_of_experience') or 0)
        cand_skill_names = {
            s.get('name', '').lower()
            for s in flat.get('skills', [])
            if isinstance(s, dict)
        }
        open_flag    = bool(flat.get('open_to_work', False))
        days_inactive = _days_since(flat.get('last_active_date', ''))
        github_raw   = _f(flat.get('github_score', 0))
        ic           = min(1.0, _f(flat.get('interview_completion_rate', 0)))

        # ── Strict top-10 constraints ──────────────────────────────────────
        if not any(x in current_title for x in [
            'ml', 'machine learning', 'ai', 'nlp', 'recommendation',
            'search', 'applied scientist',
        ]):
            score *= 0.0001

        if yoe < 4 or yoe > 12:
            score *= 0.0001

        hard_ai_lower = {k.lower() for k in HARD_AI_SKILLS}
        if sum(1 for s in cand_skill_names if s in hard_ai_lower) < 5:
            score *= 0.0001

        # ── Disqualifier penalties ─────────────────────────────────────────
        if yoe > 15:
            score *= 0.70
        elif yoe < 2:
            score *= 0.40

        if any(nai in current_title for nai in NON_AI_TITLES):
            if not (cand_skill_names & {k.lower() for k in HARD_AI_SKILLS}):
                score *= 0.30

        # ── Boosts (top 500 only) ──────────────────────────────────────────
        if idx < 500:
            if current_title in TITLE_EXACT_BOOST_SET:
                score *= 1.08

            retrieval_count = len(cand_skill_names & RETRIEVAL_SKILLS)
            if retrieval_count >= 5:
                score *= 1.06 * 1.04
            elif retrieval_count >= 3:
                score *= 1.06

            if github_raw > 30 and ic > 0.6:
                score *= 1.04

            if open_flag and days_inactive <= 60:
                score *= 1.03

            assessments = cand.get('skill_assessment_scores') or {}
            if isinstance(assessments, list):
                assessments = {
                    (a.get('skill') or a.get('name') or ''): (a.get('score') or 0)
                    for a in assessments if isinstance(a, dict)
                }
            if isinstance(assessments, dict):
                if any(
                    _f(sc) > 60
                    for name, sc in assessments.items()
                    if _lookup_skill(name)[1] > 0
                ):
                    score *= 1.05

        # ── Career keyword bonus ───────────────────────────────────────────
        score += _career_keyword_bonus(cand)

        result.append((score, cid, reasoning, cand))

    result.sort(key=lambda x: -x[0])
    return result[:TOP_N]


def _stretch_scores(raw_scores: list) -> list:
    """Stretch to [0.40, 0.98] with exponential curve for score spread."""
    if not raw_scores:
        return raw_scores
    mn, mx = min(raw_scores), max(raw_scores)
    if mx == mn:
        return [0.98] * len(raw_scores)

    TARGET_MIN, TARGET_MAX = 0.40, 0.98
    stretched = []
    for s in raw_scores:
        norm    = (s - mn) / (mx - mn)
        curved  = norm ** 0.6
        stretched.append(TARGET_MIN + curved * (TARGET_MAX - TARGET_MIN))

    # Enforce strictly non-increasing
    for i in range(1, len(stretched)):
        if stretched[i] > stretched[i - 1] - 0.0001:
            stretched[i] = stretched[i - 1] - 0.0001

    return stretched


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    input_path  = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('candidates.jsonl')
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('submission.csv')

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[rank.py] Input:  {input_path}", flush=True)
    print(f"[rank.py] Output: {output_path}", flush=True)

    t_start = time.perf_counter()
    heap    = []
    total   = 0
    counter = 0

    for candidate, cid in stream_candidates(input_path):
        try:
            cs = score_candidate(candidate)
        except Exception as exc:
            # Fallback: score 0 with error note, never crash the pipeline
            from src.ranker.schema import CandidateScore, FeatureSet
            cs = CandidateScore(
                candidate_id  = cid,
                base_score    = 0.0,
                honeypot_mult = 1.0,
                final_score   = 0.0,
                features      = FeatureSet(),
                behav_score   = 0.0,
                raw_candidate = candidate,
                stuffer_reason= f"Scoring error: {type(exc).__name__}",
            )

        score    = cs.final_score
        # Reasoning placeholder: rank=0, updated after boost assigns final rank
        reasoning = reasoning_for(cs, rank=0)

        total   += 1
        counter += 1

        if len(heap) < HEAP_POOL:
            heapq.heappush(heap, (score, counter, cid, reasoning, candidate))
        elif score > heap[0][0]:
            heapq.heapreplace(heap, (score, counter, cid, reasoning, candidate))

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
        f"[rank.py] Scored {total:,} candidates in {t_parse:.1f}s.",
        flush=True,
    )

    # Sort pool → boost layer → score stretch → write CSV
    pool_sorted = sorted(heap, key=lambda x: (-x[0], x[1]))
    pool_for_boost = [(s, cid, r, cand) for s, _, cid, r, cand in pool_sorted]

    print(f"[rank.py] Applying NDCG@10 boost layer to top {len(pool_for_boost)}...", flush=True)
    boosted = apply_ndcg10_boost(pool_for_boost)

    # Re-generate reasoning with final rank
    final_rows = []
    for rank, (_, cid, _old_r, cand) in enumerate(boosted, start=1):
        try:
            cs = score_candidate(cand)
            r  = reasoning_for(cs, rank=rank)
        except Exception:
            r = _old_r
        final_rows.append((cid, rank, 0.0, r))  # score filled below

    scores_raw = [s for s, *_ in boosted]
    scores     = _stretch_scores(scores_raw)

    submission_rows = [
        (cid, rank, adj_score, reasoning)
        for (cid, rank, _, reasoning), adj_score in zip(final_rows, scores)
    ]

    write_submission(output_path, submission_rows)

    t_total = time.perf_counter() - t_start
    print(f"[rank.py] Done in {t_total:.1f}s")
    print(f"  Rows written : {len(boosted)}")
    if scores:
        print(f"  Score range  : {scores[0]:.4f}  ->  {scores[-1]:.4f}")
    print(f"  Output       : {output_path.resolve()}")


# ── Self-test ─────────────────────────────────────────────────────────────────

def _run_selftest():
    """Quick smoke test: verify the new modular pipeline produces sane results."""
    from src.ranker.scorer    import score_candidate
    from src.ranker.reasoning import reasoning_for

    strong = {
        "candidate_id": "STRONG_001",
        "profile": {
            "current_title": "Senior Machine Learning Engineer",
            "years_of_experience": 7.0,
            "location": "Pune",
            "current_company": "Zomato AI",
        },
        "skills": [
            {"name": "Sentence Transformers", "proficiency": "expert",   "duration_months": 36, "endorsements": 80},
            {"name": "FAISS",                 "proficiency": "advanced", "duration_months": 24, "endorsements": 40},
            {"name": "PyTorch",               "proficiency": "expert",   "duration_months": 48, "endorsements": 60},
            {"name": "Machine Learning",      "proficiency": "expert",   "duration_months": 72, "endorsements": 90},
            {"name": "NLP",                   "proficiency": "advanced", "duration_months": 36, "endorsements": 50},
            {"name": "Embeddings",            "proficiency": "expert",   "duration_months": 30, "endorsements": 70},
            {"name": "Information Retrieval", "proficiency": "advanced", "duration_months": 18, "endorsements": 30},
        ],
        "redrob_signals": {
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.82,
            "interview_completion_rate": 0.90,
            "notice_period_days": 30,
            "github_activity_score": 75,
            "saved_by_recruiters_30d": 10,
            "profile_completeness_score": 95,
            "last_active_date": "2026-06-01",
        },
        "career_history": [
            {"description": "Built production embedding and retrieval pipeline for ranking. "
                            "Deployed vector search with FAISS. A/B tested NDCG improvements."},
            {"description": "Shipped recommendation models. Improved MRR by 15%."},
        ],
        "education": [
            {"degree": "M.Tech", "tier": "tier_1", "field_of_study": "Computer Science"}
        ],
    }

    weak = {
        "candidate_id": "WEAK_001",
        "profile": {"current_title": "Junior Web Developer", "years_of_experience": 1.5, "location": "London"},
        "skills": [{"name": "JavaScript", "proficiency": "intermediate", "duration_months": 12}],
        "redrob_signals": {"open_to_work_flag": False, "recruiter_response_rate": 0.2},
    }

    stuffer = {
        "candidate_id": "STUFFER_001",
        "profile": {"current_title": "Sales Executive", "years_of_experience": 5.0, "location": "Mumbai"},
        "skills": [
            {"name": "Sentence Transformers", "proficiency": "expert", "duration_months": 12},
            {"name": "FAISS", "proficiency": "expert", "duration_months": 12},
            {"name": "PyTorch", "proficiency": "expert", "duration_months": 12},
            {"name": "NLP", "proficiency": "expert", "duration_months": 12},
            {"name": "MLOps", "proficiency": "expert", "duration_months": 12},
        ],
        "redrob_signals": {},
        "career_history": [{"description": "Managed sales territory and met quarterly targets."}],
    }

    print("\n=== MODULE SELF-TEST ===")
    cs_strong  = score_candidate(strong)
    cs_weak    = score_candidate(weak)
    cs_stuffer = score_candidate(stuffer)

    print(f"STRONG  : {cs_strong.final_score:.4f}  |  {reasoning_for(cs_strong, rank=1)}")
    print(f"WEAK    : {cs_weak.final_score:.4f}")
    print(f"STUFFER : {cs_stuffer.final_score:.4f}  is_stuffer={cs_stuffer.is_stuffer}")

    assert cs_strong.final_score > 0.40,  f"Strong too low: {cs_strong.final_score:.4f}"
    assert cs_weak.final_score   < 0.45,  f"Weak too high: {cs_weak.final_score:.4f}"
    assert cs_strong.final_score > cs_weak.final_score, "Strong must outscore weak"
    assert cs_stuffer.is_stuffer,          "Sales Executive stuffer must be detected"
    assert cs_stuffer.final_score <= 0.05, f"Stuffer score too high: {cs_stuffer.final_score:.4f}"

    # ── Stuffer verification: 5 non-tech title + many AI skills profiles ──────
    from src.ranker.features import corroboration_ratio

    _AI_SKILLS = [
        {"name": "Sentence Transformers", "proficiency": "expert",   "duration_months": 12},
        {"name": "FAISS",                 "proficiency": "expert",   "duration_months": 12},
        {"name": "PyTorch",               "proficiency": "expert",   "duration_months": 12},
        {"name": "NLP",                   "proficiency": "expert",   "duration_months": 12},
        {"name": "MLOps",                 "proficiency": "expert",   "duration_months": 12},
        {"name": "Embeddings",            "proficiency": "expert",   "duration_months": 12},
        {"name": "Information Retrieval", "proficiency": "advanced", "duration_months": 12},
    ]

    _STUFFERS_5 = [
        {
            "candidate_id": "STUFFER_HR",
            "profile": {"current_title": "HR Manager", "years_of_experience": 8.0},
            "skills": _AI_SKILLS,
            "career_history": [{"description": "Managed recruitment drives. Conducted interviews. Processed payroll and benefits."}],
        },
        {
            "candidate_id": "STUFFER_LAWYER",
            "profile": {"current_title": "Legal Counsel", "years_of_experience": 6.0},
            "skills": _AI_SKILLS,
            "career_history": [{"description": "Drafted contracts. Advised on employment law. Represented clients in negotiations."}],
        },
        {
            "candidate_id": "STUFFER_LOGISTICS",
            "profile": {"current_title": "Logistics Coordinator", "years_of_experience": 5.0},
            "skills": _AI_SKILLS,
            "career_history": [{"description": "Managed supply chain and shipping schedules. Coordinated with vendors."}],
        },
        {
            "candidate_id": "STUFFER_FINANCE",
            "profile": {"current_title": "Financial Analyst", "years_of_experience": 7.0},
            "skills": _AI_SKILLS,
            "career_history": [{"description": "Prepared P&L statements. Conducted variance analysis. Built financial models in Excel."}],
        },
        {
            "candidate_id": "STUFFER_QA",
            "profile": {"current_title": "QA Engineer", "years_of_experience": 4.0},
            "skills": _AI_SKILLS,
            "career_history": [{"description": "Wrote test plans and manual test cases. Filed bug reports. Ran regression suites."}],
        },
    ]

    print("\n-- Stuffer corroboration verification --")
    print(f"{'CandID':<20} {'corrob':>7} {'final_score':>12} {'is_stuffer':>11}")
    print("-" * 56)
    for cand in _STUFFERS_5:
        corrob = corroboration_ratio(cand)
        cs     = score_candidate(cand)
        print(
            f"{cand['candidate_id']:<20} {corrob:>7.4f}"
            f" {cs.final_score:>12.4f} {str(cs.is_stuffer):>11}"
        )
        assert corrob < 0.3, (
            f"{cand['candidate_id']}: corrob={corrob:.4f} should be < 0.3"
        )
        assert cs.final_score <= 0.05, (
            f"{cand['candidate_id']}: final_score={cs.final_score:.4f} should be <= 0.05"
        )
    print("All stuffer assertions passed [OK]")



if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--test':
        _run_selftest()
    else:
        main()
