"""
Verification script: check that TCS/Infosys candidates and CV-only candidates
have dropped significantly in the new ranking vs. the JD signals.
"""
import json
import csv
import sys
sys.path.insert(0, '.')
from rank import score_nlp_ir, score_product_vs_consulting, CONSULTING_FIRMS

# ── 1. Load submission ──────────────────────────────────────────────────────
with open('submission.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
top_ids = {r['candidate_id']: int(r['rank']) for r in rows}

# ── 2. Scan all candidates and collect consulting + CV-only ones ────────────
consulting_cands = []   # (score_signal, candidate_id, company, in_top100, rank)
cv_only_cands    = []

with open('candidates.jsonl', encoding='utf-8') as f:
    for line in f:
        c = json.loads(line)
        cid = c['candidate_id']

        # product_score
        product_score = score_product_vs_consulting(c)

        # nlp_ir signal
        p = c.get('profile') or {}
        desc_parts = [p.get('headline', '') or '', p.get('summary', '') or '']
        for key in ('career_history', 'work_experience', 'experience'):
            for ch in (c.get(key) or []):
                if isinstance(ch, dict):
                    desc_parts.append(ch.get('description', '') or '')
                    desc_parts.append(ch.get('responsibilities', '') or '')
        full_text = ' '.join(desc_parts)
        nlp_ir = score_nlp_ir(full_text)

        # Collect consulting-heavy candidates (product_score <= 0.40)
        if product_score <= 0.40:
            career = c.get('career_history') or []
            companies = [ch.get('company', '?') for ch in career if isinstance(ch, dict)]
            in_top = cid in top_ids
            rank   = top_ids.get(cid, 999)
            consulting_cands.append((product_score, cid, companies[:2], in_top, rank))

        # Collect CV-only candidates (nlp_ir == 0.1)
        if nlp_ir == 0.1:
            in_top = cid in top_ids
            rank   = top_ids.get(cid, 999)
            cv_only_cands.append((cid, in_top, rank))

# ── 3. Report ────────────────────────────────────────────────────────────────
consulting_cands.sort(key=lambda x: x[0])   # worst first
cv_only_cands.sort(key=lambda x: x[2])       # by rank

print(f"\n{'='*60}")
print(f"CONSULTING-HEAVY CANDIDATES (product_score <= 0.40)")
print(f"{'='*60}")
print(f"Total found: {len(consulting_cands)}")
in_top = sum(1 for _, _, _, it, _ in consulting_cands if it)
print(f"In top-100 : {in_top}  {'<-- PROBLEM' if in_top > 0 else '-- GOOD: none in top-100'}")
if in_top:
    print("\nOnes that made it to top-100:")
    for ps, cid, cos, it, rank in consulting_cands:
        if it:
            print(f"  Rank {rank:3d} | product_score={ps:.2f} | {cid} | {cos}")

print(f"\n{'='*60}")
print(f"CV-ONLY CANDIDATES (nlp_ir=0.10 — explicit JD negative)")
print(f"{'='*60}")
print(f"Total found: {len(cv_only_cands)}")
in_top_cv = sum(1 for _, it, _ in cv_only_cands if it)
print(f"In top-100 : {in_top_cv}  {'<-- PROBLEM' if in_top_cv > 0 else '-- GOOD: none in top-100'}")
if in_top_cv:
    print("\nCV-only ones that made top-100:")
    for cid, it, rank in cv_only_cands:
        if it:
            print(f"  Rank {rank:3d} | {cid}")

print(f"\n{'='*60}")
print("TOP-20 TITLES SANITY CHECK")
print(f"{'='*60}")
for r in rows[:20]:
    print(f"  #{int(r['rank']):2d}  {r['candidate_id']}  score={r['score']}  {r['reasoning'][:75]}")
