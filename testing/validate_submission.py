"""
validate_submission.py — Validates the submission CSV against competition rules.

Usage:
    python validate_submission.py submission.csv
"""
import csv
import sys

def validate(filepath):
    errors = []

    with open(filepath, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Check header
    expected_cols = {'candidate_id', 'rank', 'score', 'reasoning'}
    actual_cols = set(rows[0].keys()) if rows else set()
    if not expected_cols.issubset(actual_cols):
        errors.append(f"Missing columns: {expected_cols - actual_cols}")

    # Check exactly 100 rows
    if len(rows) != 100:
        errors.append(f"Expected 100 rows, got {len(rows)}")

    # Check ranks 1-100
    ranks = [int(r['rank']) for r in rows]
    if ranks != list(range(1, 101)):
        errors.append(f"Ranks must be 1-100 in order. Got {ranks[:5]}...{ranks[-5:]}")

    # Check scores are non-increasing
    scores = [float(r['score']) for r in rows]
    for i in range(1, len(scores)):
        if scores[i] > scores[i - 1] + 1e-9:
            errors.append(f"Scores not non-increasing at rank {i+1}: "
                          f"{scores[i-1]:.6f} -> {scores[i]:.6f}")
            break

    # Check no duplicate candidate_ids
    cids = [r['candidate_id'] for r in rows]
    if len(set(cids)) != len(cids):
        dupes = [c for c in cids if cids.count(c) > 1]
        errors.append(f"Duplicate candidate_ids: {set(dupes)}")

    # Check all scores are positive
    if any(s <= 0 for s in scores):
        errors.append("Some scores are <= 0")

    # Check reasoning is non-empty
    empty_reasons = [r['rank'] for r in rows if not r['reasoning'].strip()]
    if empty_reasons:
        errors.append(f"Empty reasoning at ranks: {empty_reasons[:5]}")

    # Check candidate_id format
    bad_ids = [r['candidate_id'] for r in rows
               if not r['candidate_id'].startswith('CAND_')]
    if bad_ids:
        errors.append(f"Non-standard candidate_ids: {bad_ids[:3]}...")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("Submission is valid.")
        return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python validate_submission.py <submission.csv>")
        sys.exit(1)
    ok = validate(sys.argv[1])
    sys.exit(0 if ok else 1)
