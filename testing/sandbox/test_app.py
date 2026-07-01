"""
Quick import smoke-test for sandbox/app.py without launching the server.
Verifies all src.ranker modules import correctly and the ranking pipeline runs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ranker.scorer    import score_candidate
from src.ranker.reasoning import reasoning_for
from src.ranker.honeypot  import is_honeypot, penalty
from src.ranker.config    import WEIGHTS, JD

import json

data_path = Path(__file__).parent / "demo_data" / "demo_candidates.jsonl"
candidates = []
with data_path.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            candidates.append(json.loads(line))

print(f"Loaded {len(candidates)} demo candidates")

scored = []
honeypots = 0
stuffers = 0

for cand in candidates:
    cs = score_candidate(cand)
    flagged, _ = is_honeypot(cand)
    if cs.is_stuffer:
        stuffers += 1
    elif flagged:
        honeypots += 1
    else:
        scored.append((cs.final_score, cs, cand))

scored.sort(key=lambda x: -x[0])

print(f"Qualified: {len(scored)}  Honeypots: {honeypots}  Stuffers: {stuffers}")
print()
print("Top 5:")
for rank, (score, cs, cand) in enumerate(scored[:5], 1):
    r = reasoning_for(cs, rank=rank)
    p = cand.get('profile', {})
    print(f"  #{rank}  {p.get('anonymized_name','?')}  score={score:.4f}")
    print(f"       {r[:100]}")

print("\n[OK] sandbox/app.py import test passed")
