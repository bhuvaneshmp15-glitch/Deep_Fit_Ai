"""
src/ranker/io_utils.py
======================
WHAT THIS MODULE DOES:
  Provides stream_candidates(), a generator that reads candidates.jsonl
  line by line with constant memory usage. Also provides write_submission()
  for writing the final CSV.

WHAT THIS MODULE DOES NOT DO:
  Score, rank, or reason about candidates. No scoring logic here.

WHY THIS BOUNDARY EXISTS:
  I/O concerns (file encoding, error handling, progress reporting) are
  separated from scoring logic so both can evolve independently. In
  particular, the streaming pattern here ensures the ranker works on
  files of any size without OOM issues.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Generator, Tuple


def stream_candidates(
    path: Path,
) -> Generator[Tuple[dict, str], None, None]:
    """
    Yield (candidate_dict, candidate_id) tuples from a JSONL file,
    one line at a time (constant memory).

    Skips blank lines and lines that fail JSON parsing (logs count).
    """
    errors = 0
    with path.open('r', encoding='utf-8', errors='replace') as fh:
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
                f'unknown'
            )
            yield candidate, cid

    if errors:
        print(f"[io_utils] Warning: {errors} JSON parse error(s) skipped.", file=sys.stderr)


def write_submission(
    path: Path,
    rows: list,   # list of (candidate_id, rank, score, reasoning)
) -> None:
    """
    Write submission CSV with columns: candidate_id, rank, score, reasoning.
    Creates parent directories if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for cid, rank, score, reasoning in rows:
            writer.writerow([cid, rank, f'{score:.6f}', reasoning])
