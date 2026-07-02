"""
src/ranker/__init__.py
======================
Public surface of the ranker package.

Importing from this package:
    from src.ranker import score_candidate, reasoning_for
"""

from .scorer import score_candidate
from .reasoning import reasoning_for
from .schema import CandidateScore, FeatureSet

__all__ = ['score_candidate', 'reasoning_for', 'CandidateScore', 'FeatureSet']
