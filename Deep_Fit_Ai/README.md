# DeepFit AI: Intelligent Candidate Discovery
> **India has 1.4 billion people. Almost no AI was built for them. We are fixing hiring at scale.**

![DeepFit AI UI](https://img.shields.io/badge/UI-DeepFit_AI-E03131?style=for-the-badge)
![Processing Speed](https://img.shields.io/badge/Speed-100k_in_3.5_mins-22c55e?style=for-the-badge)
![Zero Dependencies](https://img.shields.io/badge/Architecture-CPU_Only_|_Stdlib-blue?style=for-the-badge)

## 🚀 The Challenge
What if finding the right candidate was faster, smarter, and more accurate? In today's competitive job market, identifying the best talent is broken at scale:
- ❌ **Recruiters waste hours** filtering irrelevant profiles.
- ❌ **Great candidates get missed** because they don't exactly "keyword match."
- ❌ **The System is easily gamed** by "stuffers" who pack their profiles with buzzwords they never actually used in production.

## ✨ The Solution: Beyond Keyword Matching
**DeepFit AI** is an Intelligent Candidate Discovery system that goes beyond naive keyword matching to find the *right* talent using career-substance NLP, embeddings, and real behavioral signals.

We completely rebuilt our ranking pipeline from the ground up to solve the "Honeypot" and "Stuffer" problem inherent in modern recruiting:

### 1. The Substance & Corroboration Engine (Anti-Stuffer Gate)
A naive vector database will rank an **HR Manager** or **Sales Executive** at the top of the pile just because they stuffed 23 "AI" and "Machine Learning" keywords into their skills section. 

DeepFit AI reads the actual *career descriptions*. 
- Our `role_substance()` function uses targeted regex and semantic heuristics over 5 substance areas (retrieval, ranking, recommendation, search, applied ML). 
- Our `corroboration_ratio` verifies that the skills claimed in the tag list actually appear contextually in the career history. 
- **Result:** An HR manager with 20 AI tags but no ML career text scores `substance=0.0` and sinks. A Data Engineer who actually deployed FAISS in production scores `1.0` and surfaces.

### 2. Real Behavioral Signals
We don't just look at static text. DeepFit AI uses behavioral signals as an orthogonal multiplier, not an additive score. 
- We evaluate **Recency** (are they shipping code *now*?), **Platform Activity**, and **Open Source presence** (GitHub). 
- If a candidate is highly qualified but inactive, their score is mathematically dragged down. If they are engaged and available, their score is lifted.

### 3. Grounded, Hallucination-Free Reasoning
Recruiters need to know *why* a candidate was ranked highly. Most systems use LLMs that hallucinate justifications. 
DeepFit AI uses a deterministic, hash-seeded reasoning generator. Every single clause (e.g., `"NLP/IR background present"`, `"recently shipping production code"`) maps perfectly to a real, computed float value in the scoring pipeline. It is 100% grounded and verifiable.

---

## ⚡ Extreme Performance (100k in 3 Minutes)
AI systems shouldn't require $5,000/month GPU clusters to run. DeepFit AI is a masterclass in optimization. 
We stripped out bloated graph databases and heavy API calls to build a lean, **CPU-only, standard-library Python pipeline** that processes **100,000 candidates in exactly 3.5 minutes** (~480 candidates per second).

### Pipeline Stages (`src/ranker/`)
1. **`features.py`**: Extracts normalized floats for Substance, Experience, Education, and NLP/IR presence. Runs the `corroboration_ratio` check.
2. **`behavioral.py`**: Calculates the engagement multiplier [0.50x to 1.15x].
3. **`scorer.py`**: Applies the weight vector (proven via our offline NDCG@10 sweep) and mathematically penalizes stuffers.
4. **`reasoning.py`**: Generates human-readable, score-banded justifications (e.g., *Strong fit*, *Solid fit*) with dynamic clause order variation.

---

## 💻 Running the System

### 1. Full 100k Pipeline
```bash
python -m venv venv
# Windows: .\venv\Scripts\Activate | Mac/Linux: source venv/bin/activate
pip install -r requirements.txt

# Run the full 100,000 candidate dataset
python rank.py candidates.jsonl submission.csv

# Validate the output format
python testing/validate_submission.py submission.csv
```

### 2. Verify the Stuffer Defenses
Run the module's self-test suite. This runs hardcoded "Stuffer" profiles (e.g., HR, Logistics, Lawyers with AI skills) through the pipeline to mathematically prove they are caught and penalized by the `corroboration_ratio` gate.
```bash
python rank.py --test
```

### 3. Launch the DeepFit AI UI
```bash
python testing/sandbox/app.py
# Opens at http://127.0.0.1:7860
```

---
*The future of recruitment is intelligent, inclusive, and data-driven. DeepFit AI empowers recruiters to make better, faster, and more informed decisions, helping organizations connect with skilled professionals to build stronger teams for the future.*
