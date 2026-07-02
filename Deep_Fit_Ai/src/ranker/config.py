"""
src/ranker/config.py
====================
SINGLE SOURCE OF TRUTH for all constants, weights, regexes, and lookup sets.
ZERO logic — no functions, no conditionals, no imports except re and frozenset.
"""

import re

# ── Reference date (fixed — never datetime.now()) ────────────────────────────
REFERENCE_DATE = '2026-06-16'

# ── Final ranking controls ────────────────────────────────────────────────────
TOP_N     = 100     # rows written to submission.csv
HEAP_POOL = 2000    # min-heap size during streaming

# ── Deployed weight vector (sum = 1.00) ───────────────────────────────────────
# Behavioral is NO LONGER a weight — it is a multiplier in [0.50, 1.15]
# applied after the base weighted sum. The 0.13 slot it occupied is
# redistributed to role_substance and exp_score.
WEIGHTS = {
    'role_substance':      0.28,  # career-text: retrieval/ranking/ML/recsys
    'skill_corroboration': 0.08,  # AI skill tags corroborated by descriptions
    'exp_score':           0.22,  # years-of-experience band (5-9 yrs target)
    'nlp_ir_signal':       0.08,  # NLP/IR background (explicit JD requirement)
    'product_score':       0.08,  # product company vs consulting penalty
    'recency_score':       0.05,  # currently shipping code vs non-coding lead
    'edu_score':           0.05,  # education tier + degree + AI field
    'loc_score':           0.03,  # location fit (Pune/Noida preferred)
    # Remaining 0.13 is absorbed into the behavioral MULTIPLIER (not additive)
}

# ── Behavioral multiplier bounds ──────────────────────────────────────────────
BEHAV_FLOOR = 0.50   # worst: unavailable / unresponsive candidate
BEHAV_CAP   = 1.15   # best: highly engaged candidate (+15%)

# ── Job Description targets ───────────────────────────────────────────────────
JD = {
    'target_yoe_min': 5,
    'target_yoe_max': 9,
    'locations': ['pune', 'noida', 'delhi', 'ncr', 'mumbai', 'bangalore', 'bengaluru',
                  'hyderabad', 'remote'],
}

# ── Title classification regexes ──────────────────────────────────────────────
STRONG_TITLE_RE = re.compile(
    r'\b(ml engineer|machine learning engineer|ai engineer|'
    r'applied (ml |ai )?scientist|ai research(er| engineer)?|'
    r'research engineer|data scientist|ml scientist|nlp engineer|'
    r'ml.?ops engineer|deep learning|staff ml|senior ml|senior ai|'
    r'ai specialist|search engineer|relevance engineer|'
    r'recommendation engineer|recommender)\b', re.I
)

ADJACENT_TITLE_RE = re.compile(
    r'\b(data engineer|senior data engineer|analytics engineer|'
    r'backend engineer|software engineer|full.?stack|sde\b|'
    r'platform engineer|developer|programmer)\b', re.I
)

NONTECH_TITLE_RE = re.compile(
    r'\b(hr\b|human.?resource|recruit|talent.?acquisition|sales\b|'
    r'marketing|content.?writer|content.?creator|graphic.?design|ui.?/?ux.?design|brand\b|'
    r'account(ant|s.?payable|s.?receivable)|finance\b|financial.?analyst|'
    r'mechanical.?engineer|civil.?engineer|site.?engineer|electrical.?engineer|'
    r'operations.?manager|operations.?executive|teacher|tutor|nurse|doctor|'
    r'physician|lawyer|legal\b|paralegal|business.?analyst|business.?development|'
    r'customer.(support|success|service)|administrat|office.?manager|'
    r'procurement|supply.?chain|logistics.?coordinator|warehouse|'
    r'project.?manager|program.?manager|qa.?engineer|quality.?assurance|'
    r'test.?engineer(?!.{0,15}(ml|ai))|executive.?assistant|receptionist|'
    r'event.(manager|coordinator)|travel|hospitality|chef\b|retail\b)', re.I
)

# ── Career-text substance area regexes ────────────────────────────────────────
SUBSTANCE_AREAS = {
    'retrieval_embeddings': re.compile(
        r'\b(embedding|embeddings|dense retrieval|semantic search|'
        r'sentence.transformer|bge|e5|vector search|nearest neighbor|'
        r'\bann\b|\bknn\b|faiss|pinecone|weaviate|qdrant|milvus)\b', re.I
    ),
    'ranking_ltr': re.compile(
        r'\b(ranking|re.?rank|learning to rank|\bltr\b|ndcg|mrr|'
        r'relevance tuning|search relevance|search quality)\b', re.I
    ),
    'recommendation': re.compile(
        r'\b(recommend|recommender|recommendation|recsys|'
        r'personali[sz]ation|collaborative filtering|matching engine)\b', re.I
    ),
    'search_ir': re.compile(
        r'\b(information retrieval|inverted index|bm25|elasticsearch|'
        r'opensearch|lucene|solr|query understanding|hybrid search|'
        r'full.text search)\b', re.I
    ),
    'applied_ml_prod': re.compile(
        r'\b(deployed|in production|productioni[sz]ed|served|serving|'
        r'model serving|trained and deployed|shipped a model|ml pipeline|'
        r'feature store|a/b test|online metric)\b', re.I
    ),
}

# ── NLP/IR signal regexes ─────────────────────────────────────────────────────
NLP_IR_RE = re.compile(
    r'\b(nlp|natural language|information retrieval|text ranking|'
    r'search relevance|recommend|recommender|language model|word embedding|'
    r'tf.?idf|question answering|query understanding|semantic search|'
    r'retrieval)\b', re.I
)

CV_SPEECH_RE = re.compile(
    r'\b(computer vision|image classification|object detection|segmentation|'
    r'opencv|speech recognition|\btts\b|\basr\b|text.to.speech|robotics)\b', re.I
)

# ── Production-recency regexes ────────────────────────────────────────────────
BUILD_SHIP_RE = re.compile(
    r'\b(built|build|shipped|deployed|implemented|developed|engineered|'
    r'wrote|coded|production|launched|delivered|owned)\b', re.I
)

NONCODING_LEAD_RE = re.compile(
    r'\b(architect|tech lead|technical lead|engineering manager|head of|'
    r'director|principal architect|leadership|strategy|roadmap|stakeholder)\b', re.I
)

# ── Consulting firm penalty set ───────────────────────────────────────────────
CONSULTING_FIRMS = frozenset({
    'tcs', 'tata consultancy', 'infosys', 'wipro', 'accenture',
    'cognizant', 'capgemini', 'mindtree', 'lti', 'ltimindtree',
    'hcl technologies', 'tech mahindra', 'mphasis', 'hexaware',
    'niit technologies', 'persistent systems', 'mastech',
})

# ── Hard AI skill SET — lowercase, for the corroboration gate ───────────────
# (used by features.score_skill_corroboration: 4+ of these with 0 description
#  substance = keyword stuffer)
HARD_AI_SKILLS = frozenset({
    'machine learning', 'deep learning', 'nlp', 'natural language processing',
    'llm', 'transformers', 'computer vision', 'pytorch', 'tensorflow',
    'hugging face', 'bert', 'gpt', 'information retrieval', 'recommender systems',
    'semantic search', 'embeddings', 'neural networks', 'scikit-learn', 'mlops',
    'sentence transformers', 'faiss', 'qdrant', 'weaviate', 'pinecone',
    'llm fine-tuning', 'rag', 'feature engineering', 'xgboost', 'lightgbm',
})

# ── Weighted AI skills — for boost layer in rank.py ──────────────────────────
HARD_AI_SKILLS_WEIGHTED = {
    'Sentence Transformers': 1.0,
    'FAISS': 1.0,
    'Embeddings': 1.0,
    'Information Retrieval': 1.0,
    'Qdrant': 0.95,
    'Weaviate': 0.95,
    'Pinecone': 0.90,
    'LLM Fine-tuning': 0.90,
    'RAG': 0.90,
    'PyTorch': 0.85,
    'TensorFlow': 0.80,
    'Hugging Face': 0.80,
    'NLP': 0.80,
    'MLOps': 0.75,
    'Machine Learning': 0.70,
    'Deep Learning': 0.70,
    'Scikit-learn': 0.65,
    'XGBoost': 0.60,
    'LightGBM': 0.60,
}

# ── Location lookup ───────────────────────────────────────────────────────────
PREFERRED_LOCATIONS = {
    'pune': 1.0, 'noida': 1.0, 'delhi': 0.9, 'ncr': 0.9, 'new delhi': 0.9,
    'gurugram': 0.9, 'gurgaon': 0.9, 'bangalore': 0.8, 'bengaluru': 0.8,
    'mumbai': 0.8, 'hyderabad': 0.8, 'remote': 0.7, 'anywhere': 0.7,
    'india': 0.5,
}

# ── Education scoring ─────────────────────────────────────────────────────────
EDU_DEGREE_SCORES = {
    'phd': 1.0, 'ph.d': 1.0, 'doctorate': 1.0,
    'm.tech': 0.90, 'mtech': 0.90, 'm.e': 0.85, 'me': 0.85,
    'ms': 0.85, 'm.s': 0.85, 'msc': 0.80, 'm.sc': 0.80,
    'mba': 0.65,
    'b.tech': 0.75, 'btech': 0.75, 'b.e': 0.70, 'be': 0.70,
    'bsc': 0.60, 'b.sc': 0.60, 'ba': 0.45,
}

TIER1_INSTITUTES = frozenset({
    'iit', 'iim', 'iisc', 'bits pilani', 'nit', 'iiit',
    'mit', 'stanford', 'carnegie mellon', 'cmu', 'berkeley', 'oxford',
    'cambridge', 'iitb', 'iitd', 'iitm', 'iitk', 'iith',
})

AI_FIELDS = frozenset({
    'computer science', 'machine learning', 'artificial intelligence',
    'data science', 'statistics', 'mathematics', 'information technology',
    'electrical engineering', 'computational linguistics', 'natural language processing',
})

# ── Boost-layer constants (used only in rank.py NDCG@10 layer) ────────────────
TITLE_EXACT_BOOST_SET = frozenset({
    'senior machine learning engineer', 'staff machine learning engineer',
    'principal machine learning engineer', 'ml engineer',
    'recommendation systems engineer', 'applied ml engineer',
    'applied scientist', 'nlp engineer', 'search engineer',
    'ai research engineer', 'senior ai engineer',
})

RETRIEVAL_SKILLS = frozenset({
    'faiss', 'sentence transformers', 'embeddings', 'information retrieval',
    'qdrant', 'weaviate', 'pinecone', 'milvus', 'semantic search',
    'dense retrieval', 'vector search',
})

NON_AI_TITLES = frozenset({
    'hr', 'sales', 'marketing', 'teacher', 'lawyer', 'nurse',
    'doctor', 'content writer', 'graphic designer',
})

CAREER_KEYWORDS = [
    'production', 'deployed', 'a/b test', 'embedding', 'retrieval',
    'ranking', 'recommendation', 'vector', 'faiss', 'transformer',
    'fine-tun', 'ndcg', 'mrr', 'recsys', 'search quality',
]
CAREER_KEYWORD_PER_HIT  = 0.01
CAREER_KEYWORD_MAX_BONUS = 0.08
