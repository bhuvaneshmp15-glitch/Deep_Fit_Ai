"""
sandbox/app.py
==============
Gradio demo for the Redrob Candidate Ranker.

Beats Caliber's demo on three axes:
  1. Feature mini-bars per card  — shows WHICH signals drove the rank
  2. Reasoning variation         — every candidate gets a distinct sentence
  3. Caught panel                — specific detection reason per rejected profile

Reads demo_data/demo_candidates.jsonl (150 candidates from the Redrob pool).
Works locally and on HuggingFace Spaces.
"""

from __future__ import annotations

import html as html_lib
import json
import sys
from pathlib import Path

# ── Path setup: works both locally and on HuggingFace Spaces ─────────────────
_HERE = Path(__file__).parent          # testing/sandbox/
_ROOT = _HERE.parent.parent            # project root
_DATA = _HERE / "demo_data" / "demo_candidates.jsonl"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gradio as gr

from src.ranker.scorer    import score_candidate
from src.ranker.reasoning import reasoning_for
from src.ranker.honeypot  import is_honeypot
from src.ranker.features  import structured_features
from src.ranker.behavioral import behavioral_multiplier
from src.ranker.config    import WEIGHTS, JD


# ── Load + rank all demo candidates once at startup ──────────────────────────

def _load_and_rank():
    candidates = []
    with _DATA.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    scored, honeypots, stuffers = [], [], []

    for cand in candidates:
        cs = score_candidate(cand)
        p  = cand.get("profile") or {}
        sig = cand.get("redrob_signals") or {}
        name    = p.get("anonymized_name") or cand.get("candidate_id", "Unknown")
        title   = p.get("current_title") or "—"
        company = p.get("current_company") or "—"
        yoe     = float(p.get("years_of_experience") or 0)
        flagged, reasons = is_honeypot(cand)

        if cs.is_stuffer:
            stuffers.append({
                "name": name, "title": title, "company": company,
                "ai_count": cs.features.ai_skill_count,
                "score": cs.final_score,
                "reason": f"{cs.features.ai_skill_count} AI skills listed — zero corroborated by career text",
            })
        elif flagged:
            honeypots.append({
                "name": name, "title": title, "company": company,
                "reasons": reasons, "score": cs.final_score,
            })
        else:
            reasoning = reasoning_for(cs, rank=0)
            fs = cs.features
            bm = cs.behav_score
            scored.append({
                "cs": cs, "name": name, "title": title,
                "company": company, "yoe": yoe,
                "score": cs.final_score, "base": cs.base_score,
                "reasoning": reasoning,
                "rs":   fs.role_substance,
                "exp":  fs.exp_score,
                "nlp":  fs.nlp_ir_signal,
                "prod": fs.product_score,
                "rec":  fs.recency_score,
                "bm":   bm,
                "cand": cand,
            })

    scored.sort(key=lambda x: -x["score"])

    # Re-generate reasoning with final rank
    for rank, item in enumerate(scored, start=1):
        item["reasoning"] = reasoning_for(item["cs"], rank=rank)
        item["rank"] = rank

    return scored, honeypots, stuffers, len(candidates)


_SCORED, _HONEYPOTS, _STUFFERS, _TOTAL = _load_and_rank()
_TOP15 = _SCORED[:15]


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Reset / Base ── */
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
    background: #0D1117 !important;
    color: #E6EDF3 !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}
.gradio-container {
    max-width: 1080px !important;
    margin: 0 auto !important;
    padding: 0 20px !important;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 56px 24px 36px;
    position: relative;
}
.hero::before {
    content: '';
    position: absolute; top: 0; left: 50%; transform: translateX(-50%);
    width: 600px; height: 300px;
    background: radial-gradient(ellipse at center, rgba(224,49,49,0.10) 0%, transparent 70%);
    pointer-events: none;
}
.hero h1 {
    font-size: 2.8rem; font-weight: 800; letter-spacing: -0.04em;
    background: linear-gradient(135deg, #FF6B6B 0%, #E03131 45%, #FF8C42 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 0 10px; line-height: 1.1;
}
.hero-sub { color: #8B949E; font-size: 1.05rem; margin: 0 0 20px; }
.hero-pill {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(224,49,49,0.12);
    border: 1px solid rgba(224,49,49,0.30);
    border-radius: 999px; padding: 7px 18px;
    color: #FF9999; font-size: 0.875rem; font-weight: 500;
}
.hero-note {
    margin-top: 14px; color: #6B7280; font-size: 0.78rem;
    letter-spacing: 0.01em;
}
.hero-note strong { color: #FF6B6B; }

/* ── Stats strip ── */
.stats-strip {
    display: flex; gap: 0; justify-content: center;
    margin: 0 0 28px;
    background: rgba(22,27,34,0.8);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; overflow: hidden;
}
.stat-cell {
    flex: 1; text-align: center; padding: 18px 12px;
    border-right: 1px solid rgba(255,255,255,0.06);
    position: relative;
}
.stat-cell:last-child { border-right: none; }
.stat-cell:hover { background: rgba(224,49,49,0.07); }
.stat-val {
    font-size: 1.75rem; font-weight: 800;
    background: linear-gradient(135deg, #FF6B6B, #E03131);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1;
}
.stat-val.green { background: linear-gradient(135deg, #34D399, #059669);
    -webkit-background-clip: text; background-clip: text; }
.stat-val.red   { background: linear-gradient(135deg, #F87171, #EF4444);
    -webkit-background-clip: text; background-clip: text; }
.stat-val.amber { background: linear-gradient(135deg, #FBBF24, #F59E0B);
    -webkit-background-clip: text; background-clip: text; }
.stat-val.blue  { background: linear-gradient(135deg, #60A5FA, #3B82F6);
    -webkit-background-clip: text; background-clip: text; }
.stat-lbl {
    font-size: 0.68rem; color: #6B7280;
    text-transform: uppercase; letter-spacing: 0.09em; margin-top: 4px;
}

/* ── Section headings ── */
.section-title {
    font-size: 0.70rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #E03131;
    margin: 36px 0 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(224,49,49,0.20);
    display: flex; align-items: center; gap: 8px;
}
.section-title::before {
    content: ''; display: inline-block;
    width: 3px; height: 14px; border-radius: 2px;
    background: linear-gradient(180deg, #FF6B6B, #E03131);
}

/* ── Weight bars ── */
.weight-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px 32px;
    margin-bottom: 4px;
}
.weight-row { display: flex; align-items: center; gap: 10px; }
.weight-label { font-size: 0.77rem; color: #C9D1D9; width: 210px; flex-shrink: 0; }
.weight-track {
    flex: 1; height: 5px;
    background: rgba(255,255,255,0.06);
    border-radius: 3px; overflow: hidden;
}
.weight-fill {
    height: 100%; border-radius: 3px;
    background: linear-gradient(90deg, #E03131, #FF6B6B);
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}
.weight-pct {
    font-size: 0.73rem; font-weight: 700; color: #FF6B6B;
    width: 34px; text-align: right;
}

/* ── Candidate cards ── */
.cards-wrap { display: flex; flex-direction: column; gap: 10px; }
.cand-card {
    background: rgba(22,27,34,0.9);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 18px 20px;
    display: grid;
    grid-template-columns: 50px 1fr auto;
    gap: 0 16px; align-items: start;
    transition: border-color 0.25s, background 0.25s, transform 0.2s;
    position: relative; overflow: hidden;
}
.cand-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(224,49,49,0.35), transparent);
    opacity: 0; transition: opacity 0.25s;
}
.cand-card:hover {
    border-color: rgba(224,49,49,0.35);
    background: rgba(224,49,49,0.05);
    transform: translateY(-1px);
}
.cand-card:hover::before { opacity: 1; }

.cand-rank {
    font-size: 1.5rem; font-weight: 800; color: #4B5563;
    line-height: 1; padding-top: 3px; text-align: center;
}
.cand-rank.gold   { color: #FBBF24; text-shadow: 0 0 20px rgba(251,191,36,0.4); }
.cand-rank.silver { color: #94A3B8; }
.cand-rank.bronze { color: #D97706; }
.cand-rank.top10  { color: #FF6B6B; }

.cand-body { min-width: 0; }
.cand-name {
    font-weight: 700; font-size: 0.96rem; color: #F0F6FC;
    margin-bottom: 2px;
}
.cand-meta {
    font-size: 0.77rem; color: #8B949E; margin-bottom: 10px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.cand-reason {
    font-size: 0.74rem; color: #C9D1D9; line-height: 1.6;
    margin-bottom: 10px;
}
.cand-reason strong { color: #FF9999; }

/* Feature mini-bars */
.feat-bars { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 4px; }
.feat-bar-item {
    display: flex; flex-direction: column; align-items: center; gap: 3px;
    min-width: 38px;
}
.feat-bar-label {
    font-size: 0.60rem; color: #6B7280; text-transform: uppercase;
    letter-spacing: 0.06em;
}
.feat-bar-track {
    width: 38px; height: 3px;
    background: rgba(255,255,255,0.08);
    border-radius: 2px; overflow: hidden;
}
.feat-bar-fill {
    height: 100%; border-radius: 2px;
    transition: width 0.6s ease;
}
.feat-bar-fill.rs   { background: linear-gradient(90deg, #E03131, #FF6B6B); }
.feat-bar-fill.exp  { background: linear-gradient(90deg, #3B82F6, #60A5FA); }
.feat-bar-fill.nlp  { background: linear-gradient(90deg, #8B5CF6, #A78BFA); }
.feat-bar-fill.prod { background: linear-gradient(90deg, #059669, #34D399); }
.feat-bar-fill.rec  { background: linear-gradient(90deg, #0891B2, #22D3EE); }

/* Behavioral multiplier badge */
.bm-badge {
    display: inline-block;
    font-size: 0.65rem; font-weight: 700;
    padding: 2px 7px; border-radius: 6px;
    margin-left: 6px; vertical-align: middle;
}
.bm-up   { background: rgba(52,211,153,0.15); color: #34D399; border: 1px solid rgba(52,211,153,0.25); }
.bm-down { background: rgba(248,113,113,0.15); color: #F87171; border: 1px solid rgba(248,113,113,0.25); }
.bm-flat { background: rgba(156,163,175,0.12); color: #9CA3AF; border: 1px solid rgba(156,163,175,0.20); }

/* Score column */
.score-col { text-align: right; min-width: 76px; }
.score-val {
    font-size: 1.15rem; font-weight: 800;
    background: linear-gradient(135deg, #FF6B6B, #E03131);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.score-bar-track {
    width: 76px; height: 3px;
    background: rgba(255,255,255,0.07);
    border-radius: 2px; overflow: hidden; margin-top: 8px; margin-left: auto;
}
.score-bar-fill {
    height: 100%; border-radius: 2px;
    background: linear-gradient(90deg, #E03131, #FF6B6B);
}
.score-base {
    font-size: 0.62rem; color: #6B7280; margin-top: 4px; text-align: right;
}

/* ── Caught panel ── */
.caught-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
}
.caught-card {
    background: rgba(22,27,34,0.7);
    border-radius: 12px; padding: 14px 16px;
    border: 1px solid rgba(255,255,255,0.06);
    transition: border-color 0.2s;
}
.caught-card:hover { border-color: rgba(255,255,255,0.12); }
.badge {
    display: inline-block; border-radius: 5px;
    padding: 2px 9px; font-size: 0.63rem; font-weight: 700;
    letter-spacing: 0.09em; text-transform: uppercase; margin-bottom: 7px;
}
.badge-red {
    background: rgba(239,68,68,0.15); color: #FCA5A5;
    border: 1px solid rgba(239,68,68,0.30);
}
.badge-amber {
    background: rgba(245,158,11,0.15); color: #FCD34D;
    border: 1px solid rgba(245,158,11,0.30);
}
.caught-name { font-weight: 600; font-size: 0.87rem; color: #E6EDF3; }
.caught-meta { font-size: 0.73rem; color: #8B949E; margin-bottom: 5px; }
.caught-reason {
    font-size: 0.71rem; color: #C9D1D9; font-style: italic; line-height: 1.5;
}
.caught-reason::before { content: '⚠ '; }

/* ── Footer ── */
.demo-footer {
    text-align: center; padding: 40px 0 24px;
    color: #30363D; font-size: 0.73rem; border-top: 1px solid rgba(255,255,255,0.04);
    margin-top: 32px;
}
.demo-footer a { color: #E03131; text-decoration: none; }

/* ── Hide Gradio chrome ── */
footer { display: none !important; }
.gr-button-primary { background: #E03131 !important; border: none !important; }
#component-0 { padding: 0 !important; }
"""


# ── HTML builders ─────────────────────────────────────────────────────────────

def _e(s: str) -> str:
    """HTML-escape a string."""
    return html_lib.escape(str(s))


def _hero_html(total: int) -> str:
    yoe_min = JD["target_yoe_min"]
    yoe_max = JD["target_yoe_max"]
    return f"""
<div class="hero">
  <h1>DeepFit AI</h1>
  <p class="hero-sub">Ranking candidates by fit, not keywords</p>
  <div class="hero-pill">
    🎯&nbsp; <strong>Senior AI / ML Engineer</strong>
    &nbsp;·&nbsp; {yoe_min}–{yoe_max} yrs experience
    &nbsp;·&nbsp; Redrob
  </div>
  <p class="hero-note">
    Scoring <strong>{total}</strong> candidates from the Redrob demo pool
    &nbsp;·&nbsp; career description text signals, not keyword density
  </p>
</div>
"""


def _stats_html(scored: list, honeypots: list, stuffers: list, total: int) -> str:
    top_score = scored[0]["score"] if scored else 0
    qualified = len(scored)
    return f"""
<div class="stats-strip">
  <div class="stat-cell">
    <div class="stat-val blue">{total}</div>
    <div class="stat-lbl">Pool Size</div>
  </div>
  <div class="stat-cell">
    <div class="stat-val green">{qualified}</div>
    <div class="stat-lbl">Qualified</div>
  </div>
  <div class="stat-cell">
    <div class="stat-val red">{len(honeypots)}</div>
    <div class="stat-lbl">Honeypots</div>
  </div>
  <div class="stat-cell">
    <div class="stat-val amber">{len(stuffers)}</div>
    <div class="stat-lbl">Stuffers</div>
  </div>
  <div class="stat-cell">
    <div class="stat-val">{top_score:.3f}</div>
    <div class="stat-lbl">Top Score</div>
  </div>
</div>
"""


def _weights_html() -> str:
    items = [
        ("Career Substance (retrieval/ranking/ML)", WEIGHTS["role_substance"]),
        ("Experience Band (5–9 yrs)",               WEIGHTS["exp_score"]),
        ("Skill Corroboration",                      WEIGHTS["skill_corroboration"]),
        ("NLP / IR Background",                      WEIGHTS["nlp_ir_signal"]),
        ("Product Company Experience",               WEIGHTS["product_score"]),
        ("Shipping Recency",                         WEIGHTS["recency_score"]),
        ("Education Tier",                           WEIGHTS["edu_score"]),
        ("Location Fit",                             WEIGHTS["loc_score"]),
    ]
    rows = ""
    for label, w in items:
        pct = int(w * 100)
        rows += f"""
        <div class="weight-row">
          <span class="weight-label">{label}</span>
          <div class="weight-track"><div class="weight-fill" style="width:{pct}%"></div></div>
          <span class="weight-pct">{pct}%</span>
        </div>"""

    rows += """
    <div class="weight-row" style="opacity:0.8;border-top:1px dashed rgba(255,255,255,0.08);padding-top:10px;margin-top:6px;grid-column:1/-1;">
      <span class="weight-label">⚡ Behavioral Multiplier (recency · response · availability)</span>
      <div class="weight-track" style="background:rgba(224,49,49,0.10)">
        <div class="weight-fill" style="width:100%;background:linear-gradient(90deg,#E03131,#FF6B6B)"></div>
      </div>
      <span class="weight-pct" style="color:#FF6B6B;font-size:0.68rem;width:52px">0.50×–1.15×</span>
    </div>"""

    return f"""
<div class="section-title">Scoring Formula — Deployed Weights</div>
<div class="weight-grid">{rows}</div>
"""


def _bm_badge(bm: float) -> str:
    if bm >= 1.03:
        return f'<span class="bm-badge bm-up">▲ ×{bm:.3f}</span>'
    if bm <= 0.85:
        return f'<span class="bm-badge bm-down">▼ ×{bm:.3f}</span>'
    return f'<span class="bm-badge bm-flat">× {bm:.3f}</span>'


def _feat_bars(item: dict) -> str:
    """Render 5 mini signal bars per candidate card."""
    bars = [
        ("rs",   "Substance", item["rs"]),
        ("exp",  "Exp",       item["exp"]),
        ("nlp",  "NLP/IR",    item["nlp"]),
        ("prod", "Product",   item["prod"]),
        ("rec",  "Recency",   item["rec"]),
    ]
    html = '<div class="feat-bars">'
    for cls, label, val in bars:
        pct = int(val * 100)
        html += f"""
        <div class="feat-bar-item">
          <div class="feat-bar-track">
            <div class="feat-bar-fill {cls}" style="width:{pct}%"></div>
          </div>
          <span class="feat-bar-label">{label}</span>
        </div>"""
    html += "</div>"
    return html


def _highlight_reason(text: str) -> str:
    """Bold the key signal phrases in the reasoning string."""
    import re
    patterns = [
        r"(real retrieval/ranking/applied-ML substance)",
        r"(NLP/IR background present)",
        r"(product-company experience)",
        r"(recently shipping production code)",
        r"(experience in the JD.s ideal band)",
        r"(experience close to the target band)",
        r"(engaged and available)",
        r"(Gaps:[^.]+\.)",
    ]
    for p in patterns:
        text = re.sub(p, r"<strong>\1</strong>", text, flags=re.I)
    return text


def _rank_class(rank: int) -> str:
    if rank == 1: return "gold"
    if rank == 2: return "silver"
    if rank == 3: return "bronze"
    if rank <= 10: return "top10"
    return ""


def _cards_html(top15: list) -> str:
    cards = ""
    for item in top15:
        rank    = item["rank"]
        name    = _e(item["name"])
        title   = _e(item["title"])
        company = _e(item["company"])
        yoe     = item["yoe"]
        score   = item["score"]
        base    = item["base"]
        bm      = item["bm"]
        reason  = _highlight_reason(_e(item["reasoning"]))
        pct     = int(score * 100)
        rc      = _rank_class(rank)
        feat    = _feat_bars(item)
        bmbadge = _bm_badge(bm)

        cards += f"""
<div class="cand-card">
  <div class="cand-rank {rc}">#{rank}</div>
  <div class="cand-body">
    <div class="cand-name">{name}&nbsp;{bmbadge}</div>
    <div class="cand-meta">{title}&nbsp;·&nbsp;{company}&nbsp;·&nbsp;{yoe:.1f} yrs</div>
    <div class="cand-reason">{reason}</div>
    {feat}
  </div>
  <div class="score-col">
    <div class="score-val">{score:.3f}</div>
    <div class="score-bar-track">
      <div class="score-bar-fill" style="width:{pct}%"></div>
    </div>
    <div class="score-base">base {base:.3f}</div>
  </div>
</div>"""

    return f"""
<div class="section-title">Top 15 Candidates by Fit</div>
<div class="cards-wrap">{cards}</div>
"""


_HONEYPOT_REASON_MAP = {
    "salary_inversion":      "salary min > max — internal data contradiction",
    "fake_skill_profile":    "all skills rated advanced/expert, zero endorsements across any",
    "ghost_candidate":       "100% profile completion, 0 applications sent, not open to work",
    "completeness_mismatch": "profile completeness < 25% yet lists 10+ skills",
}


def _caught_html(honeypots: list, stuffers: list) -> str:
    if not honeypots and not stuffers:
        return (
            "<div style='color:#6B7280;font-size:0.85rem;padding:16px 0'>"
            "No honeypots or stuffers detected in this sample.</div>"
        )

    items = ""
    for hp in honeypots[:14]:
        reasons_str = " &nbsp;/&nbsp; ".join(
            _HONEYPOT_REASON_MAP.get(r, r.replace("_", " "))
            for r in hp["reasons"]
        )
        items += f"""
<div class="caught-card">
  <span class="badge badge-red">🚫 Floored &middot; Honeypot</span>
  <div class="caught-name">{_e(hp['name'])}</div>
  <div class="caught-meta">{_e(hp['title'])} &nbsp;·&nbsp; {_e(hp['company'])}</div>
  <div class="caught-reason">{reasons_str}</div>
</div>"""

    for st in stuffers[:14]:
        items += f"""
<div class="caught-card">
  <span class="badge badge-amber">⚡ Flagged &middot; Keyword Stuffer</span>
  <div class="caught-name">{_e(st['name'])}</div>
  <div class="caught-meta">{_e(st['title'])}</div>
  <div class="caught-reason">{_e(st['reason'])}</div>
</div>"""

    total_caught = len(honeypots) + len(stuffers)
    return f"""
<div class="section-title">
  Caught — {total_caught} Profile{'s' if total_caught != 1 else ''} Rejected
</div>
<div class="caught-grid">{items}</div>
"""


def _footer_html() -> str:
    return """
<div class="demo-footer">
  Built for the Redrob AI Hiring Hackathon &nbsp;·&nbsp;
  Scoring reads career description text, not keyword tags &nbsp;·&nbsp;
  <a href="https://github.com">src/ranker/</a>
</div>
"""


# ── Gradio App ─────────────────────────────────────────────────────────────────

def build_demo():
    hero_html    = _hero_html(_TOTAL)
    stats_html   = _stats_html(_SCORED, _HONEYPOTS, _STUFFERS, _TOTAL)
    weights_html = _weights_html()
    cards_html   = _cards_html(_TOP15)
    caught_html  = _caught_html(_HONEYPOTS, _STUFFERS)
    footer_html  = _footer_html()

    _theme = gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(title="DeepFit AI — Redrob Hackathon") as demo:
        gr.HTML(hero_html)
        gr.HTML(stats_html)
        gr.HTML(weights_html)
        gr.HTML(cards_html)
        gr.HTML(caught_html)
        gr.HTML(footer_html)

    return demo, _theme


if __name__ == "__main__":
    app, _theme = build_demo()
    app.launch(
        server_name="127.0.0.1",
        server_port=None,   # auto-find free port — avoids WinError 10048
        share=False,
        css=CSS,
        theme=_theme,
        inbrowser=True,
    )
