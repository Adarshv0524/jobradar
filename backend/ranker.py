"""
Multi-signal job scoring, ranking, and feedback-adaptive re-ranking.

KEY FIXES vs original:
  1. Experience level mismatch now applies a hard MULTIPLIER (not just additive).
     A fresher searching for junior roles will see senior/lead jobs score < 0.15
     even if skills match — they will be filtered by MIN_SCORE_THRESHOLD.
  2. Location penalty: "Remote United States" jobs are penalised when user
     specifies a non-US location (e.g. India). US state/city signals in the
     job location field are detected and scored accordingly.
  3. Skill scoring weight increased (0.25→0.30) to reward tech-skill overlap more.
  4. 'lead' and 'staff' and 'principal' are now recognised as exp rank 5,
     above 'senior' (rank 4), so the diff calculation is more accurate.
"""

import math
import re
from datetime import datetime, date
from typing import Dict, List

from config import (
    HIGH_QUALITY_THRESHOLD,
    MIN_SCORE_THRESHOLD,
    US_ONLY_LOCATION_SIGNALS,
    INDIA_LOCATION_SIGNALS,
)

# Optional: sentence-transformers for semantic similarity
_st_model = None
_ST_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _st_model      = SentenceTransformer("all-MiniLM-L6-v2")
    _ST_AVAILABLE  = True
except Exception:
    pass

# TF-IDF fallback
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    _TFIDF_AVAILABLE = True
except Exception:
    _TFIDF_AVAILABLE = False


# ── Similarity ────────────────────────────────────────────────────────────────

def text_similarity(a: str, b: str) -> float:
    """Return cosine similarity [0, 1] between two texts."""
    if not a or not b:
        return 0.0

    if _ST_AVAILABLE and _st_model:
        try:
            emb = _st_model.encode([a[:512], b[:512]])
            num = float(np.dot(emb[0], emb[1]))
            den = float(np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]) + 1e-9)
            return max(0.0, min(1.0, num / den))
        except Exception:
            pass

    if _TFIDF_AVAILABLE:
        try:
            vec = TfidfVectorizer(stop_words="english", max_features=1000)
            tf  = vec.fit_transform([a, b])
            return float(cos_sim(tf[0:1], tf[1:2])[0][0])
        except Exception:
            pass

    # Last-resort: Jaccard on trigrams
    def ngrams(t, n=3):
        t = t.lower()
        return set(t[i:i+n] for i in range(len(t) - n + 1))
    a_g, b_g = ngrams(a), ngrams(b)
    if not a_g or not b_g:
        return 0.0
    return len(a_g & b_g) / len(a_g | b_g)


def _token_overlap_score(a: str, b: str) -> float:
    """
    Fast, high-precision overlap score for role/title matching.
    Helps avoid low TF-IDF scores for short strings like "data engineer".
    """
    if not a or not b:
        return 0.0
    stop = {
        "the", "and", "or", "of", "to", "in", "for", "with", "a", "an",
        "remote", "hybrid", "onsite", "on-site", "full-time", "part-time",
        "engineer",  # keep overlap meaningful; "engineer" is too generic
    }
    ta = [t for t in re.findall(r"[a-z0-9]+", a.lower()) if t and t not in stop]
    tb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not ta:
        return 0.0
    hit = sum(1 for t in ta if t in tb)
    return max(0.0, min(1.0, hit / max(len(ta), 1)))


# ── Skill matching ────────────────────────────────────────────────────────────

def skill_overlap(user_skills: List[str], job_skills: List[str]) -> float:
    if not user_skills or not job_skills:
        return 0.0
    u = {s.lower() for s in user_skills}
    j = {s.lower() for s in job_skills}
    # Partial / substring matching
    matched = sum(1 for js in j if any(us in js or js in us for us in u))
    return min(1.0, matched / max(len(j), 1))


# ── Recency ───────────────────────────────────────────────────────────────────

def recency_score(posted_date: str) -> float:
    """1.0 = today, decays to ~0.2 after 60 days."""
    if not posted_date:
        return 0.5  # unknown — neutral
    try:
        pd = datetime.strptime(posted_date[:10], "%Y-%m-%d").date()
        age_days = (date.today() - pd).days
        return max(0.1, math.exp(-age_days / 30))
    except Exception:
        return 0.5


# ── Experience level ──────────────────────────────────────────────────────────

def _exp_rank(lvl: str) -> int:
    """
    Numeric rank for experience levels.
    FIX: Added 'lead', 'staff', 'principal' at rank 5 (above senior=4).
    This makes junior→lead diff = 4 instead of 3, giving a heavier penalty.
    """
    return {
        "intern":     0,
        "fresher":    0,
        "entry":      0,
        "junior":     1,
        "mid":        2,
        "mid-senior": 3,
        "senior":     4,
        "lead":       5,
        "staff":      5,
        "principal":  5,
        "director":   6,
        "vp":         6,
    }.get(lvl.lower().strip(), 2)


def _exp_multiplier(pref_exp: str, job_exp: str) -> float:
    """
    FIX: Hard multiplier based on exp level diff.
    Original code only did additive ±0.5 which was too weak.
    Now a senior job for a junior searcher scores ×0.25 → near zero.
    """
    if not pref_exp or not job_exp:
        return 1.0
    diff = abs(_exp_rank(pref_exp) - _exp_rank(job_exp))
    return {
        0: 1.00,   # Perfect match
        1: 0.85,   # One level off (junior vs mid = acceptable)
        2: 0.55,   # Two levels (junior vs senior = borderline)
        3: 0.25,   # Three levels (junior vs lead)
        4: 0.10,   # Four levels (intern vs principal)
    }.get(diff, 0.05)   # 5+ levels: near-zero


# ── Location scoring ──────────────────────────────────────────────────────────

def _location_multiplier(pref_loc: str, job: dict) -> float:
    """
    FIX: Detect 'Remote United States only' jobs and penalise them when
    the user wants India (or any other non-US location).

    Logic:
    - If the job location or description contains strong US-only signals
      AND the user wants India → heavy penalty (0.15)
    - If the job is in India or worldwide/remote with no US restriction → bonus (1.1)
    - Otherwise neutral (1.0)
    """
    if not pref_loc:
        return 1.0

    pref_lower    = pref_loc.lower()
    job_loc       = (job.get("location") or "").lower()
    job_desc      = (job.get("description") or "")[:500].lower()
    job_remote    = (job.get("remote_type") or "").lower()

    wants_india = any(s in pref_lower for s in [
        "india", "bangalore", "bengaluru", "hyderabad", "pune", "mumbai",
        "chennai", "delhi", "remote india",
    ])

    if wants_india:
        # Check for hard US-only signals in location
        us_in_loc  = any(s in job_loc  for s in US_ONLY_LOCATION_SIGNALS)
        us_in_desc = any(s in job_desc for s in [
            "authorized to work in the us", "must be in the us",
            "remote - united states", "remote us only", "us-based only",
            "must reside in the united states", "work authorization in the us",
        ])
        if us_in_loc or us_in_desc:
            return 0.15  # Hard penalty — Indian candidates can't apply

        # Reward India-specific jobs
        india_in_loc = any(s in job_loc for s in INDIA_LOCATION_SIGNALS)
        if india_in_loc:
            return 1.15  # Bonus for India-specific roles

        # Worldwide / no restriction remote — acceptable
        if job_remote == "remote" and not us_in_loc:
            return 0.90  # Slight discount (may still be US-only)

        return 1.0

    # For non-India preferences, do a simple containment check
    pref_words = [w for w in pref_lower.split() if len(w) > 2]
    if pref_words and job_loc:
        if any(w in job_loc for w in pref_words):
            return 1.10
        if job_remote == "remote":
            return 0.85
        return 0.70   # Wrong location, not remote

    return 1.0


# ── Preference matching ───────────────────────────────────────────────────────

def preference_score(job: dict, prefs: dict) -> float:
    """
    Additive preference score [0,1] based on remote type and salary.
    NOTE: Experience level and location are now handled as multipliers
    in score_job() rather than here, so they have stronger effect.
    """
    score  = 0.0
    weight = 0

    # Remote preference
    pref_remote = prefs.get("remote_preference", "")
    job_remote  = (job.get("remote_type") or "").lower()
    if pref_remote and pref_remote != "any":
        weight += 1
        if pref_remote == "remote"   and job_remote == "remote":   score += 1.0
        elif pref_remote == "hybrid" and job_remote in ("remote", "hybrid"): score += 0.8
        elif pref_remote == "on-site" and job_remote == "on-site": score += 1.0
        elif job_remote == "remote":                                score += 0.4

    # Salary
    pref_sal  = prefs.get("min_salary", 0)
    job_sal   = _extract_salary_number(job.get("salary", ""))
    if pref_sal and job_sal:
        weight += 1
        score  += 1.0 if job_sal >= pref_sal else max(0.0, job_sal / pref_sal)

    return score / weight if weight else 0.6  # neutral if no prefs set


def _extract_salary_number(salary_str: str) -> int:
    nums = re.findall(r"[\d,]+", (salary_str or "").replace("k", "000").replace("K", "000"))
    if nums:
        try:
            return int(nums[0].replace(",", ""))
        except Exception:
            pass
    return 0


# ── Negative filtering ────────────────────────────────────────────────────────

def check_negatives(job: dict, negatives: List[str]) -> List[str]:
    """Return list of triggered negative reasons."""
    text  = ((job.get("title") or "") + " " + (job.get("description") or "") +
             " " + (job.get("company") or "")).lower()
    flags = []
    neg_map = {
        "no_agency":   ["staffing agency", "recruiting agency", "we are a staffing"],
        "no_crypto":   ["crypto", "blockchain", "web3", "nft", "defi", "token"],
        "no_sales":    ["sales development", "account executive", "cold calling", "quota"],
        "no_unpaid":   ["unpaid", "volunteer", "no compensation", "for college credit"],
        "no_relocation": ["must relocate", "relocation required"],
    }
    for key, patterns in neg_map.items():
        if key in negatives:
            for p in patterns:
                if p in text:
                    flags.append(f"Matches negative filter: {key}")
                    break
    return flags


# ── Title quality check ───────────────────────────────────────────────────────

def _is_garbage_title(title: str) -> bool:
    """
    FIX: Detect HN opinion posts and aggregator noise that pass is_fake_job().
    These have no real company or apply_url and look like opinion sentences.
    """
    if not title:
        return True
    low = title.lower()
    # HN opinion comments often start with these patterns
    garbage_signals = [
        "there is no issue", "there is always", "lol @", "i've worked",
        "i am not saying", "hey -", "ooh, let me", "as an aside",
        "the issue is more", "frankly the issue",
    ]
    if any(s in low for s in garbage_signals):
        return True
    # Titles that are clearly sentences (contain a period mid-text or are very long)
    if len(title) > 120:
        return True
    # Contains HTML entities
    if "&amp;" in title or "&#x" in title or "href=" in title:
        return True
    return False


# ── Main scoring ──────────────────────────────────────────────────────────────

def score_job(job: dict, profile: dict, prefs: dict,
              feedback_profile: dict = None) -> dict:
    """
    Returns updated job dict with 'score', 'score_breakdown',
    'match_reasons', 'reject_reasons'.

    Score formula (before multipliers):
      0.25 * semantic   — profile ↔ job description similarity
      0.30 * skill_sc   — skill overlap (INCREASED from 0.25)
      0.20 * title_sc   — role title match
      0.15 * pref_sc    — remote / salary preferences
      0.10 * rec_sc     — recency
      + src_bonus       — ATS/structured source quality

    Post-formula multipliers (applied last, can dramatically lower score):
      × exp_mult        — experience level mismatch
      × loc_mult        — US-only job for India searcher
    """
    title       = job.get("title", "")
    desc        = job.get("description", "")
    job_skills  = job.get("skills", [])

    # ── Garbage filter ────────────────────────────────────────────────────────
    if _is_garbage_title(title):
        return {
            **job,
            "score": 0.0,
            "score_breakdown": {},
            "match_reasons": [],
            "reject_reasons": ["Garbage/spam title detected"],
        }

    profile_text    = _build_profile_text(profile)
    user_skills     = profile.get("skills", [])

    # 1. Semantic similarity between profile and job
    semantic = text_similarity(profile_text, title + " " + desc)

    # 2. Skill overlap (weight: 0.30, was 0.25)
    skill_sc = skill_overlap(user_skills, job_skills)

    # 3. Title match
    pref_role  = prefs.get("role", "")
    title_sc   = text_similarity(pref_role, title) if pref_role else semantic * 0.8
    if pref_role:
        title_sc = max(title_sc, _token_overlap_score(pref_role, title))
        # If the role tokens match strongly, treat as near-perfect title alignment.
        if title_sc >= 0.95:
            title_sc = 1.0

    # 4. Preference match (remote + salary only)
    pref_sc = preference_score(job, prefs)

    # 5. Recency
    rec_sc = recency_score(job.get("posted_date", ""))

    # 6. Source quality bonus
    source_type = job.get("source_type", "")
    src_bonus   = {"jsonld": 0.05, "detail_page": 0.03, "heuristic": 0.0}.get(source_type, 0.0)

    # 7. Feedback adjustment
    fb_adj = 0.0
    if feedback_profile:
        pos_skills = feedback_profile.get("positive_skills", {})
        neg_skills = feedback_profile.get("negative_skills", {})
        company    = (job.get("company") or "").lower()

        for s in job_skills:
            fb_adj += pos_skills.get(s, 0) * 0.04
            fb_adj -= neg_skills.get(s, 0) * 0.04

        if company in feedback_profile.get("positive_companies", set()):
            fb_adj += 0.10
        if company in feedback_profile.get("negative_companies", set()):
            fb_adj -= 0.15

    # ── Weighted composite (pre-multiplier) ───────────────────────────────────
    # Dynamic weighting:
    # When the user hasn't uploaded a resume / skills, the system should still
    # produce high-confidence matches based on role-title + preferences.
    w_sem   = 0.25 if len(profile_text) >= 60 else 0.10
    w_skill = 0.30 if user_skills and job_skills else 0.0
    w_title = 0.30 if pref_role else 0.20
    w_pref  = 0.20
    w_rec   = 0.10

    total_w = w_sem + w_skill + w_title + w_pref + w_rec
    if total_w <= 0:
        total_w = 1.0

    raw_score = (
        (w_sem / total_w) * semantic
        + (w_skill / total_w) * skill_sc
        + (w_title / total_w) * title_sc
        + (w_pref / total_w) * pref_sc
        + (w_rec / total_w) * rec_sc
        + src_bonus
        + fb_adj
    )
    raw_score = max(0.0, min(1.0, raw_score))

    # ── Apply hard multipliers ─────────────────────────────────────────────────
    pref_exp  = (prefs.get("experience_level") or "").lower()
    job_exp   = (job.get("experience_level")   or "").lower()
    exp_mult  = _exp_multiplier(pref_exp, job_exp)

    pref_loc  = (prefs.get("location") or "").lower()
    loc_mult  = _location_multiplier(pref_loc, job)

    score = raw_score * exp_mult * loc_mult
    score = max(0.0, min(1.0, score))

    # ── Negative filters ──────────────────────────────────────────────────────
    negatives = prefs.get("negatives", [])
    neg_flags = check_negatives(job, negatives)

    # ── Build match/reject reasons ────────────────────────────────────────────
    match_reasons, reject_reasons = [], []

    if neg_flags:
        reject_reasons.extend(neg_flags)
        score = 0.0

    if exp_mult < 0.30:
        reject_reasons.append(
            f"Experience mismatch: looking for {pref_exp}, job is {job_exp or 'unknown'}"
        )

    if loc_mult < 0.30:
        reject_reasons.append("US-only remote — likely inaccessible for your location")

    if skill_sc > 0.4:
        matched = [s for s in user_skills if any(s.lower() in js.lower() for js in job_skills)]
        match_reasons.append(f"Skill match: {', '.join(matched[:5])}")
    if semantic > 0.5:
        match_reasons.append(f"Strong profile alignment ({semantic:.0%})")
    if pref_sc > 0.7:
        match_reasons.append("Matches your preferences")
    if rec_sc > 0.8:
        match_reasons.append("Recently posted")
    if loc_mult > 1.0:
        match_reasons.append("India-based or India-accessible role")
    if score < MIN_SCORE_THRESHOLD and not reject_reasons:
        reject_reasons.append("Low relevance to profile")

    breakdown = {
        "semantic": round(semantic, 3),
        "skill":    round(skill_sc, 3),
        "title":    round(title_sc, 3),
        "pref":     round(pref_sc, 3),
        "recency":  round(rec_sc, 3),
        "feedback": round(fb_adj, 3),
        "exp_mult": round(exp_mult, 3),
        "loc_mult": round(loc_mult, 3),
    }

    return {
        **job,
        "score":           round(score, 4),
        "score_breakdown": breakdown,
        "match_reasons":   match_reasons,
        "reject_reasons":  reject_reasons,
    }


def rank_jobs(jobs: List[dict], profile: dict, prefs: dict,
              feedback_profile: dict = None) -> List[dict]:
    scored = [score_job(j, profile, prefs, feedback_profile) for j in jobs]
    return sorted(scored, key=lambda j: j["score"], reverse=True)


def _build_profile_text(profile: dict) -> str:
    parts = [
        profile.get("summary", ""),
        " ".join(profile.get("skills", [])),
        profile.get("experience_summary", ""),
        profile.get("role_target", ""),
    ]
    return " ".join(p for p in parts if p)[:2000]
