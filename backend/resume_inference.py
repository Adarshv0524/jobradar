from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional


@dataclass
class ResumeExperienceInference:
    level: str  # intern | junior | mid | mid-senior | senior | unknown
    confidence: str  # low | medium | high
    years_estimate: Optional[float]
    internship_months: Optional[int]
    graduation_year: Optional[int]
    signals: List[str]


_IGNORE_SENIOR_PHRASES = [
    "senior secondary",
    "senior school",
    "senior high school",
]


def _find_graduation_year(text: str) -> Optional[int]:
    low = text.lower()
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", low)]
    if not years:
        return None

    # Prefer years that appear close to graduation/education keywords.
    edu_kws = ["graduat", "b.tech", "btech", "b.e", "be ", "bachelor", "degree", "cgpa", "university", "college"]
    best: Optional[int] = None
    best_score = 0
    for y in years:
        if y < 1990 or y > 2100:
            continue
        # Search windows around each year occurrence
        for m in re.finditer(rf"\b{y}\b", low):
            start = max(0, m.start() - 60)
            end = min(len(low), m.end() + 60)
            window = low[start:end]
            score = sum(1 for kw in edu_kws if kw in window)
            if score > best_score:
                best_score = score
                best = y
    return best or max(years)


def _extract_years_experience(text: str) -> Optional[float]:
    low = text.lower()
    # Common patterns: "2 years", "2+ years", "2 yrs", "2 years of experience"
    matches = re.findall(r"\b(\d{1,2})(?:\+)?\s*(?:years?|yrs?)\b(?:\s+of\s+experience)?", low)
    if not matches:
        return None
    vals = []
    for n in matches:
        try:
            vals.append(float(n))
        except Exception:
            pass
    return max(vals) if vals else None


def _extract_internship_months(text: str) -> Optional[int]:
    low = text.lower()
    # "4 months internship" / "internship (4 months)" / "4 mos intern"
    pats = [
        r"\b(\d{1,2})\s*(?:months?|mos?)\b.{0,20}\bintern",
        r"\bintern\b.{0,20}\b(\d{1,2})\s*(?:months?|mos?)\b",
    ]
    vals: List[int] = []
    for pat in pats:
        for m in re.finditer(pat, low):
            try:
                vals.append(int(m.group(1)))
            except Exception:
                pass
    return max(vals) if vals else None


def infer_resume_experience(text: str) -> ResumeExperienceInference:
    """
    Resume-specific experience inference.

    This is intentionally conservative (defaults to "unknown") to avoid
    false "senior" due to phrases like "Senior Secondary".
    """
    raw = text or ""
    low = raw.lower()

    signals: List[str] = []
    for phrase in _IGNORE_SENIOR_PHRASES:
        if phrase in low:
            signals.append(f"ignore_phrase:{phrase}")
            low = low.replace(phrase, "")

    grad_year = _find_graduation_year(low)
    if grad_year:
        signals.append(f"graduation_year:{grad_year}")

    internship_months = _extract_internship_months(low)
    if internship_months:
        signals.append(f"internship_months:{internship_months}")

    years = _extract_years_experience(low)
    if years is not None:
        signals.append(f"years_experience:{years:g}")

    # Explicit seniority words (after removing ignore phrases)
    senior_words = ["staff", "principal", "lead", "architect", "director", "vp", "head of"]
    senior_hits = [w for w in senior_words if re.search(rf"\b{re.escape(w)}\b", low)]
    if senior_hits:
        signals.append("seniority_words:" + ",".join(senior_hits[:4]))

    new_grad_signals = ["expected graduation", "graduating", "pursuing", "student", "final year", "2026 batch", "2025 batch"]
    if any(s in low for s in new_grad_signals):
        signals.append("new_grad_signal")

    intern_signals = ["intern", "internship", "trainee"]
    if any(re.search(rf"\b{re.escape(s)}\b", low) for s in intern_signals):
        signals.append("intern_signal")

    # Decide level
    today_year = date.today().year

    # Strong numeric years signal wins
    if years is not None:
        if years >= 7:
            return ResumeExperienceInference("senior", "high", years, internship_months, grad_year, signals)
        if years >= 5:
            return ResumeExperienceInference("senior", "high", years, internship_months, grad_year, signals)
        if years >= 3:
            return ResumeExperienceInference("mid-senior", "high", years, internship_months, grad_year, signals)
        if years >= 2:
            return ResumeExperienceInference("mid", "high", years, internship_months, grad_year, signals)
        if years >= 1:
            return ResumeExperienceInference("junior", "high", years, internship_months, grad_year, signals)
        return ResumeExperienceInference("intern", "medium", years, internship_months, grad_year, signals)

    # If we only have internship months and graduation looks recent/future → intern/junior
    if internship_months is not None:
        if grad_year and grad_year >= today_year:
            lvl = "intern" if internship_months <= 8 else "junior"
            return ResumeExperienceInference(lvl, "high", None, internship_months, grad_year, signals)
        return ResumeExperienceInference("junior", "medium", None, internship_months, grad_year, signals)

    # Senior words without numeric years are weak on resumes; treat cautiously
    if senior_hits:
        return ResumeExperienceInference("mid", "low", None, None, grad_year, signals)

    # New grad heuristics
    if grad_year and grad_year >= today_year:
        return ResumeExperienceInference("intern", "medium", None, None, grad_year, signals)

    return ResumeExperienceInference("unknown", "low", None, None, grad_year, signals)


def resume_profile_from_text(text: str) -> Dict[str, object]:
    exp = infer_resume_experience(text)
    return {
        "experience": {
            "level": exp.level,
            "confidence": exp.confidence,
            "years_estimate": exp.years_estimate,
            "internship_months": exp.internship_months,
            "graduation_year": exp.graduation_year,
            "signals": exp.signals,
        }
    }

