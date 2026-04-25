from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


_SECTION_ORDER = [
    "contact",
    "summary",
    "education",
    "experience",
    "projects",
    "skills",
    "certifications",
    "achievements",
    "publications",
    "links",
    "other",
]

_SECTION_ALIASES: Dict[str, List[str]] = {
    "contact": ["contact", "contacts"],
    "summary": ["summary", "profile", "about", "objective"],
    "education": ["education", "academics", "academic"],
    "experience": ["experience", "work experience", "internships", "internship", "employment"],
    "projects": ["projects", "project", "personal projects"],
    "skills": ["skills", "technical skills", "tech stack", "tools"],
    "certifications": ["certifications", "certification", "courses", "coursework"],
    "achievements": ["achievements", "awards", "accomplishments"],
    "publications": ["publications", "publication"],
    "links": ["links", "profiles"],
}


def _canonical_section(header: str) -> str:
    h = re.sub(r"[^a-z ]+", " ", (header or "").lower()).strip()
    h = re.sub(r"\s+", " ", h)
    for canon, names in _SECTION_ALIASES.items():
        for n in names:
            if h == n or h.startswith(n + " "):
                return canon
    return ""


def split_resume_sections(text: str) -> Dict[str, str]:
    """
    Very lightweight resume section segmentation.
    Works on text extracted from PDFs where headings are usually in caps.
    """
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in raw.split("\n")]
    lines = [ln for ln in lines if ln]

    sections: Dict[str, List[str]] = {"other": []}
    current = "other"

    header_re = re.compile(r"^[A-Z][A-Z &/()\-]{2,40}$")
    for ln in lines:
        # Heading heuristic: all-caps short line, or line ending with ":" that looks like a header.
        is_header = bool(header_re.match(ln)) or (ln.endswith(":") and 2 <= len(ln) <= 40)
        if is_header:
            canon = _canonical_section(ln.rstrip(":"))
            if canon:
                current = canon
                sections.setdefault(current, [])
                continue
        sections.setdefault(current, []).append(ln)

    out: Dict[str, str] = {}
    for key, vals in sections.items():
        joined = "\n".join(vals).strip()
        if joined:
            out[key] = joined
    return out


def extract_project_bullets(sections: Dict[str, str], limit: int = 12) -> List[str]:
    txt = sections.get("projects") or ""
    if not txt:
        return []
    bullets: List[str] = []
    for ln in txt.split("\n"):
        t = ln.strip()
        if not t:
            continue
        if t.startswith(("•", "-", "●", "◦")):
            t = t.lstrip("•-●◦ ").strip()
        # Prefer bullet-like or action lines
        if len(t) >= 16:
            bullets.append(t[:240])
        if len(bullets) >= limit:
            break
    return bullets


def extract_keywords(text: str, limit: int = 24) -> List[str]:
    """
    Extract "tech-looking" keywords not covered by the canonical TECH_SKILLS list.
    Conservative to avoid resume noise (names/universities).
    """
    raw = text or ""
    # Find tokens like "CI/CD", "ETL", "REST", "S3", "EC2", "Medallion", "Lakehouse"
    candidates = re.findall(r"\b[A-Za-z][A-Za-z0-9+/.\-]{1,24}\b", raw)
    stop = {
        "vit", "bhopal", "university", "bachelor", "technology", "computer", "science",
        "engineering", "email", "linkedin", "portfolio", "jan", "feb", "mar", "apr",
        "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "present",
    }
    freq: Dict[str, int] = {}
    for c in candidates:
        low = c.lower()
        if low in stop:
            continue
        if low.isdigit():
            continue
        # Require at least one of: uppercase acronym, contains slash/dot/plus, or known data keywords
        techy = (
            bool(re.fullmatch(r"[A-Z]{2,6}", c))
            or any(ch in c for ch in ["/", ".", "+", "-"])
            or low in {"etl", "elt", "medallion", "lakehouse", "dwh", "cdc", "api", "rest", "grpc", "ci/cd"}
        )
        if not techy:
            continue
        freq[low] = freq.get(low, 0) + 1

    # Sort by frequency then length (shorter acronyms first)
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))
    return [k for k, _ in ranked[:limit]]

