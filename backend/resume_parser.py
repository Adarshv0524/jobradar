from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ResumeParseResult:
    text: str
    method: str
    warnings: List[str]
    page_char_counts: List[int]


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    # Normalize common PDF extraction artifacts without being destructive.
    t = text.replace("\x00", "")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # Remove excessive blank lines
    lines = [ln.rstrip() for ln in t.split("\n")]
    out: List[str] = []
    blank_run = 0
    for ln in lines:
        if not ln.strip():
            blank_run += 1
            if blank_run <= 2:
                out.append("")
            continue
        blank_run = 0
        out.append(ln)
    return "\n".join(out).strip()


def parse_resume_bytes(
    content: bytes,
    content_type: str,
    filename: str = "",
) -> ResumeParseResult:
    """
    Best-effort resume text extraction with deterministic fallbacks.

    - PDF: try pdfplumber (layout + simple) then try pypdf, then last-resort bytes decode.
    - Text: decode as utf-8 with ignore.
    """
    warnings: List[str] = []
    page_char_counts: List[int] = []

    if content_type != "application/pdf":
        text = _normalize_text(content.decode("utf-8", errors="ignore"))
        return ResumeParseResult(text=text, method="utf8", warnings=warnings, page_char_counts=[])

    # 1) pdfplumber (primary)
    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(io.BytesIO(content)) as pdf:
            parts: List[str] = []
            for page in pdf.pages:
                # Try layout-aware extraction first (better for multi-column resumes)
                page_text = ""
                try:
                    page_text = page.extract_text(layout=True) or ""
                except Exception:
                    page_text = ""
                if not page_text:
                    try:
                        page_text = page.extract_text() or ""
                    except Exception:
                        page_text = ""
                page_char_counts.append(len(page_text))
                parts.append(page_text)
            text = _normalize_text("\n".join(parts))
            if text and len(text) >= 200:
                return ResumeParseResult(
                    text=text,
                    method="pdfplumber",
                    warnings=warnings,
                    page_char_counts=page_char_counts,
                )
            warnings.append("pdfplumber_extracted_too_little_text")
    except Exception as e:
        warnings.append(f"pdfplumber_failed:{type(e).__name__}")

    # 2) pypdf (fallback)
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(io.BytesIO(content))
        parts = []
        page_char_counts = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            page_char_counts.append(len(t))
            parts.append(t)
        text = _normalize_text("\n".join(parts))
        if text and len(text) >= 200:
            return ResumeParseResult(
                text=text,
                method="pypdf",
                warnings=warnings,
                page_char_counts=page_char_counts,
            )
        warnings.append("pypdf_extracted_too_little_text")
    except Exception as e:
        warnings.append(f"pypdf_failed:{type(e).__name__}")

    # 3) last resort: decode bytes (won't be good, but deterministic)
    warnings.append("fallback_to_utf8_decode")
    text = _normalize_text(content.decode("utf-8", errors="ignore"))
    return ResumeParseResult(text=text, method="utf8_fallback", warnings=warnings, page_char_counts=page_char_counts)

