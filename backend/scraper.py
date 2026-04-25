"""Web fetching, HTML parsing, and structured job extraction."""

import asyncio
import json
import random
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

import logging
log = logging.getLogger(__name__)

from config import (
    ATS_DOMAINS, CRAWL_DELAY, FAKE_SIGNALS, MAX_RETRIES,
    REQUEST_TIMEOUT, TECH_SKILLS, USER_AGENTS,
)

_last_req: Dict[str, float] = {}  # domain → last request timestamp


# ── HTTP ──────────────────────────────────────────────────────────────────────

async def fetch_html(url: str, use_playwright_fallback: bool = False) -> Optional[str]:
    """Fetch URL respecting per-domain rate limits. Auto-retries with backoff."""
    domain = urlparse(url).netloc

    loop = asyncio.get_event_loop()
    now  = loop.time()
    wait = CRAWL_DELAY - (now - _last_req.get(domain, 0))
    if wait > 0:
        await asyncio.sleep(wait)
    _last_req[domain] = loop.time()

    # Rotate UA per domain group
    ua = random.choice(USER_AGENTS)
    headers = {
        "User-Agent":      ua,
        "Accept":          "text/html,application/xhtml+xml,*/*;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT":             "1",
        "Connection":      "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT,
                follow_redirects=True,
                headers=headers,
                http2=True,
            ) as client:
                r = await client.get(url)
                if r.status_code == 200:
                    text = r.text
                    # If page is suspiciously short or has CAPTCHA signals, try Playwright
                    if use_playwright_fallback and (
                        len(text) < 3000
                        or "captcha" in text.lower()
                        or "cf-challenge" in text.lower()
                        or "please verify" in text.lower()
                    ):
                        from playwright_pool import fetch_with_playwright
                        pw_html = await fetch_with_playwright(url)
                        if pw_html and len(pw_html) > len(text):
                            return pw_html
                    return text
                if r.status_code == 403:
                    # Try Playwright on 403 (Cloudflare, etc.)
                    if use_playwright_fallback:
                        from playwright_pool import fetch_with_playwright
                        return await fetch_with_playwright(url)
                    return None
                if r.status_code in (429, 503):
                    wait_time = 4 * (attempt + 1) + random.uniform(0, 2)
                    await asyncio.sleep(wait_time)
                elif r.status_code in (404, 410):
                    return None
        except (httpx.TimeoutException, httpx.ConnectError):
            await asyncio.sleep(2 * (attempt + 1))
        except Exception as e:
            log.debug("fetch_html error %s: %s", url, e)
            await asyncio.sleep(1.5 * (attempt + 1))

    # Last resort: try Playwright
    if use_playwright_fallback:
        from playwright_pool import fetch_with_playwright
        return await fetch_with_playwright(url)
    return None


# ── DuckDuckGo search ─────────────────────────────────────────────────────────

async def search_web(query: str, max_results: int = 10) -> List[Dict]:
    """Return list of {url, title, snippet} dicts."""
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results, timelimit="m"):
                results.append({
                    "url":     r.get("href", ""),
                    "title":   r.get("title", ""),
                    "snippet": r.get("body", ""),
                })
        return [r for r in results if r["url"]]
    except Exception:
        return await _ddg_html_fallback(query, max_results)


async def _ddg_html_fallback(query: str, max_results: int) -> List[Dict]:
    url  = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    html = await fetch_html(url)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    out  = []
    for el in soup.select(".result")[:max_results]:
        a   = el.select_one(".result__a")
        snip = el.select_one(".result__snippet")
        if a and a.get("href"):
            out.append({
                "url":     a["href"],
                "title":   a.get_text(strip=True),
                "snippet": snip.get_text(strip=True) if snip else "",
            })
    return out


# ── Page classification ───────────────────────────────────────────────────────

def classify_page(html: str, url: str) -> str:
    """Return one of: job_listing | job_detail | careers_home | irrelevant."""
    if not html:
        return "irrelevant"

    url_lower  = url.lower()
    html_lower = html.lower()

    # ATS pages are always job-intent
    if any(a in url_lower for a in ATS_DOMAINS):
        return "job_detail" if re.search(r"/(job|position|role)/\w", url_lower) else "job_listing"

    # JSON-LD count is the best signal
    jsonld_count = len(re.findall(r'"@type"\s*:\s*"JobPosting"', html))
    if jsonld_count > 1:
        return "job_listing"
    if jsonld_count == 1:
        return "job_detail"

    # URL path patterns
    if re.search(r"/(jobs?|careers?|positions?|openings?|vacancies?)/?$", url_lower):
        return "careers_home"
    if re.search(r"/(jobs?|careers?|positions?)/[\w-]+", url_lower):
        return "job_detail"

    # Content keyword heuristic
    detail_kws  = ["responsibilities", "qualifications", "requirements", "apply now", "job description"]
    listing_kws = ["view all jobs", "open positions", "join our team", "current openings"]

    detail_score  = sum(1 for k in detail_kws  if k in html_lower)
    listing_score = sum(1 for k in listing_kws if k in html_lower)

    if detail_score >= 2:
        return "job_detail"
    if listing_score >= 2:
        return "job_listing"

    return "irrelevant"


# ── JSON-LD extraction ────────────────────────────────────────────────────────

def extract_jsonld_jobs(html: str, source_url: str) -> List[Dict]:
    jobs = []
    soup = BeautifulSoup(html, "html.parser")
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            raw  = script.string or script.get_text()
            data = json.loads(raw)
            items = data if isinstance(data, list) else [data]
            for item in items:
                if item.get("@type") == "JobPosting":
                    j = _parse_jsonld(item, source_url)
                    if j.get("title"):
                        jobs.append(j)
                elif item.get("@type") == "ItemList":
                    for el in item.get("itemListElement", []):
                        if isinstance(el, dict) and el.get("@type") == "JobPosting":
                            j = _parse_jsonld(el, source_url)
                            if j.get("title"):
                                jobs.append(j)
        except Exception:
            continue
    return jobs


def _parse_jsonld(d: dict, source_url: str) -> dict:
    org      = d.get("hiringOrganization", {})
    company  = org.get("name", "") if isinstance(org, dict) else str(org)

    loc_raw  = d.get("jobLocation", {})
    if isinstance(loc_raw, list):
        loc_raw = loc_raw[0] if loc_raw else {}
    addr     = loc_raw.get("address", {}) if isinstance(loc_raw, dict) else {}
    if isinstance(addr, str):
        location = addr
    else:
        location = ", ".join(filter(None, [
            addr.get("addressLocality", ""), addr.get("addressCountry", "")
        ]))

    desc_raw  = d.get("description", "")
    if isinstance(desc_raw, str) and "<" in desc_raw:
        desc_raw = BeautifulSoup(desc_raw, "html.parser").get_text(" ", strip=True)
    description = (desc_raw or "")[:3500]

    remote_type = "remote" if d.get("jobLocationType") == "TELECOMMUTE" else ""
    if not remote_type:
        combo = location.lower() + description.lower()
        remote_type = "remote" if "remote" in combo else "hybrid" if "hybrid" in combo else "on-site"

    salary = _parse_salary_schema(d.get("baseSalary", {}))

    return {
        "title":            d.get("title", "").strip(),
        "company":          company.strip(),
        "location":         location,
        "remote_type":      remote_type,
        "experience_level": infer_experience(d.get("title", "") + " " + description),
        "salary":           salary,
        "description":      description,
        "skills":           extract_skills(description),
        "apply_url":        d.get("url") or d.get("sameAs") or source_url,
        "posted_date":      d.get("datePosted", "")[:10],
        "source_domain":    urlparse(source_url).netloc,
        "source_type":      "jsonld",
    }


def _parse_salary_schema(s: dict) -> str:
    if not isinstance(s, dict):
        return ""
    val = s.get("value", {})
    if isinstance(val, dict):
        lo   = val.get("minValue", "")
        hi   = val.get("maxValue", "")
        unit = val.get("unitText", "")
        cur  = s.get("currency", "USD")
        if lo and hi:
            return f"{cur} {lo}–{hi} {unit}".strip()
        if lo:
            return f"{cur} {lo}+ {unit}".strip()
    return ""


# ── Heuristic listing extraction ──────────────────────────────────────────────

def extract_heuristic_jobs(html: str, source_url: str) -> List[Dict]:
    soup = BeautifulSoup(html[:200_000], "html.parser")
    containers = soup.find_all(
        ["article", "li", "div"],
        class_=re.compile(r"job|position|listing|opening|role|vacancy", re.I),
    )

    jobs, seen = [], set()
    for c in containers[:60]:
        j = _job_from_container(c, source_url)
        if not j:
            continue
        key = (j["title"].lower()[:60], j.get("company", "").lower()[:40])
        if key not in seen:
            seen.add(key)
            jobs.append(j)
    return jobs


def _job_from_container(c, source_url: str) -> Optional[dict]:
    try:
        # Title
        title_el = (
            c.find(["h1", "h2", "h3", "h4"],
                   class_=re.compile(r"title|name|position|role", re.I))
            or c.find("a")
        )
        title = title_el.get_text(strip=True) if title_el else ""
        if not title or len(title) > 160:
            return None

        company = _text(c, re.compile(r"company|employer|org", re.I))
        location = _text(c, re.compile(r"location|place|city|region", re.I))
        link = c.find("a", href=True)
        url  = urljoin(source_url, link["href"]) if link else source_url

        desc_el  = c.find(class_=re.compile(r"desc|summary|excerpt|snippet", re.I))
        desc     = (desc_el or c).get_text(" ", strip=True)[:600]
        combo    = title + " " + desc

        return {
            "title":            title[:200],
            "company":          (company or "")[:200],
            "location":         (location or "")[:200],
            "remote_type":      "remote" if "remote" in combo.lower() else "hybrid" if "hybrid" in combo.lower() else "on-site",
            "experience_level": infer_experience(combo),
            "salary":           extract_salary_text(combo),
            "description":      desc,
            "skills":           extract_skills(desc),
            "apply_url":        url,
            "posted_date":      "",
            "source_domain":    urlparse(source_url).netloc,
            "source_type":      "heuristic",
        }
    except Exception:
        return None


def _text(el, cls_pattern) -> str:
    found = el.find(class_=cls_pattern)
    return found.get_text(strip=True) if found else ""


# ── Detail page extraction ────────────────────────────────────────────────────

def extract_detail_job(html: str, url: str) -> Optional[dict]:
    soup = BeautifulSoup(html[:300_000], "html.parser")

    title = ""
    for sel in ["h1.job-title", "h1[data-testid*='title']", "h1.title", "h1"]:
        el = soup.select_one(sel)
        if el:
            t = el.get_text(strip=True)
            if 3 < len(t) < 200:
                title = t
                break
    if not title:
        return None

    company = ""
    for sel in [".company-name", "[data-testid*='company']", ".employer", ".org-name"]:
        el = soup.select_one(sel)
        if el:
            company = el.get_text(strip=True)[:200]
            break

    location = ""
    for sel in [".location", "[data-testid*='location']", ".job-location"]:
        el = soup.select_one(sel)
        if el:
            location = el.get_text(strip=True)[:200]
            break

    desc = ""
    for sel in [".job-description", ".description", "#job-description", "main", "article"]:
        el = soup.select_one(sel)
        if el:
            desc = el.get_text(" ", strip=True)[:4000]
            break
    if not desc:
        desc = soup.get_text(" ", strip=True)[:3000]

    combo = title + " " + location + " " + desc
    return {
        "title":            title,
        "company":          company,
        "location":         location,
        "remote_type":      "remote" if "remote" in combo.lower() else "hybrid" if "hybrid" in combo.lower() else "on-site",
        "experience_level": infer_experience(combo),
        "salary":           extract_salary_text(combo),
        "description":      desc,
        "skills":           extract_skills(desc),
        "apply_url":        url,
        "posted_date":      _extract_date(soup),
        "source_domain":    urlparse(url).netloc,
        "source_type":      "detail_page",
    }


# ── Link discovery ────────────────────────────────────────────────────────────

def find_job_links(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html[:200_000], "html.parser")
    seen, out = set(), []
    for a in soup.find_all("a", href=True):
        full  = urljoin(base_url, a["href"].strip())
        clean = full.split("?")[0].rstrip("/")
        if clean in seen:
            continue
        lower = clean.lower()
        if any(p in lower for p in [
            "/job/", "/jobs/", "/career", "/position", "/opening",
            "/vacancy", "/work-with", "/join-us", "/apply",
        ]):
            seen.add(clean)
            out.append(full)
    return out[:30]


async def fetch_paginated_jobs(
    base_url: str,
    query: str,
    max_pages: int = 20,
    use_js: bool = False,
) -> List[Dict]:
    """
    Fetch a paginated job listing site, following pages until empty or max_pages.
    Handles both {page} template URLs and next-link crawling.
    Returns merged list of all raw jobs found.
    """
    all_jobs: List[Dict] = []
    seen_urls: set = set()

    if "{page}" in base_url:
        # Template URL mode — increment page number
        for page_num in range(1, max_pages + 1):
            url = base_url.replace("{page}", str(page_num)).replace(
                "{query}", quote_plus(query)
            )
            if url in seen_urls:
                break
            seen_urls.add(url)

            html = None
            if use_js:
                from playwright_pool import fetch_with_playwright
                html = await fetch_with_playwright(url)
            if not html:
                html = await fetch_html(url)
            if not html:
                break

            page_type = classify_page(html, url)
            jobs, _ = await _extract_from_html(html, url, page_type)
            if not jobs:
                break  # Empty page = we've gone far enough
            all_jobs.extend(jobs)
            await asyncio.sleep(0.5)
    else:
        # Single URL + follow "next page" links
        url = base_url.replace("{query}", quote_plus(query))
        pages_done = 0
        while url and pages_done < max_pages:
            if url in seen_urls:
                break
            seen_urls.add(url)
            pages_done += 1

            html = None
            if use_js:
                from playwright_pool import fetch_with_playwright
                html = await fetch_with_playwright(url)
            if not html:
                html = await fetch_html(url)
            if not html:
                break

            page_type = classify_page(html, url)
            jobs, _ = await _extract_from_html(html, url, page_type)
            all_jobs.extend(jobs)

            # Find next page link
            url = _find_next_page(html, url)
            await asyncio.sleep(0.5)

    return all_jobs


def _find_next_page(html: str, current_url: str) -> Optional[str]:
    """Find the 'next page' link in paginated results."""
    soup = BeautifulSoup(html[:100_000], "html.parser")
    # Common next-page patterns
    for sel in [
        "a[rel='next']",
        "a.next", "a.pagination-next", "a[aria-label='Next']",
        "a[aria-label='next']", ".next-page a", "li.next a",
        "[data-testid='pagination-next'] a",
    ]:
        el = soup.select_one(sel)
        if el and el.get("href"):
            href = el["href"]
            if href.startswith("http"):
                return href
            return urljoin(current_url, href)
    return None


async def _extract_from_html(html: str, url: str, page_type: str):
    """Shared extraction logic for any HTML page."""
    jobs = []
    from scraper import extract_jsonld_jobs, extract_heuristic_jobs, extract_detail_job
    if page_type in ("job_listing", "careers_home"):
        jobs.extend(extract_jsonld_jobs(html, url))
        if not jobs:
            jobs.extend(extract_heuristic_jobs(html, url))
    elif page_type == "job_detail":
        j = extract_detail_job(html, url)
        if j:
            jobs.append(j)
    return jobs, page_type


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_skills(text: str) -> List[str]:
    if not text:
        return []
    low = text.lower()
    return [s for s in TECH_SKILLS if re.search(r"\b" + re.escape(s) + r"\b", low)][:20]


def infer_experience(text: str) -> str:
    low = text.lower()
    for pat, lvl in [
        (r"\b(10|\d{2})\+?\s+years?\b", "senior"),
        (r"\b[7-9]\+?\s+years?\b",      "senior"),
        (r"\b[5-6]\+?\s+years?\b",      "mid-senior"),
        (r"\b[3-4]\+?\s+years?\b",      "mid"),
        (r"\b[1-2]\+?\s+years?\b",      "junior"),
    ]:
        if re.search(pat, low):
            return lvl
    kw_map = {
        "senior": ["senior", "sr.", "lead", "staff", "principal", "architect"],
        "mid":    ["mid-level", "intermediate"],
        "junior": ["junior", "jr.", "entry level", "entry-level", "new grad", "graduate"],
        "intern": ["intern", "internship", "trainee"],
    }
    for lvl, kws in kw_map.items():
        if any(k in low for k in kws):
            return lvl
    return "mid"


def extract_salary_text(text: str) -> str:
    for pat in [
        r"\$[\d,]+\s*[-–]\s*\$[\d,]+(?:\s*[kK])?(?:\s*(?:per year|/yr|annually))?",
        r"\$[\d,]+[kK]\s*[-–]\s*\$?[\d,]+[kK]",
        r"[\d,]+\s*[-–]\s*[\d,]+\s*(?:USD|EUR|GBP)(?:\s*per year)?",
    ]:
        m = re.search(pat, text, re.I)
        if m:
            return m.group(0)[:100]
    return ""


def is_fake_job(title: str, description: str) -> Tuple[bool, List[str]]:
    text  = (title + " " + description).lower()
    flags = [f"Spam signal: '{s}'" for s in FAKE_SIGNALS if s in text]
    if "entry" in text and re.search(r"\b[5-9]\+?\s*years?\b", text):
        flags.append("Fake entry-level: requires 5+ years")
    if len(description) < 80:
        flags.append("Description too short")
    return bool(flags), flags


def _extract_date(soup: BeautifulSoup) -> str:
    for el in soup.find_all(attrs={"datetime": True}):
        dt = el.get("datetime", "")
        if re.match(r"\d{4}-\d{2}-\d{2}", dt):
            return dt[:10]
    for pat in [r"posted[:\s]+(\w+ \d+,?\s*\d{4})", r"(\d{4}-\d{2}-\d{2})"]:
        m = re.search(pat, soup.get_text(), re.I)
        if m:
            return m.group(1)
    return ""


# ── Main entry point ──────────────────────────────────────────────────────────

async def extract_jobs_from_url(url: str) -> Tuple[List[Dict], str]:
    """Fetch URL and extract all jobs from it. Returns (jobs, page_type)."""
    html = await fetch_html(url)
    if not html:
        return [], "failed"

    page_type = classify_page(html, url)
    if page_type == "irrelevant":
        return [], page_type

    # JSON-LD is most reliable — always try first
    jobs = extract_jsonld_jobs(html, url)

    if not jobs:
        if page_type == "job_listing":
            jobs = extract_heuristic_jobs(html, url)
        elif page_type == "job_detail":
            j = extract_detail_job(html, url)
            if j:
                jobs = [j]

    return jobs, page_type


# ── Free API fetchers ─────────────────────────────────────────────────────────
# All return List[dict] in the standard job dict format used by score_job().


async def _api_get(url: str, headers: Optional[dict] = None) -> Optional[dict]:
    """Fetch JSON from a URL."""
    hdrs = {"User-Agent": random.choice(USER_AGENTS), "Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    try:
        async with httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT, follow_redirects=True, headers=hdrs
        ) as client:
            r = await client.get(url)
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return None


def _norm(job: dict) -> dict:
    """Ensure all standard fields exist."""
    return {
        "title":            job.get("title", "").strip()[:200],
        "company":          job.get("company", "").strip()[:200],
        "location":         job.get("location", "").strip()[:200],
        "remote_type":      job.get("remote_type", "remote"),
        "experience_level": job.get("experience_level", "mid"),
        "salary":           job.get("salary", ""),
        "description":      job.get("description", "")[:3500],
        "skills":           job.get("skills", []),
        "apply_url":        job.get("apply_url", ""),
        "posted_date":      job.get("posted_date", ""),
        "source_domain":    job.get("source_domain", ""),
        "source_type":      job.get("source_type", "api"),
    }


async def fetch_remotive_jobs(query: str) -> List[Dict]:
    """
    Remotive.com free API — remote jobs, no key needed.
    https://remotive.com/api/remote-jobs
    """
    url  = f"https://remotive.com/api/remote-jobs?search={quote_plus(query)}&limit=100"
    data = await _api_get(url)
    if not data or "jobs" not in data:
        return []

    out = []
    for j in data["jobs"]:
        desc = BeautifulSoup(j.get("description", ""), "html.parser").get_text(" ", strip=True)[:3000]
        out.append(_norm({
            "title":            j.get("title", ""),
            "company":          j.get("company_name", ""),
            "location":         j.get("candidate_required_location", "Worldwide"),
            "remote_type":      "remote",
            "salary":           j.get("salary", ""),
            "description":      desc,
            "skills":           extract_skills(desc),
            "apply_url":        j.get("url", ""),
            "posted_date":      (j.get("publication_date", "") or "")[:10],
            "source_domain":    "remotive.com",
            "source_type":      "api_remotive",
            "experience_level": infer_experience(j.get("title", "") + " " + desc),
        }))
    return [j for j in out if j["title"] and j["apply_url"]]


async def fetch_arbeitnow_jobs(query: str) -> List[Dict]:
    """
    Arbeitnow.com free API — European + global jobs.
    https://www.arbeitnow.com/api/job-board-api
    """
    out = []
    for page in range(1, 4):   # pages 1-3
        data = await _api_get(f"https://arbeitnow.com/api/job-board-api?page={page}")
        if not data or "data" not in data:
            break
        for j in data["data"]:
            desc = j.get("description", "")[:3000]
            title = j.get("title", "")
            # Filter by relevance — check if query words appear
            q_words = query.lower().split()
            if not any(w in (title + desc).lower() for w in q_words):
                continue
            out.append(_norm({
                "title":            title,
                "company":          j.get("company_name", ""),
                "location":         j.get("location", ""),
                "remote_type":      "remote" if j.get("remote") else "on-site",
                "description":      desc,
                "skills":           extract_skills(desc),
                "apply_url":        j.get("url", ""),
                "posted_date":      (j.get("created_at", "") or "")[:10],
                "source_domain":    "arbeitnow.com",
                "source_type":      "api_arbeitnow",
                "experience_level": infer_experience(title + " " + desc),
            }))
    return [j for j in out if j["title"] and j["apply_url"]]


async def fetch_jobicy_jobs(query: str) -> List[Dict]:
    """
    Jobicy.com free API — remote tech jobs.
    https://jobicy.com/jobs-rss-feed
    """
    tag  = quote_plus(query.split()[0])   # first keyword as tag
    url  = f"https://jobicy.com/api/v2/remote-jobs?count=50&geo=worldwide&tag={tag}"
    data = await _api_get(url)
    if not data or "jobs" not in data:
        return []

    out = []
    for j in data["jobs"]:
        desc = BeautifulSoup(j.get("jobDescription", ""), "html.parser").get_text(" ", strip=True)[:3000]
        out.append(_norm({
            "title":            j.get("jobTitle", ""),
            "company":          j.get("companyName", ""),
            "location":         j.get("jobGeo", "Worldwide"),
            "remote_type":      "remote",
            "salary":           j.get("annualSalaryMin", "") and
                                f"${j['annualSalaryMin']:,}–${j.get('annualSalaryMax', j['annualSalaryMin']):,}/yr",
            "description":      desc,
            "skills":           extract_skills(desc),
            "apply_url":        j.get("url", ""),
            "posted_date":      (j.get("pubDate", "") or "")[:10],
            "source_domain":    "jobicy.com",
            "source_type":      "api_jobicy",
            "experience_level": infer_experience(j.get("jobTitle", "") + " " + desc),
        }))
    return [j for j in out if j["title"] and j["apply_url"]]


async def fetch_remoteok_jobs(query: str) -> List[Dict]:
    """
    RemoteOK.com free API — remote jobs JSON feed.
    https://remoteok.com/api
    """
    tag  = quote_plus(query.split()[0])
    data = await _api_get(
        f"https://remoteok.com/api?tags={tag}",
        headers={"Accept": "application/json", "User-Agent": "JobRadar/1.0"},
    )
    if not data or not isinstance(data, list):
        return []

    out = []
    for j in data:
        if not isinstance(j, dict) or not j.get("position"):
            continue
        desc = j.get("description", "")
        if "<" in desc:
            desc = BeautifulSoup(desc, "html.parser").get_text(" ", strip=True)
        desc = desc[:3000]

        salary = ""
        if j.get("salary_min"):
            mn = j["salary_min"] // 1000
            mx = (j.get("salary_max") or j["salary_min"]) // 1000
            salary = f"${mn}K–${mx}K/yr"

        out.append(_norm({
            "title":            j.get("position", ""),
            "company":          j.get("company", ""),
            "location":         "Remote",
            "remote_type":      "remote",
            "salary":           salary,
            "description":      desc,
            "skills":           extract_skills(desc + " " + " ".join(j.get("tags", []))),
            "apply_url":        j.get("apply_url") or j.get("url", ""),
            "posted_date":      (j.get("date", "") or "")[:10],
            "source_domain":    "remoteok.com",
            "source_type":      "api_remoteok",
            "experience_level": infer_experience(j.get("position", "") + " " + desc),
        }))
    return [j for j in out if j["title"] and j["apply_url"]]


async def fetch_himalayas_jobs(query: str) -> List[Dict]:
    """
    Himalayas.app free API — clean remote job data.
    https://himalayas.app/jobs/api
    """
    url  = f"https://himalayas.app/jobs/api?q={quote_plus(query)}&limit=50"
    data = await _api_get(url)
    if not data or "jobs" not in data:
        return []

    out = []
    for j in data["jobs"]:
        desc = j.get("description", "")
        if "<" in desc:
            desc = BeautifulSoup(desc, "html.parser").get_text(" ", strip=True)
        desc = desc[:3000]

        out.append(_norm({
            "title":            j.get("title", ""),
            "company":          j.get("company", {}).get("name", "") if isinstance(j.get("company"), dict) else j.get("company", ""),
            "location":         j.get("location", "Remote"),
            "remote_type":      "remote" if j.get("isRemote") else "hybrid",
            "salary":           j.get("salary", ""),
            "description":      desc,
            "skills":           extract_skills(desc),
            "apply_url":        j.get("applicationLink") or j.get("url", ""),
            "posted_date":      (j.get("createdAt", "") or "")[:10],
            "source_domain":    "himalayas.app",
            "source_type":      "api_himalayas",
            "experience_level": infer_experience(j.get("title", "") + " " + desc),
        }))
    return [j for j in out if j["title"] and j["apply_url"]]


async def fetch_themuse_jobs(query: str) -> List[Dict]:
    """
    The Muse free API — good company data, multiple pages.
    https://www.themuse.com/api/public/jobs
    """
    out = []
    q_lower = query.lower()

    for page in range(0, 3):   # pages 0, 1, 2
        url  = f"https://www.themuse.com/api/public/jobs?page={page}&descending=true"
        data = await _api_get(url)
        if not data or "results" not in data:
            break

        for j in data["results"]:
            name = j.get("name", "")
            desc_html = j.get("contents", "")
            desc = BeautifulSoup(desc_html, "html.parser").get_text(" ", strip=True)[:3000] if desc_html else ""

            # Filter by relevance
            if not any(w in (name + desc).lower() for w in q_lower.split()):
                continue

            company_obj = j.get("company", {})
            company = company_obj.get("name", "") if isinstance(company_obj, dict) else ""

            locs = j.get("locations", [])
            location = locs[0].get("name", "") if locs else "Remote"

            lvls  = j.get("levels", [])
            level_raw = lvls[0].get("name", "") if lvls else ""
            level_map = {
                "Entry Level": "junior", "Mid Level": "mid",
                "Senior Level": "senior", "Management": "senior",
            }
            level = level_map.get(level_raw, infer_experience(name + " " + desc))

            out.append(_norm({
                "title":            name,
                "company":          company,
                "location":         location,
                "remote_type":      "remote" if "remote" in location.lower() or "remote" in desc.lower() else "on-site",
                "description":      desc,
                "skills":           extract_skills(desc),
                "apply_url":        j.get("refs", {}).get("landing_page", ""),
                "posted_date":      (j.get("publication_date", "") or "")[:10],
                "source_domain":    "themuse.com",
                "source_type":      "api_themuse",
                "experience_level": level,
            }))

    return [j for j in out if j["title"] and j["apply_url"]]