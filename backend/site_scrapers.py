"""
Site-specific scrapers for 60+ job boards.
Each function returns List[dict] of raw job records.
Called by agents.py orchestrator during crawl waves.
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Optional
from urllib.parse import quote_plus, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from config import REQUEST_TIMEOUT, USER_AGENTS, ADZUNA_APP_ID, ADZUNA_APP_KEY
from config import JOB_SITES
from scraper import (
    extract_skills,
    infer_experience,
    extract_salary_text,
    fetch_html,
    fetch_paginated_jobs,
    _norm,
)
from playwright_pool import fetch_with_playwright

log = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_json(url: str, headers: dict = None) -> Optional[dict]:
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as c:
            r = await c.get(url, headers=headers or {
                "User-Agent": "Mozilla/5.0 (compatible; JobRadar/1.0)",
                "Accept": "application/json",
            })
            if r.status_code == 200:
                return r.json()
    except Exception as e:
        log.debug("JSON fetch failed %s: %s", url, e)
    return None


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html[:300_000], "html.parser")


def _make_job(title, company, location, remote_type, description,
              apply_url, source_domain, source_type="html",
              salary="", posted_date="", skills=None) -> dict:
    desc = description or ""
    return _norm({
        "title":            (title or "").strip()[:200],
        "company":          (company or "").strip()[:200],
        "location":         (location or "").strip()[:200],
        "remote_type":      remote_type or _infer_remote(desc + " " + (location or "")),
        "experience_level": infer_experience(title + " " + desc),
        "salary":           salary or extract_salary_text(desc),
        "description":      desc[:4000],
        "skills":           skills or extract_skills(desc),
        "apply_url":        apply_url,
        "posted_date":      posted_date[:10] if posted_date else "",
        "source_domain":    source_domain,
        "source_type":      source_type,
    })


def _infer_remote(text: str) -> str:
    t = text.lower()
    if "remote" in t: return "remote"
    if "hybrid" in t: return "hybrid"
    return "on-site"


# ── Workable API ──────────────────────────────────────────────────────────────

async def fetch_workable_jobs(query: str, max_pages: int = 10) -> List[dict]:
    out = []
    url = f"https://apply.workable.com/api/v3/jobs?query={quote_plus(query)}&limit=100"
    data = await _get_json(url)
    if not data:
        return out
    for j in data.get("results", []):
        out.append(_make_job(
            title=j.get("title"),
            company=j.get("company", {}).get("name"),
            location=j.get("location", {}).get("country"),
            remote_type="remote" if j.get("remote") else "on-site",
            description=j.get("description", ""),
            apply_url=j.get("url") or j.get("application_url", ""),
            source_domain="workable.com",
            source_type="api_workable",
            posted_date=j.get("published_on", ""),
        ))
    return [j for j in out if j["title"] and j["apply_url"]]


# ── Adzuna API ────────────────────────────────────────────────────────────────

async def fetch_adzuna_jobs(query: str, country: str = "us",
                             max_pages: int = 10) -> List[dict]:
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        return []
    out = []
    for page in range(1, max_pages + 1):
        url = (
            f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
            f"?app_id={ADZUNA_APP_ID}&app_key={ADZUNA_APP_KEY}"
            f"&results_per_page=50&what={quote_plus(query)}&content-type=application/json"
        )
        data = await _get_json(url)
        if not data or not data.get("results"):
            break
        for j in data["results"]:
            out.append(_make_job(
                title=j.get("title"),
                company=j.get("company", {}).get("display_name"),
                location=j.get("location", {}).get("display_name"),
                remote_type=_infer_remote(j.get("description", "")),
                description=j.get("description", ""),
                apply_url=j.get("redirect_url", ""),
                source_domain="adzuna.com",
                source_type="api_adzuna",
                salary=f"${j['salary_min']:.0f}–${j['salary_max']:.0f}" if j.get("salary_min") else "",
                posted_date=j.get("created", "")[:10],
            ))
    return [j for j in out if j["title"] and j["apply_url"]]


# ── HN Who's Hiring ───────────────────────────────────────────────────────────

async def fetch_hn_hiring(query: str, max_pages: int = 3) -> List[dict]:
    out = []
    q = quote_plus(f"hiring {query}")
    url = f"https://hn.algolia.com/api/v1/search?query={q}&tags=comment&page=0&hitsPerPage=100"
    data = await _get_json(url)
    if not data:
        return out
    for hit in data.get("hits", []):
        text = hit.get("comment_text") or hit.get("story_text") or ""
        if len(text) < 100:
            continue
        # Very rough title extraction
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        title_line = lines[0][:200] if lines else "Software Engineer"
        desc = " ".join(lines[:20])
        out.append(_make_job(
            title=title_line,
            company="HN Hiring",
            location="",
            remote_type=_infer_remote(desc),
            description=desc[:2000],
            apply_url=f"https://news.ycombinator.com/item?id={hit.get('objectID','')}",
            source_domain="news.ycombinator.com",
            source_type="hn_api",
            posted_date=hit.get("created_at", "")[:10],
        ))
    return out


# ── DevITJobs ─────────────────────────────────────────────────────────────────

async def fetch_devitjobs(query: str) -> List[dict]:
    data = await _get_json("https://devitjobs.us/api/jobsLight")
    if not data:
        return []
    q = query.lower()
    out = []
    for j in (data if isinstance(data, list) else []):
        title = j.get("position", "")
        desc  = j.get("description", "") + " " + " ".join(j.get("tags", []))
        if not any(word in (title + " " + desc).lower() for word in q.split()):
            continue
        out.append(_make_job(
            title=title,
            company=j.get("company", ""),
            location=j.get("location", "Remote"),
            remote_type="remote",
            description=desc,
            apply_url=j.get("application_link") or j.get("url", ""),
            source_domain="devitjobs.us",
            source_type="api_devitjobs",
            posted_date=j.get("publishedAt", "")[:10],
        ))
    return out


# ── We Work Remotely (HTML, paginated) ────────────────────────────────────────

async def fetch_weworkremotely(query: str, max_pages: int = 10) -> List[dict]:
    out = []
    for page in range(1, max_pages + 1):
        url = f"https://weworkremotely.com/remote-jobs/search?term={quote_plus(query)}&page={page}"
        html = await fetch_html(url)
        if not html:
            break
        soup = _soup(html)
        jobs_on_page = 0
        for li in soup.select("ul.jobs li"):
            a = li.select_one("a")
            if not a:
                continue
            href = urljoin("https://weworkremotely.com", a.get("href", ""))
            title_el = li.select_one(".title")
            company_el = li.select_one(".company")
            title = title_el.get_text(strip=True) if title_el else a.get_text(strip=True)
            company = company_el.get_text(strip=True) if company_el else ""
            out.append(_make_job(
                title=title, company=company, location="Remote",
                remote_type="remote", description="",
                apply_url=href, source_domain="weworkremotely.com",
            ))
            jobs_on_page += 1
        if jobs_on_page == 0:
            break
        await asyncio.sleep(0.5)
    return out


# ── Wellfound / AngelList (JS-heavy, uses Playwright) ────────────────────────

async def fetch_wellfound(query: str, max_pages: int = 15) -> List[dict]:
    out = []
    for page in range(1, max_pages + 1):
        url = f"https://wellfound.com/jobs?q={quote_plus(query)}&page={page}"
        html = await fetch_with_playwright(url, wait_selector="[data-test='JobSearchResult']")
        if not html:
            html = await fetch_html(url)
        if not html:
            break
        soup = _soup(html)
        items = soup.select("[data-test='JobSearchResult'], .styles_component__Ey28k")
        if not items:
            break
        for item in items:
            title_el = item.select_one("h2, h3, [data-test='job-title']")
            co_el    = item.select_one("[data-test='startup-name'], .styles_startupName")
            loc_el   = item.select_one("[data-test='location'], .styles_location")
            link_el  = item.select_one("a[href*='/jobs/']")
            title    = title_el.get_text(strip=True) if title_el else ""
            company  = co_el.get_text(strip=True) if co_el else ""
            location = loc_el.get_text(strip=True) if loc_el else ""
            href     = urljoin("https://wellfound.com", link_el["href"]) if link_el else url
            if title:
                out.append(_make_job(
                    title=title, company=company, location=location,
                    remote_type=_infer_remote(location),
                    description=item.get_text(" ", strip=True)[:600],
                    apply_url=href, source_domain="wellfound.com",
                    source_type="html_js_wellfound",
                ))
        await asyncio.sleep(0.8)
    return out


# ── Built In (JS-heavy, uses Playwright) ─────────────────────────────────────

async def fetch_builtin(query: str, subdomain: str = "", max_pages: int = 20) -> List[dict]:
    domain = f"builtin{subdomain}.com" if subdomain else "builtin.com"
    base   = f"https://{domain}"
    out    = []
    for page in range(1, max_pages + 1):
        url  = f"{base}/jobs/remote?search={quote_plus(query)}&page={page}"
        html = await fetch_with_playwright(url, wait_selector=".job-boundingbox")
        if not html:
            html = await fetch_html(url)
        if not html:
            break
        soup = _soup(html)
        cards = soup.select(".job-boundingbox, [data-id*='job']")
        if not cards:
            break
        for card in cards:
            title_el = card.select_one("h2, h3, .job-title")
            co_el    = card.select_one(".company-name, .employer")
            loc_el   = card.select_one(".location, .job-location")
            link_el  = card.select_one("a[href]")
            title    = title_el.get_text(strip=True) if title_el else ""
            if not title:
                continue
            href = urljoin(base, link_el["href"]) if link_el else url
            out.append(_make_job(
                title=title,
                company=co_el.get_text(strip=True) if co_el else "",
                location=loc_el.get_text(strip=True) if loc_el else "",
                remote_type="remote",
                description=card.get_text(" ", strip=True)[:600],
                apply_url=href, source_domain=domain,
                source_type="html_js_builtin",
            ))
        await asyncio.sleep(0.8)
    return out


# ── Startup.jobs ──────────────────────────────────────────────────────────────

async def fetch_startup_jobs(query: str, max_pages: int = 10) -> List[dict]:
    out = []
    for page in range(1, max_pages + 1):
        url  = f"https://startup.jobs/?q={quote_plus(query)}&page={page}"
        html = await fetch_html(url)
        if not html:
            break
        soup = _soup(html)
        cards = soup.select(".job, article.listing, [class*='job-card']")
        if not cards:
            break
        for card in cards:
            title_el = card.select_one("h2, h3, .title, .job-title")
            co_el    = card.select_one(".company, .employer")
            link_el  = card.select_one("a[href]")
            title    = title_el.get_text(strip=True) if title_el else ""
            if not title:
                continue
            href = urljoin("https://startup.jobs", link_el["href"]) if link_el else url
            out.append(_make_job(
                title=title,
                company=co_el.get_text(strip=True) if co_el else "",
                location="", remote_type=_infer_remote(card.get_text()),
                description=card.get_text(" ", strip=True)[:600],
                apply_url=href, source_domain="startup.jobs",
            ))
        await asyncio.sleep(0.4)
    return out


# ── Dispatcher: call the right scraper for a site_key ─────────────────────────

SITE_SCRAPER_MAP = {
    "workable":      lambda q, mp: fetch_workable_jobs(q, mp),
    "adzuna":        lambda q, mp: fetch_adzuna_jobs(q, max_pages=mp),
    "hn_whoishiring": lambda q, mp: fetch_hn_hiring(q, mp),
    "devitjobs":     lambda q, mp: fetch_devitjobs(q),
    "weworkremotely": lambda q, mp: fetch_weworkremotely(q, mp),
    "wellfound":     lambda q, mp: fetch_wellfound(q, mp),
    "builtin":       lambda q, mp: fetch_builtin(q, max_pages=mp),
    "builtin_nyc":   lambda q, mp: fetch_builtin(q, subdomain="nyc", max_pages=mp),
    "builtin_sf":    lambda q, mp: fetch_builtin(q, subdomain="sf", max_pages=mp),
    "startup_jobs":  lambda q, mp: fetch_startup_jobs(q, mp),
}


async def scrape_site(site_key: str, query: str, max_pages: int = 20) -> List[dict]:
    """Dispatch to the correct site scraper. Returns job list."""
    fn = SITE_SCRAPER_MAP.get(site_key)
    if fn:
        try:
            return await fn(query, max_pages) or []
        except Exception as e:
            log.warning("Site scraper %s failed: %s", site_key, e)
            return []

    # Generic site pipeline for declarative JOB_SITES entries.
    cfg = JOB_SITES.get(site_key) if isinstance(JOB_SITES, dict) else None
    if not cfg:
        return []

    url_template = cfg.get("url")
    if not url_template:
        return []

    site_type = (cfg.get("type") or "html").lower()
    use_js = site_type == "html_js"

    try:
        return await fetch_paginated_jobs(
            base_url=url_template, query=query, max_pages=max_pages, use_js=use_js
        )
    except Exception as e:
        log.warning("Generic site crawl failed (%s): %s", site_key, e)
        return []
