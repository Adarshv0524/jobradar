"""
LLM-orchestrated deep search agent — JobRadar

Search architecture:
  Wave 0 — Free structured APIs: Remotive, Arbeitnow, Jobicy, RemoteOK, Himalayas, TheMuse
           (instant, structured JSON, no scraping needed)
  Wave 1 — LLM plans 20+ queries → parallel DDG searches → concurrent URL fetch
           (asyncio.Semaphore limits concurrency)
  Wave N — LLM evaluates quality → generates refined queries → repeat
           (keeps going until MIN_QUALITY_JOBS satisfied or budget exhausted)

Observability:
  Every significant decision emits a "trace" SSE event the UI can display.
  Token usage is tracked per LLM call and accumulated.
  Sites discovered + jobs-per-site are emitted as "site" events.
  Running stats are emitted every 5 pages as "stats" events.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus, urlparse

from config import (
    HIGH_QUALITY_THRESHOLD,
    MAX_CONCURRENT_FETCHES,
    MAX_JOBS_PER_SESSION,
    MAX_PAGES_TOTAL,
    MAX_QUERIES_PER_SESSION,
    MAX_URLS_PER_QUERY,
    MIN_QUALITY_JOBS,
    MIN_SCORE_THRESHOLD,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)
from ranker import rank_jobs, score_job
from scraper import (
    extract_jobs_from_url,
    fetch_html,
    find_job_links,
    is_fake_job,
    search_web,
    fetch_remotive_jobs,
    fetch_arbeitnow_jobs,
    fetch_jobicy_jobs,
    fetch_remoteok_jobs,
    fetch_himalayas_jobs,
    fetch_themuse_jobs,
)
from store import (
    get_feedback_profile,
    insert_job,
    insert_crawl_plan_entry,
    update_crawl_plan,
    mark_visited,
    update_session,
    update_source,
    url_visited,
)
from site_scrapers import scrape_site, SITE_SCRAPER_MAP
from config import JOB_SITES, JS_HEAVY_SITES

log = logging.getLogger(__name__)

# ── OpenAI ────────────────────────────────────────────────────────────────────

_openai_client = None
_LLM_AVAILABLE  = False

try:
    from openai import AsyncOpenAI
    if OPENAI_API_KEY:
        _openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        _LLM_AVAILABLE  = True
except ImportError:
    log.warning("openai not installed — heuristic mode only")

# ── Session state ─────────────────────────────────────────────────────────────

@dataclass
class _State:
    session_id:    str
    query:         str
    profile:       dict
    prefs:         dict
    feedback:      dict

    jobs:          List[dict]    = field(default_factory=list)
    visited:       Set[str]      = field(default_factory=set)
    apply_urls:    Set[str]      = field(default_factory=set)
    pages_crawled: int           = 0
    queries_used:  int           = 0
    tok_prompt:    int           = 0
    tok_comp:      int           = 0
    sites:         Dict[str,int] = field(default_factory=dict)   # domain → jobs count
    url_queue:     asyncio.Queue  = field(default_factory=asyncio.Queue)

    @property
    def total_tokens(self) -> int:
        return self.tok_prompt + self.tok_comp

    @property
    def high_quality(self) -> int:
        return sum(1 for j in self.jobs if j.get("score", 0) >= HIGH_QUALITY_THRESHOLD)

# ── Tool schemas ──────────────────────────────────────────────────────────────

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_search_queries",
            "description": (
                "Generate a large, diverse set of targeted search queries to find job postings. "
                "Cover: exact role titles, skill combos, ATS site-specific (site:greenhouse.io, "
                "site:lever.co, site:ashbyhq.com, site:workday.com), company sizes, and location/remote variants."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of search queries. Target 15-20 items.",
                        "maxItems": 20,
                    },
                    "reasoning": {"type": "string"},
                    "ats_specific": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Queries that specifically target ATS job boards.",
                        "maxItems": 8,
                    },
                },
                "required": ["queries", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "prioritize_urls",
            "description": (
                "Select the most promising URLs to scrape for job postings. "
                "STRONGLY prefer: ATS pages (greenhouse.io, lever.co, workday.com), "
                "company career pages, structured job boards. "
                "STRONGLY avoid: job aggregators (Indeed, LinkedIn, Glassdoor — these block scraping), "
                "news articles, blog posts, and 'X jobs in Y city' aggregator listings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "selected_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 12,
                        "description": "URLs to visit. Best quality first.",
                    },
                    "skip_reasons": {
                        "type": "object",
                        "description": "URL → reason for skipping.",
                    },
                    "reasoning": {"type": "string"},
                },
                "required": ["selected_urls"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_and_plan",
            "description": (
                "Evaluate the quality of jobs found so far and plan the next search wave. "
                "Be strict: only declare satisfied=true when there are 50+ diverse, real job listings "
                "with proper titles and companies (not aggregator pages)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "quality_assessment": {"type": "string"},
                    "satisfied": {
                        "type": "boolean",
                        "description": "True only if results are genuinely good.",
                    },
                    "issues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Problems with current results (e.g. 'too many aggregator titles', 'missing senior roles').",
                    },
                    "next_strategy": {"type": "string"},
                    "new_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 10,
                        "description": "Next wave of queries to try.",
                    },
                },
                "required": ["quality_assessment", "satisfied"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "infer_salary",
            "description": "Estimate a salary range for a job based on role, experience level, skills, and location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_usd": {"type": "integer", "description": "Annual USD minimum"},
                    "max_usd": {"type": "integer", "description": "Annual USD maximum"},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    "basis": {"type": "string", "description": "Reason for estimate."},
                },
                "required": ["min_usd", "max_usd", "confidence", "basis"],
            },
        },
    },
]


# ── LLM helpers ───────────────────────────────────────────────────────────────

async def _llm_call(
    messages: List[dict],
    tools: Optional[List[dict]] = None,
    tool_choice: Optional[dict] = None,
    max_tokens: int = 1500,
    state: Optional[_State] = None,
) -> Tuple[Optional[dict], int, int]:
    """
    Returns (args_dict, prompt_tokens, completion_tokens).
    args_dict is None on failure.
    """
    if not _LLM_AVAILABLE or _openai_client is None:
        return None, 0, 0
    try:
        kwargs: dict = {
            "model":      OPENAI_MODEL,
            "messages":   messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"]       = tools
            kwargs["tool_choice"] = tool_choice or "auto"

        resp    = await _openai_client.chat.completions.create(**kwargs)
        usage   = resp.usage
        p_tok   = usage.prompt_tokens     if usage else 0
        c_tok   = usage.completion_tokens if usage else 0

        if state:
            state.tok_prompt += p_tok
            state.tok_comp   += c_tok

        choice = resp.choices[0]
        if choice.message.tool_calls:
            tc = choice.message.tool_calls[0]
            return json.loads(tc.function.arguments), p_tok, c_tok

        return {"text": choice.message.content or ""}, p_tok, c_tok

    except Exception as exc:
        log.warning("LLM call failed: %s", exc)
        return None, 0, 0


def preprocess_query(raw_query: str, prefs: dict) -> dict:
    """
    Normalize a raw user query into structured search intent.

    Returns dict with:
        canonical_query   — cleaned, normalized query string
        role_title        — inferred role title
        exp_level         — inferred or overridden exp level
        location_hints    — list of location terms to inject
        extra_terms       — extra terms to append to queries

    Examples:
        "data engineer fresher" → level=junior, extra=["entry level", "fresher", "0-1 years"]
        "junior python dev india" → role="python developer", location=["india", "bangalore"]
    """
    q   = raw_query.lower().strip()
    loc = (prefs.get("location") or "").lower()

    # ── Exp level hints from query text ──────────────────────────────────────
    exp_from_prefs  = prefs.get("experience_level", "")
    exp_from_query  = ""
    fresher_signals = ["fresher", "fresh graduate", "new grad", "entry level",
                       "entry-level", "0 experience", "0 years", "no experience"]
    junior_signals  = ["junior", "jr.", "jr ", "beginner", "associate", "trainee"]
    senior_signals  = ["senior", "sr.", "sr ", "lead", "staff", "principal"]

    if any(s in q for s in fresher_signals):
        exp_from_query = "junior"
    elif any(s in q for s in junior_signals):
        exp_from_query = "junior"
    elif any(s in q for s in senior_signals):
        exp_from_query = "senior"

    effective_exp = exp_from_prefs or exp_from_query or ""

    # ── Location hints ────────────────────────────────────────────────────────
    india_signals = ["india", "bangalore", "bengaluru", "hyderabad", "pune",
                     "mumbai", "chennai", "delhi", "ncr", "noida", "gurgaon"]
    location_hints = []
    if any(s in loc for s in india_signals) or any(s in q for s in india_signals):
        location_hints = ["India", "Bangalore", "Hyderabad", "Pune", "remote India"]

    # ── Extra query terms for experience level ────────────────────────────────
    extra_terms = []
    if effective_exp in ("junior", "intern"):
        extra_terms = ["junior", "entry level", "fresher", "0-2 years", "new grad",
                       "associate", "graduate trainee"]
    elif effective_exp == "mid":
        extra_terms = ["mid-level", "2-4 years", "intermediate"]
    elif effective_exp in ("senior", "lead"):
        extra_terms = ["senior", "5+ years", "lead", "staff"]

    # ── Clean the canonical query (remove level/location words) ──────────────
    stop = fresher_signals + junior_signals + senior_signals + india_signals + [
        "remote", "hybrid", "onsite", "on-site", "job", "jobs", "hiring",
        "position", "role", "opening",
    ]
    tokens = q.split()
    cleaned = " ".join(t for t in tokens if t not in stop and len(t) > 2)
    canonical = cleaned.strip() or raw_query.strip()

    return {
        "canonical_query": canonical,
        "role_title":      canonical,
        "exp_level":       effective_exp,
        "location_hints":  location_hints,
        "extra_terms":     extra_terms,
    }


async def _plan_queries(st: "_State") -> "Tuple[List[str], str]":
    """Phase 1: LLM generates diverse search queries with location + exp awareness."""

    parsed   = preprocess_query(st.query, st.prefs)
    skills   = ", ".join(st.profile.get("skills", [])[:15]) or "not specified"
    loc      = st.prefs.get("location") or "any"
    exp      = parsed["exp_level"] or st.prefs.get("experience_level") or "not specified"
    hints    = parsed["location_hints"]
    extra    = parsed["extra_terms"]
    role     = parsed["canonical_query"]

    # Build location-specific instruction
    if hints:
        loc_instruction = (
            f"Location is '{loc}'. MUST include queries targeting: {', '.join(hints[:3])}. "
            "Do NOT generate queries for United States / Americas unless also pairing with 'remote worldwide'."
        )
    else:
        loc_instruction = f"Location: {loc}."

    # Build exp-level instruction
    if exp in ("junior", "intern"):
        exp_instruction = (
            f"Experience level is '{exp}' (FRESHER / 0-2 years). "
            "ALL queries must target entry-level / junior / fresher / new-grad roles ONLY. "
            "NEVER include 'senior', 'lead', 'staff', 'principal', 'manager' in any query. "
            f"Include terms like: {', '.join(extra[:5])}."
        )
    elif exp == "senior":
        exp_instruction = (
            f"Experience level is senior (5+ years). "
            "Focus on senior / lead / staff engineer queries."
        )
    else:
        exp_instruction = f"Experience level: {exp or 'not specified'}."

    system = (
        "You are a world-class job search strategist with deep knowledge of "
        "tech job markets globally, especially India and remote-first companies. "
        "Generate targeted queries to find REAL job postings — not aggregator pages. "
        "Focus on ATS platforms and company career pages."
    )
    user = f"""Target role: {role}
Original query: {st.query}
Skills: {skills}
{loc_instruction}
Remote preference: {st.prefs.get('remote_preference') or 'any'}
{exp_instruction}
Summary: {(st.profile.get('summary') or '')[:300]}

Generate 15-20 diverse queries. MUST include:
- 5+ ATS-specific (site:greenhouse.io, site:lever.co, site:ashbyhq.com, site:boards.greenhouse.io, site:jobs.lever.co)
- Role title variations (e.g. 'data engineer', 'data pipeline engineer', 'ETL engineer')
- Skill-combo queries (combine role + top skills from profile)
- Location variants: {', '.join(hints) if hints else 'worldwide / remote'}
- Experience-level terms: {', '.join(extra[:4]) if extra else 'any level'}
- Company-type variants (startup, product company, MNC, FAANG)

IMPORTANT: If location is India, include at least 3 India-specific queries.
If level is junior/fresher, every single query must use junior/entry-level language.
"""
    result, p, c = await _llm_call(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        tools=[_TOOLS[0]],
        tool_choice={"type": "function", "function": {"name": "generate_search_queries"}},
        max_tokens=1600,
        state=st,
    )
    if result and "queries" in result:
        all_q = result.get("queries", []) + result.get("ats_specific", [])
        qs    = [q.strip() for q in all_q if q.strip()]
        return qs[:MAX_QUERIES_PER_SESSION], result.get("reasoning", "")

    return _smart_heuristic_queries(st.query, st.prefs, parsed), "heuristic fallback"


async def _pick_urls(results: List[dict], st: _State) -> Tuple[List[str], str]:
    """LLM picks best URLs from DDG results."""
    snippets = "\n".join(
        f"[{i+1}] {r.get('title','')} | {r.get('url','')} | {r.get('snippet','')[:80]}"
        for i, r in enumerate(results[:25])
    )
    result, _, _ = await _llm_call(
        messages=[
            {"role":"system","content":
                "Expert web scraper. Prefer ATS/career pages. SKIP aggregators (indeed.com, "
                "linkedin.com, glassdoor.com, naukri.com, timesjobs.com — these block scraping "
                "or return listing pages, not real jobs). SKIP pages titled 'X,000 jobs in Y city'."},
            {"role":"user","content":
                f"Role: {st.query}\n\nResults:\n{snippets}\n\nSelect best URLs to scrape for real job listings."},
        ],
        tools=[_TOOLS[1]],
        tool_choice={"type":"function","function":{"name":"prioritize_urls"}},
        max_tokens=700,
        state=st,
    )
    if result and "selected_urls" in result:
        reason = result.get("reasoning", "LLM prioritized")
        return result["selected_urls"][:MAX_URLS_PER_QUERY], reason

    # Fallback: filter known aggregators
    blocked = {"indeed.com","linkedin.com","glassdoor.com","naukri.com",
               "timesjobs.com","monster.com","ziprecruiter.com","dice.com"}
    good = [r["url"] for r in results
            if r.get("url") and not any(b in r["url"] for b in blocked)]
    return good[:MAX_URLS_PER_QUERY], "heuristic filter"


async def _evaluate(st: _State, queries_used: int) -> Tuple[bool, List[str], str]:
    """LLM evaluates quality. Returns (should_continue, new_queries, strategy)."""
    top = sorted(st.jobs, key=lambda x: x.get("score",0), reverse=True)[:12]
    summary = "\n".join(
        f"  {j.get('title','')} @ {j.get('company','?')} | score={j.get('score',0):.2f} "
        f"| {j.get('remote_type','')} | salary={j.get('salary','?')}"
        for j in top
    )
    result, _, _ = await _llm_call(
        messages=[
            {"role":"system","content":
                "Job search quality evaluator. Be strict — 'Unknown company' or aggregator-style "
                "titles ('2000 jobs in...') mean low quality. Good quality = real companies, real roles."},
            {"role":"user","content":
                f"Target: {st.query}\n"
                f"Total jobs: {len(st.jobs)} | High-quality: {st.high_quality} | "
                f"Pages crawled: {st.pages_crawled} | Queries used: {queries_used}/{MAX_QUERIES_PER_SESSION}\n\n"
                f"Top results:\n{summary}\n\n"
                "Are these results good? What should we search for next?"},
        ],
        tools=[_TOOLS[2]],
        tool_choice={"type":"function","function":{"name":"evaluate_and_plan"}},
        max_tokens=900,
        state=st,
    )
    if result:
        satisfied = result.get("satisfied", False)
        issues    = result.get("issues", [])
        new_q     = result.get("new_queries", [])
        strategy  = result.get("next_strategy", "")
        return not satisfied, new_q, f"{strategy} | issues: {'; '.join(issues[:2])}"

    # Heuristic fallback
    should_go = st.high_quality < MIN_QUALITY_JOBS and queries_used < MAX_QUERIES_PER_SESSION
    return should_go, [], ""


async def _infer_salary(job: dict, st: _State) -> Optional[str]:
    """Ask LLM to estimate salary for jobs without one."""
    if job.get("salary"):
        return None  # already has one
    result, _, _ = await _llm_call(
        messages=[
            {"role":"system","content":"Salary estimation expert. Give realistic market ranges."},
            {"role":"user","content":
                f"Estimate salary for:\nTitle: {job.get('title','')}\n"
                f"Company: {job.get('company','')}\nLocation: {job.get('location','')}\n"
                f"Remote: {job.get('remote_type','')}\nLevel: {job.get('experience_level','')}\n"
                f"Skills: {', '.join(job.get('skills',[])[:8])}\n"
                "Give annual USD range."},
        ],
        tools=[_TOOLS[3]],
        tool_choice={"type":"function","function":{"name":"infer_salary"}},
        max_tokens=200,
        state=st,
    )
    if result and "min_usd" in result:
        conf = result.get("confidence", "low")
        mn   = result["min_usd"] // 1000
        mx   = result["max_usd"] // 1000
        return f"est. ${mn}K–${mx}K/yr ({conf} confidence)"
    return None


# ── Heuristic fallback (no LLM) ───────────────────────────────────────────────

def _smart_heuristic_queries(query: str, prefs: dict, parsed: dict) -> "List[str]":
    """
    Improved heuristic fallback that injects location and exp level terms.
    Replaces the old _heuristic_queries().
    """
    base   = parsed["canonical_query"] or query.strip()
    loc    = prefs.get("location", "")
    exp    = parsed["exp_level"] or prefs.get("experience_level", "")
    hints  = parsed["location_hints"]
    extras = parsed["extra_terms"][:3]

    qs = []

    # Base + exp modifier
    for e in (extras or [base]):
        qs.append(f"{base} {e}")

    # ATS queries — always include
    qs += [
        f"{base} site:boards.greenhouse.io",
        f"{base} site:jobs.lever.co",
        f"{base} site:ashbyhq.com",
        f"{base} site:myworkdayjobs.com",
        f"{base} site:careers.icims.com",
    ]

    # Location-specific
    for h in hints[:3]:
        qs.append(f"{base} {h}")
        if extras:
            qs.append(f"{base} {extras[0]} {h}")

    # Generic fallback
    qs += [
        f"{base} remote",
        f"{base} job openings",
        f"{base} hiring 2025",
    ]

    # Deduplicate and cap
    seen, out = set(), []
    for q in qs:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out[:MAX_QUERIES_PER_SESSION]


# ── Concurrent page fetcher ────────────────────────────────────────────────────

_semaphore: Optional[asyncio.Semaphore] = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)
    return _semaphore


async def _fetch_url_bounded(
    url: str,
    st: _State,
) -> Tuple[str, List[dict], str]:
    """Bounded concurrent fetch. Returns (url, raw_jobs, page_type)."""
    async with _get_semaphore():
        if url in st.visited or await url_visited(url):
            return url, [], "skipped"

        st.visited.add(url)
        await mark_visited(url, st.session_id)

        raw_jobs, page_type = await extract_jobs_from_url(url)
        st.pages_crawled += 1

        # Follow job links from listing pages (bounded to 8 sub-pages)
        if page_type in ("job_listing", "careers_home") and st.pages_crawled < MAX_PAGES_TOTAL:
            html = await fetch_html(url)
            if html:
                links = find_job_links(html, url)
                sub_tasks = []
                for link in links[:8]:
                    if link not in st.visited and st.pages_crawled < MAX_PAGES_TOTAL:
                        st.visited.add(link)
                        sub_tasks.append(_fetch_single(link, st))
                if sub_tasks:
                    sub_results = await asyncio.gather(*sub_tasks, return_exceptions=True)
                    for sub in sub_results:
                        if isinstance(sub, list):
                            raw_jobs.extend(sub)

        domain = urlparse(url).netloc
        if raw_jobs:
            await update_source(domain, True, len(raw_jobs))
            st.sites[domain] = st.sites.get(domain, 0) + len(raw_jobs)
        else:
            await update_source(domain, False)

        return url, raw_jobs, page_type


async def _fetch_single(url: str, st: _State) -> List[dict]:
    """Simple bounded fetch for sub-pages (no link following)."""
    async with _get_semaphore():
        await mark_visited(url, st.session_id)
        jobs, _ = await extract_jobs_from_url(url)
        st.pages_crawled += 1
        return jobs


# ── Job processing ────────────────────────────────────────────────────────────

async def _process_raw_jobs(
    raw_jobs: "List[dict]",
    st: "_State",
    infer_missing_salary: bool = False,
) -> "AsyncGenerator[dict, None]":
    """Score, dedup, store jobs and yield them.
    
    FIX: Added pre-scoring filters:
    1. Garbage / HN opinion post filter
    2. Hard experience-level mismatch filter (avoids wasting LLM tokens on irrelevant jobs)
    """
    from ranker import _exp_rank, _is_garbage_title

    pref_exp = (st.prefs.get("experience_level") or "").lower()

    for raw in raw_jobs:
        if len(st.jobs) >= MAX_JOBS_PER_SESSION:
            return

        apply_url = raw.get("apply_url", "")
        if apply_url and apply_url in st.apply_urls:
            continue

        # ── FIX: Garbage title filter ────────────────────────────────────────
        title = raw.get("title", "")
        if _is_garbage_title(title):
            continue

        # ── FIX: HN opinion-post filter ──────────────────────────────────────
        # HN posts that are opinions/discussions have no apply_url AND
        # the "title" is actually the comment text (very long)
        if not apply_url and len(title) > 80:
            continue

        # ── FIX: Aggregator noise filter ─────────────────────────────────────
        if any(pat in title.lower() for pat in [
            "jobs in ", "+jobs", "k jobs", "job openings",
            "careers at indeed", "jobs near", "hiring now",
            "✓ we", "&#x",
        ]):
            continue

        fake, _ = is_fake_job(title, raw.get("description", ""))
        if fake:
            continue

        # ── FIX: Hard exp-level pre-filter ───────────────────────────────────
        # Skip senior/lead/staff jobs entirely when user wants junior/fresher.
        # This avoids scoring (and storing) thousands of irrelevant senior jobs.
        if pref_exp in ("junior", "intern", "fresher", "entry"):
            job_exp = (raw.get("experience_level") or "").lower()
            if job_exp in ("senior", "lead", "staff", "principal", "director"):
                continue
            # Also check title keywords
            title_low = title.lower()
            if any(k in title_low for k in [
                "senior", "sr.", " sr ", "lead ", "staff ", "principal",
                "head of", "director", "vp of", "manager",
            ]):
                continue

        # Infer missing salary via LLM (only for high-potential jobs)
        if infer_missing_salary and not raw.get("salary"):
            inferred = await _infer_salary(raw, st)
            if inferred:
                raw["salary"]          = inferred
                raw["salary_inferred"] = True

        scored = score_job(raw, st.profile, st.prefs, st.feedback)
        if scored.get("score", 0) < MIN_SCORE_THRESHOLD:
            continue

        scored.update({
            "session_id":   st.session_id,
            "is_fake":      False,
            "is_duplicate": False,
        })

        job_id = await insert_job(scored)
        if job_id is None:
            continue  # DB-level duplicate

        scored["id"] = job_id
        if apply_url:
            st.apply_urls.add(apply_url)
        st.jobs.append(scored)

        yield scored


# ── Main session runner ───────────────────────────────────────────────────────

def _trace(level: str, msg: str, **data) -> dict:
    icons = {
        "plan":    "🧠",
        "search":  "🔍",
        "fetch":   "⚡",
        "extract": "📦",
        "api":     "🌐",
        "llm":     "💬",
        "eval":    "📊",
        "skip":    "⏭",
        "warn":    "⚠",
        "crawl":   "🕷",
        "done":    "✓",
    }
    return {
        "type":    "trace",
        "level":   level,
        "icon":    icons.get(level, "·"),
        "message": msg,
        "data":    data,
        "ts":      time.time(),
    }


def _stats_event(st: _State, queries_used: int) -> dict:
    return {
        "type":          "stats",
        "jobs_found":    len(st.jobs),
        "pages_crawled": st.pages_crawled,
        "queries_used":  queries_used,
        "tokens_total":  st.total_tokens,
        "tokens_prompt": st.tok_prompt,
        "tokens_comp":   st.tok_comp,
        "high_quality":  st.high_quality,
        "sites_count":   len(st.sites),
        "llm_enabled":   _LLM_AVAILABLE,
    }


async def run_search_session(
    session_id: str,
    query:      str,
    preferences: dict,
    profile:    dict,
) -> AsyncGenerator[dict, None]:
    """
    Async generator — yields SSE-compatible dicts:
      {"type": "progress",  "message": "…"}
      {"type": "trace",     "level": "plan|search|fetch|...", "message": "…", "data": {...}}
      {"type": "stats",     "jobs_found": N, "pages_crawled": M, "tokens_total": T, …}
      {"type": "site",      "domain": "…", "jobs_count": N}
      {"type": "job",       "job": {...}}
      {"type": "done",      "jobs_found": N, "pages_crawled": M}
    """

    feedback = await get_feedback_profile(session_id)
    st = _State(
        session_id=session_id,
        query=query,
        profile=profile,
        prefs=preferences,
        feedback=feedback,
    )

    mode_badge = "GPT-4.1 guided" if _LLM_AVAILABLE else "heuristic"
    yield {"type": "progress", "message": f"JobRadar starting [{mode_badge}]…"}
    yield _stats_event(st, 0)

    # ══════════════════════════════════════════════════════════════════════════
    # Wave 0 — Free structured APIs (fast, high-quality structured data)
    # ══════════════════════════════════════════════════════════════════════════

    yield _trace("api", "Calling free job APIs (no scraping needed)…")

    api_tasks = [
        ("Remotive",   fetch_remotive_jobs(query)),
        ("Arbeitnow",  fetch_arbeitnow_jobs(query)),
        ("Jobicy",     fetch_jobicy_jobs(query)),
        ("RemoteOK",   fetch_remoteok_jobs(query)),
        ("Himalayas",  fetch_himalayas_jobs(query)),
        ("TheMuse",    fetch_themuse_jobs(query)),
    ]

    api_results = await asyncio.gather(
        *[t for _, t in api_tasks], return_exceptions=True
    )

    for (name, _), result in zip(api_tasks, api_results):
        if isinstance(result, Exception):
            yield _trace("warn", f"{name}: failed ({result})")
            continue

        jobs_from_api = result if isinstance(result, list) else []
        yield _trace("api", f"{name} → {len(jobs_from_api)} jobs found",
                     source=name, count=len(jobs_from_api))

        if jobs_from_api:
            st.sites[name.lower()] = len(jobs_from_api)
            yield {"type": "site", "domain": name.lower(), "jobs_count": len(jobs_from_api)}

        async for job in _process_raw_jobs(jobs_from_api, st, infer_missing_salary=False):
            yield {"type": "job", "job": job}

    yield _stats_event(st, 0)
    yield _trace("api", f"Free APIs done: {len(st.jobs)} jobs so far")

    # ══════════════════════════════════════════════════════════════════════════
    # Wave 0.5 — Site-specific crawlers (60+ sites, deep pagination)
    # ══════════════════════════════════════════════════════════════════════════

    yield _trace("plan", f"Building crawl plan for {len(JOB_SITES)} job sites…")

    # Build crawl plan in DB
    crawlable_sites = {k: v for k, v in JOB_SITES.items()
                       if v.get("type") not in ("api",) and k in SITE_SCRAPER_MAP}

    for site_key, site_cfg in crawlable_sites.items():
        await insert_crawl_plan_entry(
            session_id=session_id,
            site_key=site_key,
            site_label=site_cfg["label"],
            url_template=site_cfg.get("url", ""),
            max_pages=site_cfg.get("max_pages", 20),
        )
        yield {
            "type": "plan_update",
            "site_key": site_key,
            "site_label": site_cfg["label"],
            "status": "pending",
            "jobs_found": 0,
            "pages_done": 0,
            "pages_total": site_cfg.get("max_pages", 20),
        }

    yield _trace("plan", f"Crawl plan built — {len(crawlable_sites)} sites queued")

    # Run site-specific crawlers in batches of 5 (respects semaphore internally)
    BATCH_SIZE = 5
    site_items = list(crawlable_sites.items())

    for batch_start in range(0, len(site_items), BATCH_SIZE):
        batch = site_items[batch_start: batch_start + BATCH_SIZE]

        # Mark batch as running
        for site_key, site_cfg in batch:
            await update_crawl_plan(session_id, site_key, status="running")
            yield {
                "type": "plan_update",
                "site_key": site_key,
                "site_label": site_cfg["label"],
                "status": "running",
                "jobs_found": 0,
                "pages_done": 0,
                "pages_total": site_cfg.get("max_pages", 20),
            }

        # Run batch concurrently
        async def _crawl_site(site_key: str, site_cfg: dict):
            max_p = site_cfg.get("max_pages", 20)
            try:
                raw_jobs = await scrape_site(site_key, query, max_pages=max_p)
                return site_key, site_cfg, raw_jobs, None
            except Exception as e:
                return site_key, site_cfg, [], str(e)

        tasks = [_crawl_site(sk, sc) for sk, sc in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in results:
            if isinstance(res, Exception):
                continue
            site_key, site_cfg, raw_jobs, err = res

            if err:
                await update_crawl_plan(session_id, site_key, status="failed", error_msg=err)
                yield {
                    "type": "plan_update", "site_key": site_key,
                    "site_label": site_cfg["label"], "status": "failed",
                    "jobs_found": 0, "pages_done": 0,
                    "pages_total": site_cfg.get("max_pages", 20),
                }
                continue

            job_count = 0
            async for job in _process_raw_jobs(raw_jobs, st, infer_missing_salary=False):
                yield {"type": "job", "job": job}
                job_count += 1

            await update_crawl_plan(
                session_id, site_key, status="done",
                jobs_found=job_count, pages_done=site_cfg.get("max_pages", 20)
            )
            yield {
                "type": "plan_update",
                "site_key": site_key,
                "site_label": site_cfg["label"],
                "status": "done",
                "jobs_found": job_count,
                "pages_done": site_cfg.get("max_pages", 20),
                "pages_total": site_cfg.get("max_pages", 20),
            }
            if job_count > 0:
                st.sites[site_cfg["label"]] = job_count
                yield {"type": "site", "domain": site_cfg["label"], "jobs_count": job_count}

        yield _stats_event(st, 0)

    yield _trace("api", f"Site crawl complete: {len(st.jobs)} jobs total across {len(st.sites)} sources")

    # ══════════════════════════════════════════════════════════════════════════
    # Wave 1+ — LLM-planned web crawl
    # ══════════════════════════════════════════════════════════════════════════

    yield _trace("plan", "LLM planning search queries…")

    if _LLM_AVAILABLE:
        pending_queries, reasoning = await _plan_queries(st)
        yield _trace("plan",
                     f"Generated {len(pending_queries)} queries",
                     reasoning=reasoning,
                     queries=pending_queries[:6],
                     tokens=st.total_tokens)
    else:
        pending_queries = _heuristic_queries(query, preferences)
        yield _trace("plan", f"Heuristic: {len(pending_queries)} queries",
                     queries=pending_queries[:6])

    yield {"type": "progress",
           "message": f"Starting deep crawl: {len(pending_queries)} search strategies"}

    queries_used = 0

    while pending_queries and queries_used < MAX_QUERIES_PER_SESSION:
        # Take 3–5 queries per wave and run them in parallel
        wave      = pending_queries[:4]
        pending_queries = pending_queries[4:]
        queries_used   += len(wave)

        # ── Parallel DDG searches ──────────────────────────────────────────
        yield _trace("search", f"Searching {len(wave)} queries in parallel…",
                     queries=wave)

        search_tasks = [search_web(q, max_results=MAX_URLS_PER_QUERY + 4) for q in wave]
        search_batches = await asyncio.gather(*search_tasks, return_exceptions=True)

        all_results: List[dict] = []
        for q, batch in zip(wave, search_batches):
            if isinstance(batch, Exception):
                yield _trace("warn", f"Search failed: {q[:50]}")
                continue
            yield _trace("search", f"'{q[:55]}' → {len(batch)} URLs",
                         query=q, count=len(batch))
            all_results.extend(batch)

        if not all_results:
            continue

        # ── LLM picks best URLs ────────────────────────────────────────────
        if _LLM_AVAILABLE:
            urls_to_fetch, pick_reason = await _pick_urls(all_results, st)
            yield _trace("llm", f"URL prioritization: {pick_reason[:80]}",
                         selected=len(urls_to_fetch),
                         tokens=st.total_tokens)
        else:
            blocked = {"indeed.com","linkedin.com","glassdoor.com","naukri.com",
                       "timesjobs.com","monster.com","ziprecruiter.com","shine.com"}
            urls_to_fetch = [
                r["url"] for r in all_results
                if r.get("url") and not any(b in r.get("url","") for b in blocked)
            ][:MAX_URLS_PER_QUERY]

        # ── Filter already-visited ─────────────────────────────────────────
        fresh_urls = [u for u in urls_to_fetch if u not in st.visited]

        if not fresh_urls:
            yield _trace("skip", "All URLs already visited — skipping")
            continue

        # ── Concurrent fetch ───────────────────────────────────────────────
        yield _trace("fetch", f"Fetching {len(fresh_urls)} URLs concurrently…",
                     urls=fresh_urls[:6])

        fetch_tasks = [_fetch_url_bounded(u, st) for u in fresh_urls
                       if st.pages_crawled < MAX_PAGES_TOTAL]
        fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        for item in fetch_results:
            if isinstance(item, Exception):
                yield _trace("warn", f"Fetch error: {item}")
                continue
            url, raw_jobs, page_type = item

            domain   = urlparse(url).netloc
            job_count = 0

            async for job in _process_raw_jobs(
                raw_jobs, st,
                infer_missing_salary=(_LLM_AVAILABLE and len(st.jobs) < 100),
            ):
                yield {"type": "job", "job": job}
                job_count += 1

            if job_count > 0:
                yield _trace("extract",
                             f"{domain} → {job_count} jobs",
                             url=url, jobs=job_count, page_type=page_type)
                yield {"type": "site", "domain": domain, "jobs_count": job_count}
            else:
                yield _trace("fetch",
                             f"{domain} → no jobs ({page_type})",
                             url=url, page_type=page_type)

        # ── Emit stats every wave ──────────────────────────────────────────
        yield _stats_event(st, queries_used)
        yield {"type": "progress",
               "message": (f"⚡ {len(st.jobs)} jobs · "
                           f"{st.pages_crawled} pages · "
                           f"{st.total_tokens:,} tokens")}

        # ── Budget check ───────────────────────────────────────────────────
        if (st.pages_crawled >= MAX_PAGES_TOTAL
                or len(st.jobs) >= MAX_JOBS_PER_SESSION):
            yield _trace("eval", "Crawl limit reached — stopping")
            break

        # ── Quality evaluation (every 2 waves) ────────────────────────────
        if queries_used % 2 == 0:
            if _LLM_AVAILABLE:
                yield _trace("eval",
                             f"Evaluating quality ({st.high_quality} high-quality jobs)…")
                should_go, new_qs, strategy = await _evaluate(st, queries_used)

                yield _trace("eval",
                             f"{'Continue' if should_go else 'Satisfied'}: {strategy[:90]}",
                             should_continue=should_go,
                             new_queries=new_qs,
                             tokens=st.total_tokens)

                if should_go and new_qs:
                    yield _trace("plan",
                                 f"Adding {len(new_qs)} refined queries",
                                 queries=new_qs)
                    pending_queries = new_qs + pending_queries
                elif not should_go:
                    break
            else:
                if st.high_quality >= MIN_QUALITY_JOBS:
                    break

    # ══════════════════════════════════════════════════════════════════════════
    # Finalize
    # ══════════════════════════════════════════════════════════════════════════

    if st.jobs:
        feedback = await get_feedback_profile(session_id)
        rank_jobs(st.jobs, profile, preferences, feedback)

    await update_session(session_id, "done", len(st.jobs), st.pages_crawled)

    yield _stats_event(st, queries_used)
    yield {
        "type":          "done",
        "jobs_found":    len(st.jobs),
        "pages_crawled": st.pages_crawled,
        "queries_used":  queries_used,
        "tokens_total":  st.total_tokens,
        "tokens_prompt": st.tok_prompt,
        "tokens_comp":   st.tok_comp,
        "high_quality":  st.high_quality,
        "sites_count":   len(st.sites),
        "llm_guided":    _LLM_AVAILABLE,
        "model":         OPENAI_MODEL if _LLM_AVAILABLE else None,
    }