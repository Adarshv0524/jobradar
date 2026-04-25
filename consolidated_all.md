# Consolidated Code: `jobradar`

## `backend/agents.py`

```python
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


async def _plan_queries(st: _State) -> Tuple[List[str], str]:
    """Phase 1: LLM generates diverse search queries."""
    skills = ", ".join(st.profile.get("skills", [])[:15]) or "not specified"
    system = (
        "You are a world-class job search strategist. "
        "Generate the most comprehensive set of queries to find REAL job postings — "
        "not aggregator pages. Focus on ATS platforms and company career pages."
    )
    user = f"""Target: {st.query}
Skills: {skills}
Location: {st.prefs.get('location') or 'any'}
Remote: {st.prefs.get('remote_preference') or 'any'}
Level: {st.prefs.get('experience_level') or 'not specified'}
Summary: {(st.profile.get('summary') or '')[:300]}

Generate 15-20 diverse queries. MUST include:
- 5+ ATS-specific (site:greenhouse.io, site:lever.co, site:ashbyhq.com, site:boards.greenhouse.io, site:jobs.lever.co)
- Role title variations (e.g. 'data engineer', 'data pipeline engineer', 'ETL engineer')
- Skill-combo queries
- Company-type variants (startup, FAANG, fintech, etc.)
- Remote/location variants
"""
    result, p, c = await _llm_call(
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        tools=[_TOOLS[0]],
        tool_choice={"type":"function","function":{"name":"generate_search_queries"}},
        max_tokens=1400,
        state=st,
    )
    if result and "queries" in result:
        all_q = result.get("queries", []) + result.get("ats_specific", [])
        qs    = [q.strip() for q in all_q if q.strip()]
        return qs[:MAX_QUERIES_PER_SESSION], result.get("reasoning", "")
    return _heuristic_queries(st.query, st.prefs), "heuristic fallback"


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

def _heuristic_queries(query: str, prefs: dict) -> List[str]:
    base = query.strip()
    loc  = prefs.get("location", "")
    qs   = [
        base,
        f"{base} remote",
        f"{base} site:boards.greenhouse.io",
        f"{base} site:jobs.lever.co",
        f"{base} site:ashbyhq.com",
        f"{base} site:workday.com",
        f"{base} site:careers.google.com",
        f"{base} \"we are hiring\"",
        f"{base} engineer jobs",
        f"{base} developer hiring 2024",
    ]
    if loc:
        qs.insert(2, f"{base} {loc}")
    return qs[:MAX_QUERIES_PER_SESSION]


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
    raw_jobs: List[dict],
    st: _State,
    infer_missing_salary: bool = False,
) -> AsyncGenerator[dict, None]:
    """Score, dedup, store jobs and yield them."""
    for raw in raw_jobs:
        if len(st.jobs) >= MAX_JOBS_PER_SESSION:
            return

        apply_url = raw.get("apply_url", "")
        if apply_url and apply_url in st.apply_urls:
            continue

        fake, _ = is_fake_job(raw.get("title", ""), raw.get("description", ""))
        if fake:
            continue

        # Skip clearly bad titles from aggregator pages
        title = raw.get("title", "")
        if any(pat in title.lower() for pat in [
            "jobs in ", "+jobs", "k jobs", "job openings",
            "careers at indeed", "jobs near", "hiring now",
        ]):
            continue

        # Infer missing salary via LLM (only for high-potential jobs)
        if infer_missing_salary and not raw.get("salary"):
            inferred = await _infer_salary(raw, st)
            if inferred:
                raw["salary"]           = inferred
                raw["salary_inferred"]  = True

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
```

## `backend/config.py`

```python
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
RESUME_DIR = DATA_DIR / "resumes"
RESUME_DIR.mkdir(exist_ok=True)

DB_PATH = str(DATA_DIR / "jobsearch.db")

# ── Search limits ─────────────────────────────────────────────────────────────
MAX_QUERIES_PER_SESSION  = int(os.environ.get("MAX_QUERIES",   "40"))
MAX_URLS_PER_QUERY       = int(os.environ.get("MAX_URLS",      "15"))
MAX_PAGES_TOTAL          = int(os.environ.get("MAX_PAGES",     "8000"))
MAX_JOBS_PER_SESSION     = int(os.environ.get("MAX_JOBS",      "3000"))
MIN_QUALITY_JOBS         = int(os.environ.get("MIN_QUALITY",   "50"))
MAX_CONCURRENT_FETCHES   = int(os.environ.get("MAX_CONCURRENT","40"))

# ── HTTP ──────────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT  = 18
MAX_RETRIES      = 3
CRAWL_DELAY      = 0.6   # seconds between requests to same domain

# ── Scoring ───────────────────────────────────────────────────────────────────
MIN_SCORE_THRESHOLD    = 0.10
HIGH_QUALITY_THRESHOLD = 0.55

# ── Playwright ────────────────────────────────────────────────────────────────
USE_PLAYWRIGHT         = os.environ.get("USE_PLAYWRIGHT", "true").lower() == "true"
PLAYWRIGHT_POOL_SIZE   = int(os.environ.get("PLAYWRIGHT_POOL", "3"))
PLAYWRIGHT_TIMEOUT     = 25_000   # ms

# ── ATS platform host fragments ───────────────────────────────────────────────
ATS_DOMAINS = {
    "greenhouse.io", "lever.co", "ashbyhq.com", "workday.com",
    "bamboohr.com", "jobvite.com", "icims.com", "smartrecruiters.com",
    "taleo.net", "successfactors.com", "recruitee.com", "personio.de",
    "breezy.hr", "pinpoint.com", "rippling.com", "dover.com",
    "workable.com", "applytojob.com", "careers.google.com",
    "myworkdayjobs.com", "hire.com", "freshteam.com",
    "jobs.lever.co", "boards.greenhouse.io", "apply.workable.com",
    "jobs.jobvite.com", "careers.icims.com",
}

# ── 60+ Job sites to crawl (key → metadata) ──────────────────────────────────
# Each entry: label, url_template ({query} is replaced), max_pages, needs_js
JOB_SITES: dict[str, dict] = {
    # ── Free JSON APIs ────────────────────────────────────────────────────────
    "remotive":      {"label": "Remotive",       "type": "api", "max_pages": 1},
    "arbeitnow":     {"label": "Arbeitnow",       "type": "api", "max_pages": 5},
    "jobicy":        {"label": "Jobicy",          "type": "api", "max_pages": 1},
    "remoteok":      {"label": "RemoteOK",        "type": "api", "max_pages": 1},
    "himalayas":     {"label": "Himalayas",       "type": "api", "max_pages": 1},
    "themuse":       {"label": "The Muse",        "type": "api", "max_pages": 5},
    "adzuna":        {"label": "Adzuna",          "type": "api", "max_pages": 10},
    "devitjobs":     {"label": "DevITjobs",       "type": "api", "max_pages": 3},

    # ── ATS Aggregators (structured, highly reliable) ─────────────────────────
    "greenhouse":    {
        "label": "Greenhouse ATS",
        "url": "https://boards.greenhouse.io/embed/job_board?for={company}",
        "search": "https://www.google.com/search?q=site:boards.greenhouse.io+{query}",
        "type": "ats", "max_pages": 30,
    },
    "lever":         {
        "label": "Lever ATS",
        "search": "https://www.google.com/search?q=site:jobs.lever.co+{query}",
        "type": "ats", "max_pages": 30,
    },
    "ashby":         {
        "label": "Ashby ATS",
        "search": "https://www.google.com/search?q=site:ashbyhq.com+{query}",
        "type": "ats", "max_pages": 20,
    },
    "workday":       {
        "label": "Workday ATS",
        "search": "https://www.google.com/search?q=site:myworkdayjobs.com+{query}",
        "type": "ats", "max_pages": 20,
    },
    "bamboohr":      {
        "label": "BambooHR ATS",
        "search": "https://www.google.com/search?q=site:bamboohr.com/jobs+{query}",
        "type": "ats", "max_pages": 15,
    },
    "icims":         {
        "label": "iCIMS ATS",
        "search": "https://www.google.com/search?q=site:careers.icims.com+{query}",
        "type": "ats", "max_pages": 15,
    },
    "smartrecruiters": {
        "label": "SmartRecruiters",
        "search": "https://www.google.com/search?q=site:careers.smartrecruiters.com+{query}",
        "type": "ats", "max_pages": 15,
    },
    "workable":      {
        "label": "Workable",
        "url": "https://apply.workable.com/api/v3/jobs?query={query}&limit=100",
        "type": "ats_api", "max_pages": 10,
    },
    "recruitee":     {
        "label": "Recruitee",
        "search": "https://www.google.com/search?q=site:recruitee.com+{query}",
        "type": "ats", "max_pages": 10,
    },
    "breezyhr":      {
        "label": "Breezy HR",
        "search": "https://www.google.com/search?q=site:breezy.hr+{query}",
        "type": "ats", "max_pages": 10,
    },
    "jobvite":       {
        "label": "Jobvite",
        "search": "https://www.google.com/search?q=site:jobs.jobvite.com+{query}",
        "type": "ats", "max_pages": 10,
    },

    # ── Open Job Boards (HTML) ────────────────────────────────────────────────
    "weworkremotely": {
        "label": "We Work Remotely",
        "url": "https://weworkremotely.com/remote-jobs/search?term={query}&page={page}",
        "type": "html", "max_pages": 10,
    },
    "startup_jobs":  {
        "label": "Startup Jobs",
        "url": "https://startup.jobs/?q={query}&page={page}",
        "type": "html", "max_pages": 10,
    },
    "ycombinator":   {
        "label": "Y Combinator",
        "url": "https://www.ycombinator.com/jobs?q={query}",
        "type": "html_js", "max_pages": 5,
    },
    "wellfound":     {
        "label": "Wellfound (AngelList)",
        "url": "https://wellfound.com/jobs?q={query}&page={page}",
        "type": "html_js", "max_pages": 20,
    },
    "otta":          {
        "label": "Otta",
        "url": "https://app.otta.com/jobs/search?query={query}",
        "type": "html_js", "max_pages": 10,
    },
    "simplify":      {
        "label": "Simplify Jobs",
        "url": "https://simplify.jobs/jobs?search={query}&page={page}",
        "type": "html_js", "max_pages": 15,
    },
    "builtin":       {
        "label": "Built In",
        "url": "https://builtin.com/jobs/remote?search={query}&page={page}",
        "type": "html_js", "max_pages": 20,
    },
    "builtin_nyc":   {
        "label": "Built In NYC",
        "url": "https://www.builtinnyc.com/jobs?search={query}&page={page}",
        "type": "html_js", "max_pages": 10,
    },
    "builtin_sf":    {
        "label": "Built In SF",
        "url": "https://www.builtinsf.com/jobs?search={query}&page={page}",
        "type": "html_js", "max_pages": 10,
    },
    "remote_co":     {
        "label": "Remote.co",
        "url": "https://remote.co/remote-jobs/search/?search_keywords={query}&page={page}",
        "type": "html", "max_pages": 10,
    },
    "remoteLeaf":    {
        "label": "RemoteLeaf",
        "url": "https://remoteleaf.com/jobs?search={query}",
        "type": "html", "max_pages": 5,
    },
    "working_nomads": {
        "label": "Working Nomads",
        "url": "https://www.workingnomads.com/jobs?category={query}&page={page}",
        "type": "html", "max_pages": 10,
    },
    "just_remote":   {
        "label": "JustRemote",
        "url": "https://justremote.co/remote-jobs?search={query}&p={page}",
        "type": "html", "max_pages": 10,
    },
    "authentic_jobs": {
        "label": "Authentic Jobs",
        "url": "https://authenticjobs.com/?search={query}&page={page}",
        "type": "html", "max_pages": 10,
    },
    "jobspresso":    {
        "label": "Jobspresso",
        "url": "https://jobspresso.co/remote-work/?s={query}&paged={page}",
        "type": "html", "max_pages": 5,
    },
    "flexjobs":      {
        "label": "FlexJobs",
        "url": "https://www.flexjobs.com/telecommuting-jobs/kw-{query}?page={page}",
        "type": "html", "max_pages": 15,
    },
    "dice":          {
        "label": "Dice",
        "url": "https://www.dice.com/jobs?q={query}&page={page}",
        "type": "html_js", "max_pages": 20,
    },
    "hired":         {
        "label": "Hired",
        "url": "https://hired.com/jobs#!?q={query}&page={page}",
        "type": "html_js", "max_pages": 10,
    },
    "arc_dev":       {
        "label": "Arc.dev",
        "url": "https://arc.dev/remote-jobs/{query}?page={page}",
        "type": "html_js", "max_pages": 10,
    },
    "contra":        {
        "label": "Contra",
        "url": "https://contra.com/opportunities?q={query}&page={page}",
        "type": "html_js", "max_pages": 10,
    },
    "braintrust":    {
        "label": "Braintrust",
        "url": "https://app.usebraintrust.com/jobs/?search={query}&page={page}",
        "type": "html_js", "max_pages": 10,
    },
    "gun_io":        {
        "label": "Gun.io",
        "url": "https://gun.io/find-work/jobs?q={query}",
        "type": "html_js", "max_pages": 5,
    },
    "toptal":        {
        "label": "Toptal",
        "url": "https://www.toptal.com/jobs/search?query={query}",
        "type": "html_js", "max_pages": 5,
    },
    "cord":          {
        "label": "Cord",
        "url": "https://cord.co/jobs?query={query}&page={page}",
        "type": "html_js", "max_pages": 10,
    },
    "lemon_io":      {
        "label": "Lemon.io",
        "url": "https://lemon.io/jobs/?query={query}",
        "type": "html", "max_pages": 5,
    },
    "underdog":      {
        "label": "Underdog.io",
        "url": "https://underdog.io/jobs?query={query}&page={page}",
        "type": "html_js", "max_pages": 5,
    },
    "pallet":        {
        "label": "Pallet",
        "url": "https://pallet.xyz/explore/jobs?q={query}&page={page}",
        "type": "html_js", "max_pages": 5,
    },
    "crunchboard":   {
        "label": "CrunchBoard",
        "url": "https://www.crunchboard.com/jobs?q={query}&page={page}",
        "type": "html", "max_pages": 10,
    },
    "joblist":       {
        "label": "JobList",
        "url": "https://joblist.app/search?q={query}&page={page}",
        "type": "html_js", "max_pages": 10,
    },
    "proxify":       {
        "label": "Proxify",
        "url": "https://proxify.io/jobs?search={query}&page={page}",
        "type": "html", "max_pages": 5,
    },
    "hn_whoishiring": {
        "label": "HN: Who's Hiring",
        "url": "https://hn.algolia.com/api/v1/search?query={query}+hiring&tags=comment",
        "type": "hn_api", "max_pages": 3,
    },
    "stackoverflow": {
        "label": "Stack Overflow Jobs",
        "url": "https://stackoverflow.com/jobs?q={query}&pg={page}",
        "type": "html", "max_pages": 20,
    },
    "eurotechjobs":  {
        "label": "EuroTech Jobs",
        "url": "https://www.eurotechjobs.com/search/?q={query}&page={page}",
        "type": "html", "max_pages": 10,
    },
    "honeypot":      {
        "label": "Honeypot",
        "url": "https://app.honeypot.io/search?q={query}&page={page}",
        "type": "html_js", "max_pages": 10,
    },
    "cwjobs":        {
        "label": "CW Jobs (UK)",
        "url": "https://www.cwjobs.co.uk/jobs/{query}/in-remote?page={page}",
        "type": "html", "max_pages": 10,
    },
    "jobserve":      {
        "label": "JobServe",
        "url": "https://www.jobserve.com/gb/en/Job-Search/?query={query}&page={page}",
        "type": "html", "max_pages": 10,
    },
    "instahyre":     {
        "label": "Instahyre (India)",
        "url": "https://www.instahyre.com/search-jobs/?q={query}&page={page}",
        "type": "html_js", "max_pages": 10,
    },
    "iimjobs":       {
        "label": "IIMJobs (India)",
        "url": "https://www.iimjobs.com/j/{query}.html?page={page}",
        "type": "html", "max_pages": 10,
    },
    "nodesk":        {
        "label": "NoDesk",
        "url": "https://nodesk.co/remote-jobs/{query}/?page={page}",
        "type": "html", "max_pages": 5,
    },
    "remotehub":     {
        "label": "RemoteHub",
        "url": "https://remotehub.io/remote-jobs?query={query}&page={page}",
        "type": "html_js", "max_pages": 10,
    },
    "jobgether":     {
        "label": "Jobgether",
        "url": "https://jobgether.com/offer?search={query}&page={page}",
        "type": "html", "max_pages": 10,
    },
    "remoteco":      {
        "label": "Remote.co",
        "url": "https://remote.co/remote-jobs/search/?search_keywords={query}",
        "type": "html", "max_pages": 10,
    },
    "career_vault":  {
        "label": "CareerVault",
        "url": "https://careervault.io/remote/{query}?page={page}",
        "type": "html", "max_pages": 10,
    },
}

# ── Sites needing Playwright (JS-heavy) ───────────────────────────────────────
JS_HEAVY_SITES = {k for k, v in JOB_SITES.items() if v.get("type") == "html_js"}

# ── Free job API endpoints (no key required) ──────────────────────────────────
FREE_JOB_APIS = {
    "remotive":  "https://remotive.com/api/remote-jobs?search={query}&limit=100",
    "arbeitnow": "https://arbeitnow.com/api/job-board-api?page={page}",
    "jobicy":    "https://jobicy.com/api/v2/remote-jobs?count=100&geo=worldwide&tag={query}",
    "remoteok":  "https://remoteok.com/api?tags={query}",
    "himalayas": "https://himalayas.app/jobs/api?q={query}&limit=100&offset={offset}",
    "themuse":   "https://www.themuse.com/api/public/jobs?page={page}&descending=true",
    "adzuna":    "https://api.adzuna.com/v1/api/jobs/{country}/search/{page}?app_id={app_id}&app_key={app_key}&results_per_page=50&what={query}",
    "devitjobs": "https://devitjobs.us/api/jobsLight",
}

# ── Open job board HTML templates ─────────────────────────────────────────────
OPEN_JOB_BOARDS = [
    "https://remotive.com/remote-jobs/search?search={query}&page={page}",
    "https://weworkremotely.com/remote-jobs/search?term={query}&page={page}",
    "https://startup.jobs/?q={query}&page={page}",
    "https://www.ycombinator.com/jobs?q={query}",
    "https://jobs.ashbyhq.com/search?query={query}",
]

# ── Known aggregator domains to skip ─────────────────────────────────────────
AGGREGATOR_DOMAINS = {
    "indeed.com", "linkedin.com", "glassdoor.com", "naukri.com",
    "timesjobs.com", "monster.com", "ziprecruiter.com", "shine.com",
    "simplyhired.com", "careerbuilder.com", "dice.com",
    "reed.co.uk", "totaljobs.com", "jobsite.co.uk",
}

# ── Spam signals ─────────────────────────────────────────────────────────────
FAKE_SIGNALS = [
    "earn from home", "work from home earn", "be your own boss",
    "multi-level", "multilevel", "mlm", "commission only",
    "unlimited earning potential", "pyramid", "get rich",
    "passive income", "no experience needed but", "make money online",
]

# ── Canonical skill list ──────────────────────────────────────────────────────
TECH_SKILLS = [
    "python", "javascript", "typescript", "java", "go", "golang", "rust",
    "c++", "c#", "ruby", "scala", "kotlin", "swift", "php",
    "react", "vue", "angular", "svelte", "nextjs", "nuxt",
    "fastapi", "django", "flask", "spring", "rails", "laravel",
    "nodejs", "express", "graphql", "rest", "grpc",
    "docker", "kubernetes", "k8s", "terraform", "ansible",
    "aws", "gcp", "azure", "cloud", "lambda", "s3",
    "postgresql", "mysql", "sqlite", "mongodb", "redis",
    "elasticsearch", "kafka", "spark", "airflow", "dbt",
    "pytorch", "tensorflow", "sklearn", "pandas", "numpy",
    "llm", "machine learning", "deep learning", "nlp", "computer vision",
    "sql", "nosql", "microservices", "devops", "sre", "ci/cd", "git",
    "linux", "bash", "data engineering", "data science", "mlops",
    "flink", "databricks", "snowflake", "bigquery", "redshift",
    "celery", "rabbitmq", "protobuf", "openapi",
    "react native", "flutter", "ios", "android",
]

# ── CORS ──────────────────────────────────────────────────────────────────────
CORS_ORIGINS = [
    "http://localhost:4321",
    "http://localhost:3000",
    "http://127.0.0.1:4321",
    "http://127.0.0.1:3000",
]

# ── User agents ───────────────────────────────────────────────────────────────
USER_AGENTS = [
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
]

# ── Adzuna (optional — get free key at developer.adzuna.com) ──────────────────
ADZUNA_APP_ID  = os.environ.get("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY = os.environ.get("ADZUNA_APP_KEY", "")

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.environ.get("OPENAI_MODEL", "gpt-4.1")
```

## `backend/main.py`

```python
"""FastAPI application — all HTTP routes and SSE streaming."""

import asyncio
import json
import os
from typing import Optional

from dotenv import load_dotenv

# Load environment variables before config usage
load_dotenv()

import logging
log = logging.getLogger(__name__)


from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from agents import run_search_session
from config import CORS_ORIGINS, RESUME_DIR
from store import (
    create_session,
    delete_session,
    get_all_sessions,
    get_jobs,
    get_session,
    init_db,
    insert_feedback,
    update_session,
)


app = FastAPI(title="JobRadar API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    await init_db()
    from config import PLAYWRIGHT_POOL_SIZE, USE_PLAYWRIGHT
    if USE_PLAYWRIGHT:
        from playwright_pool import init_playwright_pool
        await init_playwright_pool(size=PLAYWRIGHT_POOL_SIZE)
        log.info("Playwright pool ready")


@app.on_event("shutdown")
async def shutdown():
    from playwright_pool import close_playwright_pool
    await close_playwright_pool()


# ── Pydantic models ───────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query:              str
    role:               Optional[str]       = ""
    location:           Optional[str]       = ""
    remote_preference:  Optional[str]       = "any"   # remote | hybrid | on-site | any
    experience_level:   Optional[str]       = ""
    min_salary:         Optional[int]       = 0
    negatives:          Optional[list]      = []      # no_agency, no_crypto, etc.
    skills:             Optional[list]      = []
    # Profile text fields
    summary:            Optional[str]       = ""
    experience_summary: Optional[str]       = ""
    role_target:        Optional[str]       = ""


class FeedbackRequest(BaseModel):
    job_id:     str
    session_id: str
    rating:     int  # +1 or -1


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/api/search")
async def start_search(req: SearchRequest):
    """Start a new search session. Returns session_id immediately."""
    preferences = {
        "role":               req.role or req.query,
        "location":           req.location,
        "remote_preference":  req.remote_preference,
        "experience_level":   req.experience_level,
        "min_salary":         req.min_salary,
        "negatives":          req.negatives,
        "skills":             req.skills,
    }
    profile = {
        "skills":             req.skills,
        "summary":            req.summary,
        "experience_summary": req.experience_summary,
        "role_target":        req.role_target or req.role or req.query,
    }

    sid = await create_session(req.query, preferences, profile)
    return {"session_id": sid, "status": "started"}


@app.get("/api/search/{session_id}/stream")
async def stream_search(session_id: str):
    """SSE stream — yields job events as they are discovered."""
    session = await get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    async def event_gen():
        try:
            async for event in run_search_session(
                session_id,
                session["query"],
                session["preferences"],
                session["profile"],
            ):
                yield {"data": json.dumps(event)}
                await asyncio.sleep(0)  # yield control
        except asyncio.CancelledError:
            await update_session(session_id, "cancelled")
        except Exception as e:
            yield {"data": json.dumps({"type": "error", "message": str(e)})}
            await update_session(session_id, "error")

    return EventSourceResponse(event_gen())


@app.get("/api/jobs/{session_id}")
async def get_session_jobs(session_id: str):
    session = await get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    jobs = await get_jobs(session_id)
    return {"session": session, "jobs": jobs, "count": len(jobs)}


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    if req.rating not in (-1, 1):
        raise HTTPException(400, "Rating must be +1 or -1")
    await insert_feedback(req.job_id, req.session_id, req.rating)
    return {"status": "ok"}


@app.get("/api/sessions")
async def list_sessions():
    sessions = await get_all_sessions()
    return {"sessions": sessions}


@app.get("/api/sessions/{session_id}")
async def get_session_detail(session_id: str):
    s = await get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return s


@app.delete("/api/sessions/{session_id}")
async def remove_session(session_id: str):
    await delete_session(session_id)
    return {"status": "deleted"}


@app.post("/api/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """Accept a plain-text or PDF resume and return extracted profile fields."""
    content = await file.read()

    profile_text = ""
    if file.content_type == "application/pdf":
        try:
            import pdfplumber
            import io
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                profile_text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        except Exception:
            profile_text = content.decode("utf-8", errors="ignore")
    else:
        profile_text = content.decode("utf-8", errors="ignore")

    # Save to disk
    safe_name = file.filename.replace("/", "_").replace("..", "_")
    path      = RESUME_DIR / safe_name
    path.write_bytes(content)

    # Extract skills from resume text
    from scraper import extract_skills, infer_experience
    skills   = extract_skills(profile_text)
    exp_lvl  = infer_experience(profile_text)

    # Very simple summary extraction (first 500 chars of substantial content)
    lines   = [l.strip() for l in profile_text.splitlines() if len(l.strip()) > 40]
    summary = " ".join(lines[:5])[:500]

    return {
        "skills":           skills,
        "experience_level": exp_lvl,
        "summary":          summary,
        "raw_length":       len(profile_text),
    }


@app.get("/api/health")
async def health():
    from agents import _LLM_AVAILABLE, OPENAI_MODEL

    return {
        "status": "ok",
        "llm_enabled": _LLM_AVAILABLE,
        "llm_model": OPENAI_MODEL if _LLM_AVAILABLE else None,
    }
```

## `backend/playwright_pool.py`

```python
"""
Playwright browser pool for JS-heavy sites.

Usage:
    async with get_page() as page:
        await page.goto(url)
        html = await page.content()
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

log = logging.getLogger(__name__)

_pool: Optional["PlaywrightPool"] = None
_playwright_available = False

try:
    from playwright.async_api import async_playwright, Browser, Page, BrowserContext
    _playwright_available = True
except ImportError:
    log.warning("playwright not installed — JS-heavy sites will use httpx fallback")


class PlaywrightPool:
    """A fixed pool of Playwright browser pages with stealth settings."""

    def __init__(self, size: int = 3):
        self._size = size
        self._semaphore = asyncio.Semaphore(size)
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._ready = False

    async def start(self):
        if not _playwright_available:
            return
        try:
            from playwright.async_api import async_playwright
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--window-size=1920,1080",
                ],
            )
            self._ready = True
            log.info("Playwright pool started (size=%d)", self._size)
        except Exception as e:
            log.warning("Playwright failed to start: %s", e)
            self._ready = False

    async def stop(self):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    @asynccontextmanager
    async def get_page(self) -> AsyncGenerator[Optional["Page"], None]:
        if not self._ready or self._browser is None:
            yield None
            return

        async with self._semaphore:
            ctx: BrowserContext = await self._browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                ),
                locale="en-US",
                timezone_id="America/New_York",
                extra_http_headers={
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
            )
            # Stealth: hide automation markers
            await ctx.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
                Object.defineProperty(navigator, 'languages', {get: () => ['en-US','en']});
                window.chrome = {runtime: {}};
            """)
            page = await ctx.new_page()
            try:
                yield page
            finally:
                await ctx.close()


async def fetch_with_playwright(url: str, wait_selector: str = "body",
                                 timeout: int = 25_000) -> Optional[str]:
    """
    Fetch a JS-heavy URL using the global Playwright pool.
    Returns HTML string or None on failure.
    Falls back to httpx if Playwright unavailable.
    """
    global _pool
    if _pool is None or not _pool._ready:
        return None

    try:
        async with _pool.get_page() as page:
            if page is None:
                return None
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            # Wait for key selector to appear
            try:
                await page.wait_for_selector(wait_selector, timeout=5_000)
            except Exception:
                pass
            # Scroll to trigger lazy loading
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
            await asyncio.sleep(0.8)
            return await page.content()
    except Exception as e:
        log.debug("Playwright fetch failed for %s: %s", url, e)
        return None


async def init_playwright_pool(size: int = 3) -> None:
    global _pool
    if not _playwright_available:
        return
    _pool = PlaywrightPool(size=size)
    await _pool.start()


async def close_playwright_pool() -> None:
    global _pool
    if _pool:
        await _pool.stop()
        _pool = None
```

## `backend/ranker.py`

```python
"""Multi-signal job scoring, ranking, and feedback-adaptive re-ranking."""

import math
import re
from datetime import datetime, date
from typing import Dict, List

from config import HIGH_QUALITY_THRESHOLD, MIN_SCORE_THRESHOLD

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


# ── Preference matching ───────────────────────────────────────────────────────

def preference_score(job: dict, prefs: dict) -> float:
    score  = 0.0
    weight = 0

    # Remote preference
    pref_remote = prefs.get("remote_preference", "")
    job_remote  = (job.get("remote_type") or "").lower()
    if pref_remote:
        weight += 1
        if pref_remote == "remote"   and job_remote == "remote":   score += 1.0
        elif pref_remote == "hybrid" and job_remote in ("remote", "hybrid"): score += 0.8
        elif pref_remote == "on-site" and job_remote == "on-site": score += 1.0
        elif job_remote == "remote":                                score += 0.6

    # Location
    pref_loc = (prefs.get("location") or "").lower()
    job_loc  = (job.get("location")   or "").lower()
    if pref_loc and job_loc:
        weight += 1
        if pref_loc in job_loc or job_loc in pref_loc:
            score += 1.0
        elif job_remote == "remote":
            score += 0.7

    # Experience level
    pref_exp = (prefs.get("experience_level") or "").lower()
    job_exp  = (job.get("experience_level")   or "").lower()
    if pref_exp and job_exp:
        weight += 1
        if pref_exp == job_exp:
            score += 1.0
        elif abs(_exp_rank(pref_exp) - _exp_rank(job_exp)) == 1:
            score += 0.5

    # Salary
    pref_sal  = prefs.get("min_salary", 0)
    job_sal   = _extract_salary_number(job.get("salary", ""))
    if pref_sal and job_sal:
        weight += 1
        score  += 1.0 if job_sal >= pref_sal else max(0.0, job_sal / pref_sal)

    return score / weight if weight else 0.6  # neutral if no prefs set


def _exp_rank(lvl: str) -> int:
    return {"intern": 0, "junior": 1, "mid": 2, "mid-senior": 3, "senior": 4}.get(lvl, 2)


def _extract_salary_number(salary_str: str) -> int:
    nums = re.findall(r"[\d,]+", salary_str.replace("k", "000").replace("K", "000"))
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


# ── Main scoring ──────────────────────────────────────────────────────────────

def score_job(job: dict, profile: dict, prefs: dict,
              feedback_profile: dict = None) -> dict:
    """
    Returns updated job dict with 'score', 'score_breakdown',
    'match_reasons', 'reject_reasons'.
    """
    title       = job.get("title", "")
    desc        = job.get("description", "")
    job_skills  = job.get("skills", [])

    profile_text    = _build_profile_text(profile)
    user_skills     = profile.get("skills", [])

    # 1. Semantic similarity between profile and job
    semantic = text_similarity(profile_text, title + " " + desc)

    # 2. Skill overlap
    skill_sc = skill_overlap(user_skills, job_skills)

    # 3. Title match
    pref_role  = prefs.get("role", "")
    title_sc   = text_similarity(pref_role, title) if pref_role else semantic * 0.8

    # 4. Preference match
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

    # Weighted composite
    score = (
        0.30 * semantic
        + 0.25 * skill_sc
        + 0.20 * title_sc
        + 0.15 * pref_sc
        + 0.10 * rec_sc
        + src_bonus
        + fb_adj
    )
    score = max(0.0, min(1.0, score))

    # Negative filters
    negatives = prefs.get("negatives", [])
    neg_flags = check_negatives(job, negatives)

    # Build match/reject reasons
    match_reasons, reject_reasons = [], []

    if neg_flags:
        reject_reasons.extend(neg_flags)
        score = 0.0

    if skill_sc > 0.4:
        matched = [s for s in user_skills if any(s.lower() in js.lower() for js in job_skills)]
        match_reasons.append(f"Skill match: {', '.join(matched[:5])}")
    if semantic > 0.5:
        match_reasons.append(f"Strong profile alignment ({semantic:.0%})")
    if pref_sc > 0.7:
        match_reasons.append("Matches your preferences")
    if rec_sc > 0.8:
        match_reasons.append("Recently posted")
    if score < MIN_SCORE_THRESHOLD and not reject_reasons:
        reject_reasons.append("Low relevance to profile")

    breakdown = {
        "semantic": round(semantic, 3),
        "skill":    round(skill_sc, 3),
        "title":    round(title_sc, 3),
        "pref":     round(pref_sc, 3),
        "recency":  round(rec_sc, 3),
        "feedback": round(fb_adj, 3),
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
```

## `backend/requirements.txt`

```
fastapi==0.111.0
uvicorn[standard]==0.29.0
httpx[http2]==0.27.0
beautifulsoup4==4.12.3
lxml==5.2.2
duckduckgo_search==6.1.0
aiosqlite==0.20.0
sse-starlette==2.1.0
pydantic==2.7.1
python-multipart==0.0.9
scikit-learn==1.4.2
pdfplumber==0.11.0
openai>=1.40.0
python-dotenv==1.0.1

# Playwright (JS-heavy sites) — after pip install run: playwright install chromium
playwright==1.44.0

# Optional: much better semantic ranking
sentence-transformers==3.0.0
```

## `backend/scraper.py`

```python
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
```

## `backend/site_scrapers.py`

```python
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
from scraper import extract_skills, infer_experience, extract_salary_text, fetch_html, _norm
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
    return []
```

## `backend/store.py`

```python
"""SQLite persistence layer — all database interactions live here."""

import aiosqlite
import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict

from config import DB_PATH


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id           TEXT PRIMARY KEY,
                query        TEXT NOT NULL,
                preferences  TEXT NOT NULL DEFAULT '{}',
                profile      TEXT NOT NULL DEFAULT '{}',
                status       TEXT NOT NULL DEFAULT 'running',
                jobs_found   INTEGER DEFAULT 0,
                pages_crawled INTEGER DEFAULT 0,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id               TEXT PRIMARY KEY,
                session_id       TEXT NOT NULL,
                title            TEXT,
                company          TEXT,
                location         TEXT,
                remote_type      TEXT,
                experience_level TEXT,
                salary           TEXT,
                skills           TEXT DEFAULT '[]',
                description      TEXT,
                apply_url        TEXT,
                source_domain    TEXT,
                source_type      TEXT,
                posted_date      TEXT,
                score            REAL DEFAULT 0.0,
                score_breakdown  TEXT DEFAULT '{}',
                match_reasons    TEXT DEFAULT '[]',
                reject_reasons   TEXT DEFAULT '[]',
                is_fake          INTEGER DEFAULT 0,
                is_duplicate     INTEGER DEFAULT 0,
                extracted_at     TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id         TEXT PRIMARY KEY,
                job_id     TEXT NOT NULL,
                session_id TEXT NOT NULL,
                rating     INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sources (
                domain         TEXT PRIMARY KEY,
                success_count  INTEGER DEFAULT 0,
                fail_count     INTEGER DEFAULT 0,
                jobs_extracted INTEGER DEFAULT 0,
                last_seen      TEXT,
                is_blocked     INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS visited_urls (
                url        TEXT PRIMARY KEY,
                session_id TEXT,
                page_type  TEXT,
                visited_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS crawl_plan (
                id            TEXT PRIMARY KEY,
                session_id    TEXT NOT NULL,
                site_key      TEXT NOT NULL,
                site_label    TEXT NOT NULL,
                url_template  TEXT NOT NULL,
                page_num      INTEGER DEFAULT 1,
                max_pages     INTEGER DEFAULT 20,
                status        TEXT DEFAULT 'pending',
                jobs_found    INTEGER DEFAULT 0,
                pages_done    INTEGER DEFAULT 0,
                last_crawled  TEXT,
                error_msg     TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_jobs_session  ON jobs(session_id);
            CREATE INDEX IF NOT EXISTS idx_jobs_score    ON jobs(score DESC);
            CREATE INDEX IF NOT EXISTS idx_feedback_job  ON feedback(job_id);
            CREATE INDEX IF NOT EXISTS idx_plan_session  ON crawl_plan(session_id);
            CREATE INDEX IF NOT EXISTS idx_plan_status   ON crawl_plan(status);
        """)
        await db.commit()


# ── Sessions ──────────────────────────────────────────────────────────────────

async def create_session(query: str, preferences: dict, profile: dict) -> str:
    sid = str(uuid.uuid4())
    now = _now()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?)",
            (sid, query, json.dumps(preferences), json.dumps(profile),
             "running", 0, 0, now, now)
        )
        await db.commit()
    return sid


async def get_session(sid: str) -> Optional[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM sessions WHERE id=?", (sid,)) as cur:
            row = await cur.fetchone()
            if row:
                d = dict(row)
                d["preferences"] = json.loads(d["preferences"])
                d["profile"]     = json.loads(d["profile"])
                return d
    return None


async def get_all_sessions() -> List[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id,query,status,jobs_found,pages_crawled,created_at FROM sessions ORDER BY created_at DESC LIMIT 50"
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def update_session(sid: str, status: str, jobs_found: int = 0, pages_crawled: int = 0):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE sessions SET status=?, jobs_found=?, pages_crawled=?, updated_at=? WHERE id=?",
            (status, jobs_found, pages_crawled, _now(), sid)
        )
        await db.commit()


async def delete_session(sid: str):
    async with aiosqlite.connect(DB_PATH) as db:
        for tbl in ("feedback", "jobs", "visited_urls", "sessions"):
            col = "id" if tbl == "sessions" else "session_id"
            await db.execute(f"DELETE FROM {tbl} WHERE {col}=?", (sid,))
        await db.commit()


# ── Jobs ──────────────────────────────────────────────────────────────────────

async def insert_job(job: dict) -> Optional[str]:
    """Insert job; skip if apply_url already exists for this session."""
    apply_url = job.get("apply_url", "")
    sid       = job.get("session_id", "")

    async with aiosqlite.connect(DB_PATH) as db:
        # Dedup check
        if apply_url:
            async with db.execute(
                "SELECT id FROM jobs WHERE session_id=? AND apply_url=?", (sid, apply_url)
            ) as cur:
                if await cur.fetchone():
                    return None

        jid = str(uuid.uuid4())
        await db.execute(
            """INSERT INTO jobs
               (id,session_id,title,company,location,remote_type,experience_level,
                salary,skills,description,apply_url,source_domain,source_type,
                posted_date,score,score_breakdown,match_reasons,reject_reasons,
                is_fake,is_duplicate,extracted_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                jid, sid,
                job.get("title"), job.get("company"), job.get("location"),
                job.get("remote_type"), job.get("experience_level"), job.get("salary"),
                json.dumps(job.get("skills", [])),
                job.get("description"), apply_url,
                job.get("source_domain"), job.get("source_type"),
                job.get("posted_date"), job.get("score", 0.0),
                json.dumps(job.get("score_breakdown", {})),
                json.dumps(job.get("match_reasons", [])),
                json.dumps(job.get("reject_reasons", [])),
                1 if job.get("is_fake") else 0,
                1 if job.get("is_duplicate") else 0,
                _now()
            )
        )
        await db.commit()
        return jid


async def get_jobs(sid: str) -> List[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT j.*, COALESCE(AVG(f.rating),0) AS avg_feedback
            FROM jobs j
            LEFT JOIN feedback f ON j.id = f.job_id
            WHERE j.session_id=? AND j.is_duplicate=0 AND j.is_fake=0 AND j.score > 0.1
            GROUP BY j.id
            ORDER BY (j.score + COALESCE(AVG(f.rating),0)*0.15) DESC
        """, (sid,)) as cur:
            rows = []
            for r in await cur.fetchall():
                d = dict(r)
                d["skills"]          = json.loads(d.get("skills", "[]"))
                d["score_breakdown"] = json.loads(d.get("score_breakdown", "{}"))
                d["match_reasons"]   = json.loads(d.get("match_reasons", "[]"))
                d["reject_reasons"]  = json.loads(d.get("reject_reasons", "[]"))
                rows.append(d)
            return rows


# ── URL tracking ──────────────────────────────────────────────────────────────

async def url_visited(url: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT 1 FROM visited_urls WHERE url=?", (url,)) as cur:
            return await cur.fetchone() is not None


async def mark_visited(url: str, sid: str, page_type: str = "unknown"):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR IGNORE INTO visited_urls VALUES (?,?,?,?)",
            (url, sid, page_type, _now())
        )
        await db.commit()


# ── Sources ───────────────────────────────────────────────────────────────────

async def update_source(domain: str, success: bool, jobs_added: int = 0):
    async with aiosqlite.connect(DB_PATH) as db:
        s, f = (1, 0) if success else (0, 1)
        await db.execute(
            """INSERT INTO sources (domain,success_count,fail_count,jobs_extracted,last_seen)
               VALUES (?,?,?,?,?)
               ON CONFLICT(domain) DO UPDATE SET
                 success_count=success_count+?, fail_count=fail_count+?,
                 jobs_extracted=jobs_extracted+?, last_seen=?""",
            (domain, s, f, jobs_added, _now(), s, f, jobs_added, _now())
        )
        await db.commit()


# ── Feedback ──────────────────────────────────────────────────────────────────

async def insert_feedback(job_id: str, sid: str, rating: int):
    """Upsert feedback: enforce one vote per (job_id, session_id)."""
    async with aiosqlite.connect(DB_PATH) as db:
        # Deterministic ID ensures proper replacement behavior
        feedback_id = f"{job_id}:{sid}"

        await db.execute(
            "INSERT OR REPLACE INTO feedback VALUES (?,?,?,?,?)",
            (feedback_id, job_id, sid, rating, _now())
        )
        await db.commit()


async def get_feedback_profile(sid: str) -> dict:
    """Aggregate feedback signals for re-ranking."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT j.skills, j.company, f.rating
            FROM feedback f JOIN jobs j ON f.job_id=j.id
            WHERE f.session_id=?
        """, (sid,)) as cur:
            rows = await cur.fetchall()

    pos_skills: Dict[str, int] = {}
    neg_skills: Dict[str, int] = {}
    pos_companies: set          = set()
    neg_companies: set          = set()

    for r in rows:
        skills  = json.loads(r["skills"] or "[]")
        rating  = r["rating"]
        company = (r["company"] or "").lower()
        if rating > 0:
            pos_companies.add(company)
            for s in skills:
                pos_skills[s] = pos_skills.get(s, 0) + 1
        else:
            neg_companies.add(company)
            for s in skills:
                neg_skills[s] = neg_skills.get(s, 0) + 1

    return {
        "positive_skills":    pos_skills,
        "negative_skills":    neg_skills,
        "positive_companies": pos_companies,
        "negative_companies": neg_companies,
    }



# ── Crawl Plan ────────────────────────────────────────────────────────────────

async def insert_crawl_plan_entry(
    session_id: str, site_key: str, site_label: str,
    url_template: str, max_pages: int = 20
) -> str:
    import uuid as _uuid
    pid = str(_uuid.uuid4())
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT OR IGNORE INTO crawl_plan
               (id, session_id, site_key, site_label, url_template, max_pages, status)
               VALUES (?,?,?,?,?,?,'pending')""",
            (pid, session_id, site_key, site_label, url_template, max_pages)
        )
        await db.commit()
    return pid


async def update_crawl_plan(
    session_id: str, site_key: str, *,
    status: str, jobs_found: int = 0, pages_done: int = 0, error_msg: str = ""
):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE crawl_plan
               SET status=?, jobs_found=jobs_found+?, pages_done=pages_done+?,
                   last_crawled=?, error_msg=?
               WHERE session_id=? AND site_key=?""",
            (status, jobs_found, pages_done, _now(), error_msg or "", session_id, site_key)
        )
        await db.commit()


async def get_crawl_plan(session_id: str) -> List[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM crawl_plan WHERE session_id=? ORDER BY site_key",
            (session_id,)
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

def _now() -> str:
    return datetime.utcnow().isoformat()
```

## `frontend/ .env`

```dotenv
# Backend
PORT=8000
HOST=0.0.0.0

# Frontend — points to backend
PUBLIC_API_URL=http://localhost:8000

# Optional: sentence-transformers model cache dir
SENTENCE_TRANSFORMERS_HOME=~/.cache/torch/sentence_transformers
```

## `frontend/astro.config.mjs`

```javascript
import { defineConfig } from "astro/config";
import svelte           from "@astrojs/svelte";
import tailwind         from "@astrojs/tailwind";

export default defineConfig({
  integrations: [svelte(), tailwind()],
  server: { port: 4321 },
  vite: {
    define: {
      "import.meta.env.PUBLIC_API_URL":
        JSON.stringify(process.env.PUBLIC_API_URL || "http://localhost:8000"),
    },
  },
});
```

## `frontend/package.json`

```json
{
  "name": "jobradar-frontend",
  "type": "module",
  "version": "1.0.0",
  "scripts": {
    "dev":   "astro dev",
    "build": "astro build",
    "preview": "astro preview"
  },
  "dependencies": {
    "astro":           "^4.8.0",
    "@astrojs/svelte": "^5.4.0",
    "@astrojs/tailwind": "^5.1.0",
    "svelte":          "^4.2.17",
    "tailwindcss":     "^3.4.3"
  }
}
```

## `frontend/src/components/AppShell.svelte`

```svelte
<script lang="ts">
  import SearchForm   from "./SearchForm.svelte";
  import JobBoard     from "./JobBoard.svelte";
  import SiteSidebar  from "./SiteSidebar.svelte";
  import type { SearchPayload } from "../lib/api";

  let boardRef: JobBoard;
  let sidebarRef: SiteSidebar;
  let pendingPayload: SearchPayload | null = null;
  let isSearching = false;

  function handleSearch(e: CustomEvent<SearchPayload>) {
    pendingPayload = e.detail;
  }

  function handleSearchStart() { isSearching = true; }
  function handleSearchDone()  { isSearching = false; }

  $: if (pendingPayload && boardRef) {
    boardRef.startNewSearch(pendingPayload);
    pendingPayload = null;
  }
</script>

<div style="display:flex;gap:20px;align-items:flex-start;">

  <!-- ── Left Sidebar ─────────────────────────────────────── -->
  <aside style="
    width: 280px;
    flex-shrink: 0;
    position: sticky;
    top: 72px;
    max-height: calc(100vh - 92px);
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
  ">
    <SiteSidebar bind:this={sidebarRef} {isSearching} />
  </aside>

  <!-- ── Main Content ─────────────────────────────────────── -->
  <main style="flex:1;min-width:0;display:flex;flex-direction:column;gap:16px;">
    <SearchForm on:search={handleSearch} />
    <JobBoard
      bind:this={boardRef}
      on:searchStart={handleSearchStart}
      on:searchDone={handleSearchDone}
      on:planUpdate={(e) => sidebarRef?.handlePlanUpdate(e.detail)}
      on:trace={(e) => sidebarRef?.addTrace(e.detail)}
      on:siteFound={(e) => sidebarRef?.markSiteFound(e.detail)}
    />
  </main>

</div>
```

## `frontend/src/components/JobBoard.svelte`

```svelte
<script lang="ts">
  import JobCard from "./JobCard.svelte";
  import { createEventDispatcher } from "svelte";
  import {
    createSSEStream, getSessionJobs, submitFeedback,
    type Job, type SearchPayload,
  } from "../lib/api";
  import { startSearch } from "../lib/api";

  const dispatch = createEventDispatcher();

  export let payload: SearchPayload | null = null;

  let sessionId    = "";
  let jobs: Job[]  = [];
  let status: "idle" | "searching" | "done" | "error" = "idle";
  let statusMsg    = "";
  let jobsFound    = 0;
  let pagesCrawled = 0;
  let tokensTotal   = 0;
  let highQuality  = 0;
  let sitesCount   = 0;
  let traces: { icon: string; message: string; ts: number }[] = [];
  let sitesDiscovered: Record<string, number> = {};
  let es: EventSource | null = null;

  // Filters
  let filterRemote    = "all";
  let filterLevel     = "all";
  let sortBy          = "score";
  let searchTerm      = "";

  $: filteredJobs = jobs
    .filter(j => {
      if (filterRemote !== "all" && j.remote_type?.toLowerCase() !== filterRemote) return false;
      if (filterLevel  !== "all" && j.experience_level?.toLowerCase() !== filterLevel)  return false;
      if (searchTerm && !`${j.title} ${j.company}`.toLowerCase().includes(searchTerm.toLowerCase())) return false;
      return true;
    })
    .sort((a, b) => {
      if (sortBy === "score")   return b.score - a.score;
      if (sortBy === "recent") {
        const da = a.posted_date || "", db = b.posted_date || "";
        return db.localeCompare(da);
      }
      if (sortBy === "company") return (a.company || "").localeCompare(b.company || "");
      return 0;
    });

  export async function startNewSearch(p: SearchPayload) {
    // Cancel previous stream
    if (es) { es.close(); es = null; }

    jobs       = [];
    status     = "searching";
    dispatch("searchStart");
    statusMsg  = "Starting search…";
    jobsFound  = 0;
    pagesCrawled = 0;
    tokensTotal  = 0;
    highQuality = 0;
    sitesCount  = 0;
    traces      = [];
    sitesDiscovered = {};

    try {
      const res = await startSearch(p);
      sessionId = res.session_id;

      es = createSSEStream(
        sessionId,
        (job) => {
          jobs = [...jobs, job];
          jobsFound = jobs.length;
        },
        (msg) => { statusMsg = msg; },
        (stats) => {
          status     = "done";
          dispatch("searchDone");
          statusMsg  = `Found ${stats.jobs_found} jobs across ${stats.pages_crawled} pages`;
          jobsFound  = stats.jobs_found   ?? jobsFound;
          pagesCrawled = stats.pages_crawled ?? pagesCrawled;
        },
        (err) => {
          status    = "error";
          statusMsg = err;
        },
        (trace) => {
          if (trace.data) {
            traces = [...traces, { icon: trace.icon || "·", message: trace.message || "", ts: Date.now() }];
            traces = traces.slice(-20);
          }
        },
        (stats) => {
          jobsFound    = stats.jobs_found ?? jobsFound;
          pagesCrawled = stats.pages_crawled ?? pagesCrawled;
          tokensTotal  = stats.tokens_total ?? tokensTotal;
          highQuality = stats.high_quality ?? highQuality;
          sitesCount  = stats.sites_count ?? sitesCount;
        },
        (site) => {
          if (site.domain && site.jobs_count) {
            sitesDiscovered[site.domain] = (sitesDiscovered[site.domain] || 0) + site.jobs_count;
            sitesDiscovered = sitesDiscovered;
          }
        },
      );
    } catch (err: any) {
      status    = "error";
      statusMsg = err.message;
    }
  }

  async function handleFeedback(jobId: string, rating: 1 | -1) {
    if (!sessionId) return;
    try {
      await submitFeedback(jobId, sessionId, rating);
    } catch { /* silent */ }
  }

  // Watch payload prop for external triggers
  $: if (payload) startNewSearch(payload);
</script>

<div class="space-y-4">

  <!-- ── Status bar ─────────────────────────────────── -->
  {#if status !== "idle"}
    <div class="bg-card border border-border rounded-lg px-4 py-3">
      <div class="flex items-center justify-between flex-wrap gap-2">
        <div class="flex items-center gap-3">
          {#if status === "searching"}
            <span class="pulse-dot"></span>
          {:else if status === "done"}
            <span class="text-emerald text-lg">✓</span>
          {:else if status === "error"}
            <span class="text-rose text-lg">✕</span>
          {/if}
          <span class="text-sm font-500 text-text">{statusMsg}</span>
        </div>
        <div class="flex items-center gap-4">
          {#each [["Jobs", jobsFound], ["Pages", pagesCrawled], ["Sites", sitesCount], ["★ Quality", highQuality]] as [label, val]}
            <div class="text-center">
              <div class="text-base font-700 text-text font-mono">{val}</div>
              <div class="text-[10px] text-muted">{label}</div>
            </div>
          {/each}
        </div>
      </div>
    </div>
  {/if}

  <!-- Filter + sort bar -->
  {#if jobs.length > 0}
    <div class="flex flex-wrap gap-2 items-center pb-2 border-b border-border">
      <input
        bind:value={searchTerm}
        placeholder="Filter results…"
        class="bg-surface border border-border rounded px-3 py-1.5 text-xs font-mono
               text-text placeholder:text-muted focus:outline-none focus:border-amber/40 w-44"
      />

      <select
        bind:value={filterRemote}
        class="bg-surface border border-border rounded px-2 py-1.5 text-xs font-mono text-dim
               focus:outline-none"
      >
        <option value="all">All modes</option>
        <option value="remote">Remote</option>
        <option value="hybrid">Hybrid</option>
        <option value="on-site">On-site</option>
      </select>

      <select
        bind:value={filterLevel}
        class="bg-surface border border-border rounded px-2 py-1.5 text-xs font-mono text-dim
               focus:outline-none"
      >
        <option value="all">All levels</option>
        <option value="intern">Intern</option>
        <option value="junior">Junior</option>
        <option value="mid">Mid</option>
        <option value="mid-senior">Mid-Senior</option>
        <option value="senior">Senior</option>
      </select>

      <div class="ml-auto flex gap-1">
        {#each [["score","Relevance"],["recent","Recent"],["company","Company"]] as [val, label]}
          <button
            on:click={() => sortBy = val}
            class="text-xs font-mono px-2.5 py-1 rounded border transition-colors {
              sortBy === val
                ? 'border-amber/50 text-amber bg-amber/10'
                : 'border-border text-muted hover:border-dim'
            }"
          >
            {label}
          </button>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Job list -->
  {#if filteredJobs.length > 0}
    <div class="space-y-3">
      {#each filteredJobs as job, i (job.id)}
        <JobCard
          {job}
          index={i}
          on:feedback={(e) => handleFeedback(job.id, e.detail.rating)}
        />
      {/each}
    </div>
  {:else if status === "idle"}
    <!-- Empty state -->
    <div class="text-center py-20 space-y-3">
      <div class="text-4xl">⌖</div>
      <p class="font-display text-2xl font-700 text-dim tracking-wide">START SEARCHING</p>
      <p class="font-mono text-xs text-muted max-w-sm mx-auto">
        Enter a query above. JobRadar will search across job boards, ATS pages,
        and company career sites — ranking by fit, not just keywords.
      </p>
    </div>
  {:else if status === "searching" && jobs.length === 0}
    <div class="space-y-2 pt-4">
      {#each Array(4) as _, i}
        <div
          class="bg-card border border-border rounded-lg p-4 animate-pulse"
          style="animation-delay: {i * 150}ms"
        >
          <div class="flex gap-3">
            <div class="flex-1 space-y-2">
              <div class="h-3 bg-border rounded w-24"></div>
              <div class="h-5 bg-border rounded w-64"></div>
              <div class="h-3 bg-border rounded w-48"></div>
              <div class="flex gap-1 mt-2">
                {#each Array(4) as _}
                  <div class="h-4 bg-border rounded w-16"></div>
                {/each}
              </div>
            </div>
            <div class="flex flex-col gap-2">
              <div class="h-8 w-12 bg-border rounded"></div>
              <div class="h-7 w-20 bg-border rounded"></div>
            </div>
          </div>
        </div>
      {/each}
    </div>
  {:else if status === "done" && filteredJobs.length === 0}
    <div class="text-center py-16">
      <p class="font-display text-xl text-dim">No results match current filters</p>
      <button
        on:click={() => { filterRemote="all"; filterLevel="all"; searchTerm=""; }}
        class="mt-2 font-mono text-xs text-amber hover:underline"
      >
        Clear filters
      </button>
    </div>
  {/if}

</div>
```

## `frontend/src/components/JobCard.svelte`

```svelte
<script lang="ts">
  import { submitFeedback, type Job } from "../lib/api";

  export let job:       Job;
  export let index:     number = 0;
  export let sessionId: string = "";

  let expanded  = false;
  let myRating: 1 | -1 | 0 = 0;

  async function rate(r: 1 | -1) {
    if (!sessionId || !job.id) return;
    myRating = myRating === r ? 0 : r;
    if (myRating !== 0) await submitFeedback(job.id, sessionId, r);
  }

  $: scorePercent = Math.round((job.score ?? 0) * 100);
  $: scoreColor =
    scorePercent >= 70 ? "var(--green)" :
    scorePercent >= 45 ? "var(--accent)" :
                         "var(--text-lo)";

  $: remoteLabel = {
    remote:   "Remote",
    hybrid:   "Hybrid",
    "on-site":"On-site",
  }[job.remote_type?.toLowerCase() ?? ""] ?? job.remote_type ?? "";

  $: remoteBadgeClass = {
    remote:   "badge-green",
    hybrid:   "badge-blue",
    "on-site": "badge-neutral",
  }[job.remote_type?.toLowerCase() ?? ""] ?? "badge-neutral";

  function fmt(d: string): string {
    if (!d) return "";
    try { return new Date(d).toLocaleDateString("en-US", { month: "short", day: "numeric" }); }
    catch { return d; }
  }

  $: sourceLabel =
    job.source_type === "jsonld"       ? "structured" :
    job.source_type === "detail_page"  ? "detail page" :
    job.source_type?.startsWith("api") ? "API" : "scraped";
</script>

<article
  class="card card-hover fade-in"
  style="
    padding: 16px;
    position: relative;
    animation-delay: {Math.min(index * 40, 400)}ms;
  "
>
  <!-- Score bar (top edge) -->
  <div style="
    position:absolute;top:0;left:0;right:0;height:2px;
    border-radius:8px 8px 0 0;overflow:hidden;
  ">
    <div style="
      height:100%;
      width:{scorePercent}%;
      background:{scoreColor};
      border-radius:inherit;
      transition:width 0.7s;
    "/>
  </div>

  <!-- ── Main row ─────────────────────────────────── -->
  <div style="display:flex;gap:14px;align-items:flex-start;">

    <!-- Left: info -->
    <div style="flex:1;min-width:0;">

      <!-- Meta line -->
      <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:6px;">
        <span class="mono" style="font-size:10px;color:var(--text-dim);">#{String(index+1).padStart(2,"0")}</span>
        <span class="badge {remoteBadgeClass}">{remoteLabel}</span>
        {#if job.posted_date}
          <span class="mono" style="font-size:10px;color:var(--text-dim);">{fmt(job.posted_date)}</span>
        {/if}
        <span class="mono" style="font-size:10px;color:var(--text-dim);margin-left:auto;">{sourceLabel}</span>
      </div>

      <!-- Title -->
      <h3 style="
        font-size:15px;font-weight:600;color:var(--text-hi);
        margin:0 0 4px;line-height:1.3;
        overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
      ">{job.title || "Untitled Role"}</h3>

      <!-- Company / location / salary -->
      <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:10px;">
        <span style="font-size:13px;font-weight:500;color:var(--text);">{job.company || "Unknown"}</span>
        {#if job.location}
          <span style="color:var(--text-dim);">·</span>
          <span style="font-size:12px;color:var(--text-lo);">{job.location}</span>
        {/if}
        {#if job.salary}
          <span style="color:var(--text-dim);">·</span>
          <span class="badge badge-accent">{job.salary}</span>
        {/if}
      </div>

      <!-- Skills -->
      {#if job.skills?.length}
        <div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px;">
          {#each job.skills.slice(0,8) as skill}
            <span class="skill-tag">{skill}</span>
          {/each}
          {#if job.skills.length > 8}
            <span class="badge badge-neutral">+{job.skills.length - 8}</span>
          {/if}
        </div>
      {/if}

      <!-- Match reason -->
      {#if job.match_reasons?.length && !expanded}
        <p style="font-size:11px;color:var(--green);font-family:'JetBrains Mono',monospace;">
          ✓ {job.match_reasons[0]}
        </p>
      {/if}
    </div>

    <!-- Right: score + actions -->
    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:8px;flex-shrink:0;">

      <!-- Score pill -->
      <div class="mono" style="
        font-size:13px;font-weight:600;
        padding:4px 10px;border-radius:5px;
        border:1px solid {scoreColor}40;
        background:{scoreColor}12;
        color:{scoreColor};
      ">{scorePercent}%</div>

      <!-- Feedback buttons -->
      <div style="display:flex;gap:4px;">
        <button
          on:click={() => rate(1)}
          style="
            width:28px;height:28px;border-radius:5px;cursor:pointer;
            border:1px solid {myRating===1 ? 'var(--green)' : 'var(--border)'};
            background:{myRating===1 ? 'var(--green-lo)' : 'transparent'};
            color:{myRating===1 ? 'var(--green)' : 'var(--text-dim)'};
            display:flex;align-items:center;justify-content:center;
            font-size:14px;transition:all 0.15s;
          "
          title="Good match"
        >↑</button>
        <button
          on:click={() => rate(-1)}
          style="
            width:28px;height:28px;border-radius:5px;cursor:pointer;
            border:1px solid {myRating===-1 ? 'var(--red)' : 'var(--border)'};
            background:{myRating===-1 ? 'var(--red-lo)' : 'transparent'};
            color:{myRating===-1 ? 'var(--red)' : 'var(--text-dim)'};
            display:flex;align-items:center;justify-content:center;
            font-size:14px;transition:all 0.15s;
          "
          title="Not relevant"
        >↓</button>
      </div>

      <!-- Apply -->
      <a
        href={job.apply_url} target="_blank" rel="noopener noreferrer"
        style="
          font-size:11px;font-weight:600;font-family:'JetBrains Mono',monospace;
          padding:5px 12px;border-radius:5px;
          border:1px solid var(--accent);
          background:var(--accent-lo);
          color:var(--accent);
          text-decoration:none;
          transition:all 0.15s;letter-spacing:0.05em;
        "
        on:mouseover={e => e.currentTarget.style.background='var(--accent-hi)'}
        on:mouseout={e => e.currentTarget.style.background='var(--accent-lo)'}
      >APPLY →</a>
    </div>
  </div>

  <!-- Expand toggle -->
  <button
    on:click={() => expanded = !expanded}
    style="
      margin-top:8px;font-size:11px;font-family:'JetBrains Mono',monospace;
      color:var(--text-dim);background:none;border:none;cursor:pointer;
      padding:0;transition:color 0.15s;
    "
    on:mouseover={e => e.currentTarget.style.color='var(--text-lo)'}
    on:mouseout={e => e.currentTarget.style.color='var(--text-dim)'}
  >{expanded ? "▲ hide details" : "▼ show details"}</button>

  <!-- Expanded panel -->
  {#if expanded}
    <div class="fade-in" style="margin-top:12px;padding-top:12px;border-top:1px solid var(--border);">

      <!-- Score breakdown -->
      {#if job.score_breakdown && Object.keys(job.score_breakdown).length}
        <p style="font-size:10px;font-weight:600;color:var(--text-dim);letter-spacing:0.07em;margin-bottom:8px;">SIGNAL BREAKDOWN</p>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:12px;">
          {#each Object.entries(job.score_breakdown) as [key, val]}
            {@const n = Number(val)}
            <div class="surface" style="padding:8px;text-align:center;">
              <p style="font-size:10px;color:var(--text-dim);letter-spacing:0.06em;text-transform:uppercase;margin:0 0 2px;">{key}</p>
              <p class="mono" style="font-size:14px;font-weight:600;margin:0;color:{n>0.5?'var(--green)':n>0.25?'var(--accent)':'var(--text-dim)'}">
                {Math.round(n*100)}%
              </p>
            </div>
          {/each}
        </div>
      {/if}

      <!-- Match reasons -->
      {#if job.match_reasons?.length}
        <p style="font-size:10px;font-weight:600;color:var(--text-dim);letter-spacing:0.07em;margin-bottom:6px;">WHY IT MATCHED</p>
        <div style="margin-bottom:12px;">
          {#each job.match_reasons as r}
            <p style="font-size:11px;font-family:'JetBrains Mono',monospace;color:var(--green);margin:2px 0;">✓ {r}</p>
          {/each}
        </div>
      {/if}

      <!-- Description -->
      {#if job.description}
        <p style="font-size:10px;font-weight:600;color:var(--text-dim);letter-spacing:0.07em;margin-bottom:6px;">DESCRIPTION</p>
        <p style="font-size:12px;color:var(--text-lo);line-height:1.6;margin:0 0 12px;">
          {job.description.slice(0,700)}{job.description.length>700?"…":""}
        </p>
      {/if}

      <div style="display:flex;gap:12px;font-size:11px;font-family:'JetBrains Mono',monospace;color:var(--text-dim);">
        <span>src: {job.source_domain}</span>
        {#if job.experience_level}
          <span>· {job.experience_level}</span>
        {/if}
      </div>
    </div>
  {/if}
</article>
```

## `frontend/src/components/SearchForm.svelte`

```svelte
<script lang="ts">
  import { createEventDispatcher } from "svelte";
  import { uploadResume, type SearchPayload } from "../lib/api";

  const dispatch = createEventDispatcher<{ search: SearchPayload }>();

  let query            = "";
  let role             = "";
  let location         = "";
  let remotePreference = "any";
  let experienceLevel  = "";
  let minSalary        = "";
  let skills           = "";
  let summary          = "";
  let negatives: string[] = [];
  let uploading        = false;
  let uploadMsg        = "";
  let showAdvanced     = false;

  const NEGATIVE_OPTIONS = [
    { key: "no_agency",      label: "No agencies" },
    { key: "no_crypto",      label: "No crypto/web3" },
    { key: "no_sales",       label: "No sales roles" },
    { key: "no_unpaid",      label: "No unpaid work" },
    { key: "no_relocation",  label: "No relocation req." },
  ];

  function toggleNeg(key: string) {
    negatives = negatives.includes(key)
      ? negatives.filter(k => k !== key)
      : [...negatives, key];
  }

  async function handleResumeUpload(e: Event) {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    uploading = true; uploadMsg = "";
    try {
      const r = await uploadResume(file);
      skills  = r.skills.join(", ");
      summary = r.summary;
      if (r.experience_level) experienceLevel = r.experience_level;
      uploadMsg = `✓ ${r.skills.length} skills extracted`;
    } catch {
      uploadMsg = "⚠ Upload failed";
    } finally { uploading = false; }
  }

  function handleSubmit() {
    if (!query.trim()) return;
    dispatch("search", {
      query,
      role:               role || query,
      location,
      remote_preference:  remotePreference,
      experience_level:   experienceLevel,
      min_salary:         minSalary ? parseInt(minSalary) * 1000 : 0,
      negatives,
      skills:             skills.split(",").map(s => s.trim()).filter(Boolean),
      summary,
      experience_summary: "",
      role_target:        role,
    });
  }
</script>

<div class="card" style="padding:20px;">

  <!-- Search bar row -->
  <div style="display:flex;gap:8px;margin-bottom:16px;">
    <input
      class="input"
      bind:value={query}
      placeholder="e.g. Senior Python Engineer, Data Scientist, DevOps Lead…"
      on:keydown={e => e.key === "Enter" && handleSubmit()}
      style="flex:1;"
    />
    <button class="btn btn-primary" on:click={handleSubmit} disabled={!query.trim()}>
      Search
    </button>
  </div>

  <!-- Quick filters row -->
  <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">

    <select class="input" bind:value={remotePreference} style="width:auto;padding-right:28px;">
      <option value="any">Any work type</option>
      <option value="remote">Remote only</option>
      <option value="hybrid">Hybrid</option>
      <option value="on-site">On-site</option>
    </select>

    <select class="input" bind:value={experienceLevel} style="width:auto;padding-right:28px;">
      <option value="">Any level</option>
      <option value="junior">Junior</option>
      <option value="mid">Mid-level</option>
      <option value="senior">Senior</option>
      <option value="lead">Lead / Staff</option>
    </select>

    <input
      class="input"
      bind:value={location}
      placeholder="Location (optional)"
      style="width:160px;"
    />

    <button
      class="btn btn-ghost"
      on:click={() => showAdvanced = !showAdvanced}
      style="font-size:12px;"
    >
      {showAdvanced ? "▲" : "▼"} Advanced
    </button>
  </div>

  <!-- Advanced panel -->
  {#if showAdvanced}
    <div class="fade-in" style="margin-top:16px;padding-top:16px;border-top:1px solid var(--border);">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px;">

        <div>
          <label class="label">Min Salary (K USD)</label>
          <input class="input" type="number" bind:value={minSalary} placeholder="e.g. 80" />
        </div>

        <div>
          <label class="label">Skills (comma-separated)</label>
          <input class="input" bind:value={skills} placeholder="python, react, aws…" />
        </div>
      </div>

      <div style="margin-bottom:12px;">
        <label class="label">Profile Summary (improves ranking)</label>
        <textarea
          class="input" bind:value={summary} rows="2"
          placeholder="Brief description of your background and goals…"
          style="resize:vertical;"
        ></textarea>
      </div>

      <!-- Resume upload -->
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
        <label class="btn btn-ghost" style="cursor:pointer;font-size:12px;">
          {uploading ? "Uploading…" : "↑ Upload Resume (PDF / TXT)"}
          <input
            type="file" accept=".pdf,.txt" style="display:none;"
            on:change={handleResumeUpload} disabled={uploading}
          />
        </label>
        {#if uploadMsg}
          <span style="font-size:12px;color:var(--green);font-family:'JetBrains Mono',monospace;">
            {uploadMsg}
          </span>
        {/if}
      </div>

      <!-- Negatives -->
      <div>
        <label class="label">Exclude</label>
        <div style="display:flex;flex-wrap:wrap;gap:6px;">
          {#each NEGATIVE_OPTIONS as opt}
            <button
              on:click={() => toggleNeg(opt.key)}
              style="
                padding:4px 10px;border-radius:5px;font-size:11px;font-weight:500;
                cursor:pointer;transition:all 0.15s;border:1px solid;
                background:{negatives.includes(opt.key) ? 'var(--red-lo)' : 'transparent'};
                border-color:{negatives.includes(opt.key) ? 'rgba(248,113,113,0.3)' : 'var(--border)'};
                color:{negatives.includes(opt.key) ? 'var(--red)' : 'var(--text-lo)'};
              "
            >{opt.label}</button>
          {/each}
        </div>
      </div>
    </div>
  {/if}
</div>
```

## `frontend/src/components/SiteSidebar.svelte`

```svelte
<script lang="ts">
  export let isSearching = false;

  type SiteStatus = "pending" | "running" | "done" | "failed" | "skipped";
  type SitePlan = {
    site_key: string;
    site_label: string;
    status: SiteStatus;
    jobs_found: number;
    pages_done: number;
    pages_total: number;
  };

  type TraceEntry = {
    level: string;
    icon: string;
    message: string;
    ts: number;
  };

  let sites: Map<string, SitePlan>  = new Map();
  let traces: TraceEntry[]          = [];
  let activeTab: "sites" | "reasoning" = "sites";
  let totalJobs = 0;
  let totalSites = 0;

  export function handlePlanUpdate(data: SitePlan) {
    sites = new Map(sites.set(data.site_key, data));
    if (data.status === "done" && data.jobs_found > 0) {
      totalJobs += data.jobs_found;
    }
    totalSites = [...sites.values()].filter(s => s.status === "done" || s.status === "running").length;
  }

  export function addTrace(trace: TraceEntry) {
    traces = [trace, ...traces].slice(0, 60);
  }

  export function markSiteFound(data: { domain: string; jobs_count: number }) {
    // Legacy support for "site" events
  }

  $: runningCount = [...sites.values()].filter(s => s.status === "running").length;
  $: doneCount    = [...sites.values()].filter(s => s.status === "done").length;
  $: failedCount  = [...sites.values()].filter(s => s.status === "failed").length;

  function statusIcon(s: SiteStatus): string {
    return { pending: "○", running: "◌", done: "●", failed: "✕", skipped: "–" }[s] ?? "○";
  }
  function statusColor(s: SiteStatus): string {
    return {
      pending: "var(--text-dim)",
      running: "var(--accent)",
      done:    "var(--green)",
      failed:  "var(--red)",
      skipped: "var(--text-dim)",
    }[s] ?? "var(--text-dim)";
  }

  function traceColor(level: string): string {
    return {
      plan: "var(--accent)", search: "var(--blue)", fetch: "var(--text-lo)",
      extract: "var(--green)", api: "var(--blue)", llm: "#a78bfa",
      eval: "var(--accent)", warn: "var(--red)", crawl: "var(--text-lo)",
      done: "var(--green)",
    }[level] ?? "var(--text-lo)";
  }
</script>

<!-- ── Header ─────────────────────────────────────── -->
<div class="card" style="padding:14px;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
    <span style="font-size:12px;font-weight:600;color:var(--text-hi);letter-spacing:0.05em;">CRAWL STATUS</span>
    {#if isSearching}
      <div style="display:flex;align-items:center;gap:5px;">
        <div class="pulse-dot"></div>
        <span style="font-size:10px;font-family:'JetBrains Mono',monospace;color:var(--green);">LIVE</span>
      </div>
    {/if}
  </div>

  <!-- Stats row -->
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:14px;">
    {#each [
      {label:"Done",   val: doneCount,   color: "var(--green)"},
      {label:"Active", val: runningCount, color: "var(--accent)"},
      {label:"Failed", val: failedCount,  color: "var(--red)"},
    ] as stat}
      <div style="
        background:var(--surface);border:1px solid var(--border);
        border-radius:6px;padding:8px;text-align:center;
      ">
        <div style="font-size:18px;font-weight:700;color:{stat.color};font-family:'JetBrains Mono',monospace;">{stat.val}</div>
        <div style="font-size:10px;color:var(--text-dim);margin-top:2px;">{stat.label}</div>
      </div>
    {/each}
  </div>

  <!-- Tabs -->
  <div style="display:flex;gap:4px;margin-bottom:12px;">
    {#each [["sites","Sites"],["reasoning","Reasoning"]] as [tab, label]}
      <button
        on:click={() => activeTab = tab}
        style="
          flex:1; padding:5px 8px; border-radius:5px; font-size:11px; font-weight:500;
          cursor:pointer; transition:all 0.15s; border:1px solid;
          background:{activeTab===tab ? 'var(--elevated)' : 'transparent'};
          border-color:{activeTab===tab ? 'var(--border-hi)' : 'transparent'};
          color:{activeTab===tab ? 'var(--text)' : 'var(--text-dim)'};
        "
      >{label}</button>
    {/each}
  </div>

  <!-- ── Sites tab ──────────────────────────────────── -->
  {#if activeTab === "sites"}
    <div style="display:flex;flex-direction:column;gap:3px;max-height:420px;overflow-y:auto;">
      {#if sites.size === 0}
        <p style="font-size:12px;color:var(--text-dim);text-align:center;padding:20px 0;">
          {isSearching ? "Building crawl plan…" : "Start a search to see sites"}
        </p>
      {:else}
        {#each [...sites.values()].sort((a,b) => {
          const order = {running:0,done:1,pending:2,failed:3,skipped:4};
          return (order[a.status]??5) - (order[b.status]??5);
        }) as site (site.site_key)}
          <div class="slide-in" style="
            display:flex;align-items:center;gap:8px;
            padding:5px 6px; border-radius:5px;
            background:{site.status==='running' ? 'rgba(245,158,11,0.06)' : 'transparent'};
            border:1px solid {site.status==='running' ? 'rgba(245,158,11,0.15)' : 'transparent'};
            transition:all 0.2s;
          ">
            <span style="
              font-size:10px; width:10px; flex-shrink:0;
              color:{statusColor(site.status)};
              font-family:'JetBrains Mono',monospace;
            ">{statusIcon(site.status)}</span>
            <span style="flex:1;font-size:11px;color:{site.status==='pending' ? 'var(--text-dim)' : 'var(--text)' };min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
              {site.site_label}
            </span>
            {#if site.status === "done" && site.jobs_found > 0}
              <span style="font-size:10px;font-family:'JetBrains Mono',monospace;color:var(--green);flex-shrink:0;">
                +{site.jobs_found}
              </span>
            {:else if site.status === "running"}
              <span style="font-size:10px;font-family:'JetBrains Mono',monospace;color:var(--accent);flex-shrink:0;">
                p.{site.pages_done}
              </span>
            {/if}
          </div>
        {/each}
      {/if}
    </div>

  <!-- ── Reasoning tab ──────────────────────────────── -->
  {:else}
    <div style="display:flex;flex-direction:column;gap:4px;max-height:420px;overflow-y:auto;">
      {#if traces.length === 0}
        <p style="font-size:12px;color:var(--text-dim);text-align:center;padding:20px 0;">
          {isSearching ? "Agent is thinking…" : "Agent reasoning will appear here"}
        </p>
      {:else}
        {#each traces as trace (trace.ts)}
          <div class="slide-in" style="padding:4px 6px;border-radius:4px;">
            <div style="display:flex;gap:6px;align-items:flex-start;">
              <span style="font-size:12px;flex-shrink:0;margin-top:1px;">{trace.icon}</span>
              <p style="font-size:11px;color:var(--text-lo);line-height:1.5;margin:0;word-break:break-word;">
                <span style="color:{traceColor(trace.level)};font-weight:500;">[{trace.level}]</span>
                {" "}{trace.message}
              </p>
            </div>
          </div>
        {/each}
      {/if}
    </div>
  {/if}
</div>
```

## `frontend/src/env.d.ts`

```typescript
/// <reference path="../.astro/types.d.ts" />
```

## `frontend/src/lib/api.ts`

```typescript
const API_URL = import.meta.env.PUBLIC_API_URL || "http://localhost:8000";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface Job {
  id:               string;
  session_id:       string;
  title:            string;
  company:          string;
  location:         string;
  remote_type:      string;
  experience_level: string;
  salary:           string;
  skills:           string[];
  description:      string;
  apply_url:        string;
  source_domain:    string;
  source_type:      string;
  posted_date:      string;
  score:            number;
  score_breakdown:  Record<string, number>;
  match_reasons:    string[];
  reject_reasons:   string[];
  avg_feedback:     number;
}

export interface Session {
  id:            string;
  query:         string;
  status:        string;
  jobs_found:    number;
  pages_crawled: number;
  created_at:    string;
}

export interface SearchPayload {
  query:             string;
  role?:             string;
  location?:         string;
  remote_preference?: string;
  experience_level?: string;
  min_salary?:       number;
  negatives?:        string[];
  skills?:           string[];
  summary?:          string;
  experience_summary?: string;
  role_target?:      string;
}

export interface SSEEvent {
  type:          "job" | "progress" | "done" | "error" | "trace" | "stats" | "site" | "plan_update";
  job?:          Job;
  message?:      string;
  query_index?:  number;
  jobs_found?:   number;
  pages_crawled?: number;
  level?:        string;
  icon?:         string;
  data?:         Record<string, unknown>;
  tokens_total?: number;
  tokens_prompt?: number;
  tokens_comp?:  number;
  high_quality?: number;
  sites_count?:  number;
  llm_enabled?:  boolean;
  llm_guided?:   boolean;
  model?:        string;
  queries_used?: number;
  domain?:       string;
  jobs_count?:   number;
}

// ── API calls ─────────────────────────────────────────────────────────────────

export async function startSearch(payload: SearchPayload): Promise<{ session_id: string }> {
  const res = await fetch(`${API_URL}/api/search`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Search failed: ${res.statusText}`);
  return res.json();
}

export function createSSEStream(
  sessionId: string,
  onJob:      (job: Job)       => void,
  onProgress: (msg: string)    => void,
  onDone:     (stats: SSEEvent) => void,
  onError:    (err: string)    => void,
  onTrace?:      (trace: SSEEvent) => void,
  onStats?:      (stats: SSEEvent) => void,
  onSite?:       (site: SSEEvent) => void,
  onPlanUpdate?: (plan: SSEEvent) => void,
): EventSource {
  const es = new EventSource(`${API_URL}/api/search/${sessionId}/stream`);

  es.onmessage = (e: MessageEvent) => {
    try {
      const event: SSEEvent = JSON.parse(e.data);
      if (event.type === "job"         && event.job)     onJob(event.job);
      if (event.type === "progress"    && event.message) onProgress(event.message);
      if (event.type === "trace"       && onTrace)       onTrace(event);
      if (event.type === "stats"       && onStats)       onStats(event);
      if (event.type === "site"        && onSite)        onSite(event);
      if (event.type === "plan_update" && onPlanUpdate)  onPlanUpdate(event);
      if (event.type === "done")  { onDone(event);                  es.close(); }
      if (event.type === "error") { onError(event.message || "Unknown error"); es.close(); }
    } catch { /* malformed event */ }
  };

  es.onerror = () => { onError("Connection lost"); es.close(); };
  return es;
}

export async function getSessionJobs(
  sessionId: string
): Promise<{ session: Session; jobs: Job[]; count: number }> {
  const res = await fetch(`${API_URL}/api/jobs/${sessionId}`);
  if (!res.ok) throw new Error("Failed to fetch jobs");
  return res.json();
}

export async function submitFeedback(
  jobId: string, sessionId: string, rating: 1 | -1
): Promise<void> {
  await fetch(`${API_URL}/api/feedback`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ job_id: jobId, session_id: sessionId, rating }),
  });
}

export async function getSessions(): Promise<{ sessions: Session[] }> {
  const res = await fetch(`${API_URL}/api/sessions`);
  if (!res.ok) throw new Error("Failed to fetch sessions");
  return res.json();
}

export async function deleteSession(sessionId: string): Promise<void> {
  await fetch(`${API_URL}/api/sessions/${sessionId}`, { method: "DELETE" });
}

export async function uploadResume(file: File): Promise<{
  skills: string[];
  experience_level: string;
  summary: string;
}> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_URL}/api/upload-resume`, { method: "POST", body: form });
  if (!res.ok) throw new Error("Upload failed");
  return res.json();
}
```

## `frontend/src/pages/index.astro`

```astro
---
import AppShell from "../components/AppShell.svelte";
---
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>JobRadar — Deep Job Discovery</title>
  <meta name="description" content="AI-powered job discovery. Search 60+ sites, 100 pages deep." />
  <link rel="icon" type="image/svg+xml"
    href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><text y='26' font-size='26'>◎</text></svg>" />
</head>
<body>
  <!-- Top nav bar -->
  <header style="
    position: sticky; top: 0; z-index: 50;
    background: rgba(13,14,16,0.85);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid #2a2d36;
    padding: 0 24px;
    height: 52px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  ">
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="font-size:18px;color:#f59e0b;">◎</span>
      <span style="font-size:15px;font-weight:700;color:#f1f2f4;letter-spacing:0.02em;">JobRadar</span>
      <span style="font-size:11px;color:#555b6e;font-family:'JetBrains Mono',monospace;margin-left:4px;">v2.0</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
      {["60+ Sites","Deep Crawl","AI Ranked"].map(l => (
        <span style="
          font-family:'JetBrains Mono',monospace;
          font-size:10px;
          color:#8891a4;
          border:1px solid #2a2d36;
          border-radius:4px;
          padding:2px 8px;
          letter-spacing:0.05em;
        ">{l}</span>
      ))}
    </div>
  </header>

  <main style="max-width:1400px;margin:0 auto;padding:24px 20px;">
    <AppShell client:load />
  </main>
</body>
</html>
```

## `frontend/src/styles/global.css`

```css
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap");

@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --bg:        #0d0e10;
  --surface:   #13151a;
  --elevated:  #1a1d24;
  --card:      #1e2028;
  --border:    #2a2d36;
  --border-hi: #3a3d48;
  --accent:    #f59e0b;
  --accent-lo: rgba(245,158,11,0.12);
  --accent-hi: rgba(245,158,11,0.25);
  --green:     #22c55e;
  --green-lo:  rgba(34,197,94,0.12);
  --blue:      #60a5fa;
  --blue-lo:   rgba(96,165,250,0.12);
  --red:       #f87171;
  --red-lo:    rgba(248,113,113,0.12);
  --text-hi:   #f1f2f4;
  --text:      #c9ccd4;
  --text-lo:   #8891a4;
  --text-dim:  #555b6e;
}

html { scroll-behavior: smooth; }

body {
  background-color: var(--bg);
  color:            var(--text);
  font-family:      "Inter", system-ui, -apple-system, sans-serif;
  font-size:        14px;
  line-height:      1.6;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* Scrollbar */
::-webkit-scrollbar       { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }

/* ── Utility classes ── */

.card {
  background:   var(--card);
  border:       1px solid var(--border);
  border-radius: 8px;
}

.card-hover {
  transition: border-color 0.15s, background 0.15s, box-shadow 0.15s;
}
.card-hover:hover {
  border-color: var(--border-hi);
  background:   var(--elevated);
  box-shadow:   0 4px 24px rgba(0,0,0,0.3);
}

.surface {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
}

.badge {
  display:      inline-flex;
  align-items:  center;
  gap:          4px;
  padding:      2px 8px;
  border-radius: 4px;
  font-size:    11px;
  font-weight:  500;
  font-family:  "JetBrains Mono", monospace;
  letter-spacing: 0.02em;
  border:       1px solid;
}

.badge-accent   { color: var(--accent); border-color: var(--accent-hi); background: var(--accent-lo); }
.badge-green    { color: var(--green);  border-color: rgba(34,197,94,0.3); background: var(--green-lo); }
.badge-blue     { color: var(--blue);   border-color: rgba(96,165,250,0.3); background: var(--blue-lo); }
.badge-neutral  { color: var(--text-lo); border-color: var(--border); background: var(--surface); }
.badge-red      { color: var(--red);    border-color: rgba(248,113,113,0.3); background: var(--red-lo); }

.skill-tag {
  display:      inline-block;
  padding:      2px 7px;
  border-radius: 3px;
  font-size:    11px;
  font-family:  "JetBrains Mono", monospace;
  background:   rgba(96,165,250,0.08);
  border:       1px solid rgba(96,165,250,0.2);
  color:        var(--blue);
}

.score-bar-track {
  height:       2px;
  background:   var(--border);
  border-radius: 2px;
  overflow:     hidden;
}
.score-bar-fill {
  height:       100%;
  border-radius: 2px;
  transition:   width 0.7s cubic-bezier(0.4,0,0.2,1);
}

.btn {
  display:      inline-flex;
  align-items:  center;
  gap:          6px;
  padding:      7px 14px;
  border-radius: 6px;
  font-size:    13px;
  font-weight:  500;
  cursor:       pointer;
  transition:   all 0.15s;
  border:       1px solid;
  font-family:  inherit;
}

.btn-primary {
  background:   var(--accent);
  border-color: var(--accent);
  color:        #0d0e10;
  font-weight:  600;
}
.btn-primary:hover {
  background:   #fbbf24;
  border-color: #fbbf24;
}
.btn-primary:disabled {
  opacity: 0.5;
  cursor:  not-allowed;
}

.btn-ghost {
  background:   transparent;
  border-color: var(--border);
  color:        var(--text-lo);
}
.btn-ghost:hover {
  background:   var(--elevated);
  border-color: var(--border-hi);
  color:        var(--text);
}

.input {
  background:   var(--surface);
  border:       1px solid var(--border);
  border-radius: 6px;
  color:        var(--text-hi);
  font-family:  inherit;
  font-size:    14px;
  padding:      8px 12px;
  width:        100%;
  transition:   border-color 0.15s;
  outline:      none;
}
.input::placeholder { color: var(--text-dim); }
.input:focus {
  border-color: var(--accent-hi);
  box-shadow:   0 0 0 3px var(--accent-lo);
}
.input:hover { border-color: var(--border-hi); }

.label {
  font-size:    11px;
  font-weight:  600;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color:        var(--text-lo);
  display:      block;
  margin-bottom: 5px;
}

/* Pulse indicator */
.pulse-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--green);
  animation: pulse-anim 2s ease-in-out infinite;
}
@keyframes pulse-anim {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%       { opacity: 0.4; transform: scale(1.4); }
}

/* Fade-in animation */
.fade-in {
  animation: fadeIn 0.3s ease-out both;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* Slide-in for sidebar items */
.slide-in {
  animation: slideIn 0.25s ease-out both;
}
@keyframes slideIn {
  from { opacity: 0; transform: translateX(-6px); }
  to   { opacity: 1; transform: translateX(0); }
}

/* Status dots */
.dot-pending  { color: var(--text-dim); }
.dot-running  { color: var(--accent); }
.dot-done     { color: var(--green); }
.dot-failed   { color: var(--red); }

/* Divider */
.divider {
  height: 1px;
  background: var(--border);
  margin: 16px 0;
}

/* Monospace text */
.mono { font-family: "JetBrains Mono", monospace; }

/* Truncate */
.truncate-2 {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
```

## `frontend/tailwind.config.mjs`

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{astro,html,js,jsx,ts,tsx,svelte}"],
  theme: {
    extend: {
      fontFamily: {
        sans:  ['"Inter"', "system-ui", "sans-serif"],
        mono:  ['"JetBrains Mono"', "monospace"],
      },
      colors: {
        bg:       "#0d0e10",
        surface:  "#13151a",
        elevated: "#1a1d24",
        card:     "#1e2028",
        border:   "#2a2d36",
        "border-hi": "#3a3d48",
        accent:   "#f59e0b",
        green:    "#22c55e",
        blue:     "#60a5fa",
        red:      "#f87171",
        "text-hi": "#f1f2f4",
        text:     "#c9ccd4",
        "text-lo": "#8891a4",
        "text-dim": "#555b6e",
      },
      spacing: { "18": "4.5rem", "88": "22rem" },
      borderRadius: { DEFAULT: "6px" },
    },
  },
  plugins: [],
};
```
