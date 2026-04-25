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
import uuid
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
    MAX_WATCH_CYCLES,
    META_AGENT,
    META_AGENT_MAX_STEPS,
    SEARCH_EMPTY_WAVE_LIMIT,
    MIN_QUALITY_JOBS,
    MIN_SCORE_THRESHOLD,
    OPENAI_API_KEY,
    OPENAI_API_KEY_HEADER,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    USE_FREE_APIS,
    WATCH_HEARTBEAT_SEC,
)
from ranker import rank_jobs, score_job
from scraper import (
    extract_jobs_from_url,
    fetch_html,
    find_job_links,
    discover_career_urls,
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
    get_session_status,
    insert_job,
    insert_agent_event,
    insert_crawl_plan_entry,
    update_crawl_plan,
    mark_visited,
    bump_session_progress,
    update_session,
    update_source,
    url_visited,
)
from site_scrapers import scrape_site, SITE_SCRAPER_MAP
from config import JOB_SITES, JS_HEAVY_SITES
from config import JS_HEAVY_DOMAINS

log = logging.getLogger(__name__)


def _expand_career_candidates(urls: List[str]) -> List[str]:
    """Generate career site candidates from arbitrary URLs."""
    out: List[str] = []
    seen: Set[str] = set()

    def add(u: str):
        if not u or u in seen:
            return
        seen.add(u)
        out.append(u)

    for u in urls:
        try:
            p = urlparse(u)
            host = (p.netloc or "").lower()
            if host.startswith("www."):
                host = host[4:]
            if not host or "." not in host:
                continue

            # Try root domain (best-effort; avoids publicsuffix dependency)
            parts = host.split(".")
            root = ".".join(parts[-2:]) if len(parts) >= 2 else host

            scheme = p.scheme or "https"
            add(f"{scheme}://{host}/careers")
            add(f"{scheme}://{host}/jobs")
            add(f"{scheme}://{host}/careers/jobs")
            add(f"{scheme}://{host}/careers/search")
            add(f"{scheme}://{host}/jobs/search")

            add(f"https://careers.{root}")
            add(f"https://career.{root}")
            add(f"https://jobs.{root}")
            add(f"https://work.{root}/careers")
        except Exception:
            continue

    return out[:250]

# ── OpenAI ────────────────────────────────────────────────────────────────────

_openai_client = None
_LLM_AVAILABLE  = False

try:
    from openai import AsyncOpenAI
    if OPENAI_API_KEY:
        client_kwargs: dict = {"api_key": OPENAI_API_KEY}
        if OPENAI_BASE_URL:
            client_kwargs["base_url"] = OPENAI_BASE_URL
        if (OPENAI_API_KEY_HEADER or "").lower() == "api-key":
            client_kwargs["default_headers"] = {"api-key": OPENAI_API_KEY}
        _openai_client = AsyncOpenAI(**client_kwargs)
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
                        "maxItems": 40,
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

_META_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "crawl_site",
            "description": "Run a site-specific crawler for a known job site. Use this when you want fresh results from a concrete source, and repeat it if needed after learning from prior extraction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "site_key": {"type": "string", "description": "Known site key to crawl. Use '__next__' to pick the next pending site."},
                    "max_pages": {"type": "integer", "description": "Pages to crawl this pass (1-20).", "default": 5},
                    "note": {"type": "string", "description": "Short rationale (<= 200 chars)."},
                },
                "required": ["site_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for job/careers URLs using a query string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query. Use '__next__' to use the next planned query."},
                    "max_results": {"type": "integer", "description": "Max results to return (<= 30).", "default": 18},
                    "note": {"type": "string", "description": "Short rationale (<= 200 chars)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_urls",
            "description": "Select the best URLs to crawl from the given candidate list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selected_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Chosen URLs to crawl (<= 30).",
                        "maxItems": 30,
                    },
                    "note": {"type": "string", "description": "Short rationale (<= 200 chars)."},
                },
                "required": ["selected_urls"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand_urls",
            "description": "Expand a base list of URLs into likely company career page targets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Base URLs to expand.",
                    },
                    "note": {"type": "string", "description": "Short rationale (<= 200 chars)."},
                },
                "required": ["urls"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crawl_url",
            "description": "Fetch a URL (optionally rendered) and extract jobs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to crawl. Use '__next__' to pop from the crawl queue."},
                    "note": {"type": "string", "description": "Short rationale (<= 200 chars)."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_and_plan",
            "description": "Evaluate progress and decide whether to continue; may add refined queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "satisfied": {"type": "boolean", "description": "True if results are good enough to stop."},
                    "new_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional refined queries to add next (<= 10).",
                        "maxItems": 10,
                    },
                    "note": {"type": "string", "description": "Short rationale (<= 200 chars)."},
                },
                "required": ["satisfied"],
            },
        },
    },
]


async def _meta_llm_choose(
    messages: List[dict],
    st: Optional[_State] = None,
    max_tokens: int = 350,
) -> Tuple[Optional[dict], List[dict], int, int]:
    """
    Ask the LLM to choose the next meta-agent tool call(s).
    Returns (assistant_message_dict, tool_calls, prompt_tokens, completion_tokens).
    """
    if not _LLM_AVAILABLE or _openai_client is None:
        return None, [], 0, 0
    try:
        async def _create(token_param: str):
            return await _openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                tools=_META_TOOLS,
                tool_choice="auto",
                **{token_param: max_tokens},
            )

        try:
            resp = await _create("max_completion_tokens")
        except Exception as e:
            msg = str(e)
            if "Unsupported parameter" in msg and "max_completion_tokens" in msg:
                resp = await _create("max_tokens")
            elif "Unsupported parameter" in msg and "max_tokens" in msg:
                resp = await _create("max_completion_tokens")
            else:
                raise
        usage = resp.usage
        p_tok = usage.prompt_tokens if usage else 0
        c_tok = usage.completion_tokens if usage else 0
        if st:
            st.tok_prompt += p_tok
            st.tok_comp += c_tok
        msg = resp.choices[0].message
        tool_calls = msg.tool_calls or []
        return {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": tool_calls,
        }, tool_calls, p_tok, c_tok
    except Exception as exc:
        log.warning("Meta-agent LLM choose failed: %s", exc)
        return None, [], 0, 0


def _format_candidates_for_llm(candidates: List[dict], limit: int = 12) -> str:
    """Compact, safe candidate list for LLM (no HTML dumps)."""
    lines = []
    for r in (candidates or [])[:limit]:
        u = (r.get("url") or "").strip()
        if not u:
            continue
        t = (r.get("title") or "").strip().replace("\n", " ")
        s = (r.get("snippet") or "").strip().replace("\n", " ")
        lines.append(f"- {u} :: {t[:80]} :: {s[:120]}")
    return "\n".join(lines) if lines else "- (none)"


def _format_site_state_for_llm(crawlable_sites: Dict[str, dict], site_runs: Dict[str, dict], limit: int = 18) -> str:
    lines: List[str] = []
    for site_key, cfg in list(crawlable_sites.items())[:limit]:
        run = site_runs.get(site_key, {})
        status = run.get("status", "pending")
        jobs = int(run.get("jobs_found", 0) or 0)
        pages = int(run.get("pages_done", 0) or 0)
        max_pages = int(run.get("max_pages", cfg.get("max_pages", 20)) or 20)
        runs = int(run.get("runs", 0) or 0)
        lines.append(
            f"- {site_key} :: status={status} runs={runs} jobs={jobs} pages={pages}/{max_pages} label={cfg.get('label', site_key)}"
        )
    return "\n".join(lines) if lines else "- (none)"


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
        }
        if tools:
            kwargs["tools"]       = tools
            kwargs["tool_choice"] = tool_choice or "auto"

        async def _create(token_param: str):
            kk = dict(kwargs)
            kk[token_param] = max_tokens
            return await _openai_client.chat.completions.create(**kk)

        # Newer gateways/models require `max_completion_tokens` instead of `max_tokens`.
        try:
            resp = await _create("max_completion_tokens")
        except Exception as e:
            msg = str(e)
            if "Unsupported parameter" in msg and "max_completion_tokens" in msg:
                resp = await _create("max_tokens")
            elif "Unsupported parameter" in msg and "max_tokens" in msg:
                resp = await _create("max_completion_tokens")
            else:
                raise

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
        selected = [u for u in result.get("selected_urls", []) if isinstance(u, str) and u.strip()]
        # Always fill up to MAX_URLS_PER_QUERY with heuristic-good URLs so crawl scale
        # doesn't collapse to a tiny set when LLM is conservative.
        blocked = {"indeed.com","linkedin.com","glassdoor.com","naukri.com",
                   "timesjobs.com","monster.com","ziprecruiter.com","dice.com"}
        for r in results:
            u = (r.get("url") or "").strip()
            if not u:
                continue
            if any(b in u for b in blocked):
                continue
            if u not in selected:
                selected.append(u)
            if len(selected) >= MAX_URLS_PER_QUERY:
                break
        return selected[:MAX_URLS_PER_QUERY], reason

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
) -> Tuple[str, List[dict], str, dict]:
    """Bounded concurrent fetch. Returns (url, raw_jobs, page_type, fetch_meta)."""
    async with _get_semaphore():
        if url in st.visited or await url_visited(url):
            return url, [], "skipped", {"engine": "skipped", "rendered": False}

        st.visited.add(url)
        await mark_visited(url, st.session_id)

        raw_jobs, page_type, fetch_meta = await extract_jobs_from_url(url)
        st.pages_crawled += 1
        await bump_session_progress(st.session_id, pages_delta=1)

        # Follow job links from listing pages (bounded to 8 sub-pages)
        if page_type in ("job_listing", "careers_home") and st.pages_crawled < MAX_PAGES_TOTAL:
            html = await fetch_html(url, use_playwright_fallback=(urlparse(url).netloc in JS_HEAVY_DOMAINS))
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

        return url, raw_jobs, page_type, (fetch_meta or {"engine": "unknown", "rendered": False})


async def _fetch_single(url: str, st: _State) -> List[dict]:
    """Simple bounded fetch for sub-pages (no link following)."""
    async with _get_semaphore():
        await mark_visited(url, st.session_id)
        jobs, _, _fetch_meta = await extract_jobs_from_url(url)
        st.pages_crawled += 1
        await bump_session_progress(st.session_id, pages_delta=1)
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
        if scored.get("score", 0) < _effective_score_threshold(st):
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
        await bump_session_progress(st.session_id, jobs_delta=1)

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

def _agent_event_payload(
    *,
    kind: str,
    stage: str,
    title: str,
    detail: str = "",
    tool: Optional[dict] = None,
) -> dict:
    ts_ms = int(time.time() * 1000)
    payload = {
        "id": str(uuid.uuid4()),
        "ts": ts_ms,
        "kind": kind,
        "stage": stage,
        "title": title[:160],
    }
    if detail:
        payload["detail"] = detail[:500]
    if tool:
        payload["tool"] = tool
    return payload


def _sentence(text: str, fallback: str = "") -> str:
    out = (text or fallback or "").strip()
    if not out:
        return ""
    out = out[0].upper() + out[1:]
    if out[-1] not in ".!?":
        out += "."
    return out


def _meta_reasoning_text(
    tool_name: str,
    note: str,
    args: dict,
    st: "_State",
    pending_queries: List[str],
    crawl_queue: List[str],
    crawlable_sites: Optional[Dict[str, dict]] = None,
) -> str:
    if note:
        return _sentence(note)
    if tool_name == "web_search":
        query = (args.get("query") or "").strip()
        if query == "__next__":
            query = pending_queries[0] if pending_queries else st.query
        return _sentence(f"I need more leads, so I am searching the web for {query!r}")
    if tool_name == "select_urls":
        return "I found promising leads and I am queueing the strongest ATS or company career pages next."
    if tool_name == "expand_urls":
        return "I want broader company coverage, so I am expanding the current URLs into likely career pages."
    if tool_name == "crawl_url":
        target = (args.get("url") or "").strip()
        if target == "__next__":
            target = crawl_queue[0] if crawl_queue else ""
        if target:
            return _sentence(f"I am opening {target} to extract concrete job postings")
        return "I am draining the crawl queue to turn promising leads into actual job postings."
    if tool_name == "crawl_site":
        site_key = (args.get("site_key") or "").strip()
        if site_key == "__next__":
            site_key = next(iter(crawlable_sites or {}), "")
        label = (crawlable_sites or {}).get(site_key, {}).get("label", site_key)
        pages = int(args.get("max_pages") or 5)
        return _sentence(f"I want fresh structured results, so I am running the {label} crawler for about {pages} pages")
    if tool_name == "evaluate_and_plan":
        return "I have enough signal for a quick quality check, so I am deciding whether to continue or refine the plan."
    return _sentence(f"I picked {tool_name} as the next best action")


def _effective_score_threshold(st: "_State") -> float:
    profile = st.profile or {}
    prefs = st.prefs or {}
    profile_text = " ".join([
        profile.get("summary", ""),
        profile.get("experience_summary", ""),
        " ".join(profile.get("skills", []) or []),
    ]).strip()
    has_resume_like_profile = bool(profile.get("skills")) or len(profile_text) >= 80
    has_strict_filters = bool(prefs.get("experience_level")) or bool(prefs.get("min_salary")) or (prefs.get("remote_preference") not in ("", "any", None))

    if has_resume_like_profile or has_strict_filters:
        return MIN_SCORE_THRESHOLD
    return min(MIN_SCORE_THRESHOLD, 0.55)


async def _emit_agent_event(session_id: str, payload: dict) -> dict:
    """Persist and return an SSE event wrapper."""
    try:
        await insert_agent_event(session_id, payload["id"], payload["ts"], payload)
    except Exception:
        # Persistence must never break the search loop.
        pass
    return {"type": "agent_event", "data": payload}


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
    yield await _emit_agent_event(
        session_id,
        _agent_event_payload(
            kind="stage",
            stage="init",
            title="Session started",
            detail=f"mode={mode_badge}",
        ),
    )
    yield {"type": "progress", "message": f"JobRadar starting [{mode_badge}]…"}
    yield _stats_event(st, 0)

    watch_mode = bool(preferences.get("watch_mode") or False)
    watch_interval_sec = max(1, int(preferences.get("watch_interval_sec") or 900))
    watch_max_cycles_req = int(preferences.get("watch_max_cycles") or 0)
    watch_cycles_limit = watch_max_cycles_req if watch_max_cycles_req > 0 else MAX_WATCH_CYCLES
    watch_cycles = 0

    async def _pause_gate() -> AsyncGenerator[dict, None]:
        """If paused, wait until resumed or stopped; emits heartbeat events."""
        yield await _emit_agent_event(
            session_id,
            _agent_event_payload(
                kind="wait",
                stage="watch_wait",
                title="Paused",
                detail="Awaiting resume/stop",
            ),
        )
        yield {"type": "progress", "message": "⏸ Paused — waiting for resume…"}
        while True:
            status = await get_session_status(session_id)
            if status == "stopped":
                yield await _emit_agent_event(
                    session_id,
                    _agent_event_payload(
                        kind="error",
                        stage="done",
                        title="Stopped while paused",
                        detail="User requested stop",
                    ),
                )
                return
            if status != "paused":
                return
            yield await _emit_agent_event(
                session_id,
                _agent_event_payload(
                    kind="metric",
                    stage="watch_wait",
                    title="Paused heartbeat",
                    detail=f"next_check_sec={WATCH_HEARTBEAT_SEC}",
                ),
            )
            await asyncio.sleep(WATCH_HEARTBEAT_SEC)

    async def _sleep_with_controls(seconds: int, title: str, detail: str) -> AsyncGenerator[dict, None]:
        """Sleep in small increments so pause/stop stays responsive."""
        end = time.time() + max(1, int(seconds))
        yield await _emit_agent_event(
            session_id,
            _agent_event_payload(kind="wait", stage="watch_wait", title=title, detail=detail),
        )
        while True:
            status = await get_session_status(session_id)
            if status == "stopped":
                yield await _emit_agent_event(
                    session_id,
                    _agent_event_payload(kind="error", stage="done", title="Stopped", detail="User requested stop"),
                )
                return
            if status == "paused":
                async for ev in _pause_gate():
                    yield ev
                # If pause gate returns due to stop, end now.
                status2 = await get_session_status(session_id)
                if status2 == "stopped":
                    return
            now = time.time()
            if now >= end:
                return
            await asyncio.sleep(min(WATCH_HEARTBEAT_SEC, max(0.2, end - now)))

    # ══════════════════════════════════════════════════════════════════════════
    # Wave 0 — Free structured APIs (fast, high-quality structured data)
    # ══════════════════════════════════════════════════════════════════════════

    if USE_FREE_APIS:
        yield await _emit_agent_event(
            session_id,
            _agent_event_payload(
                kind="stage",
                stage="search",
                title="Fetching free job APIs",
                detail="remotive/arbeitnow/jobicy/remoteok/himalayas/themuse",
            ),
        )
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
                yield await _emit_agent_event(
                    session_id,
                    _agent_event_payload(
                        kind="tool_result",
                        stage="search",
                        title=f"{name} API failed",
                        detail=str(result)[:180],
                        tool={
                            "name": f"api:{name}",
                            "input_summary": query[:140],
                            "output_summary": "error",
                            "status": "error",
                        },
                    ),
                )
                continue

            jobs_from_api = result if isinstance(result, list) else []
            yield _trace("api", f"{name} → {len(jobs_from_api)} jobs found",
                         source=name, count=len(jobs_from_api))
            yield await _emit_agent_event(
                session_id,
                _agent_event_payload(
                    kind="tool_result",
                    stage="search",
                    title=f"{name} API results",
                    detail=f"jobs={len(jobs_from_api)}",
                    tool={
                        "name": f"api:{name}",
                        "input_summary": query[:140],
                        "output_summary": f"{len(jobs_from_api)} jobs",
                        "status": "ok",
                    },
                ),
            )

            if jobs_from_api:
                st.sites[name.lower()] = len(jobs_from_api)
                yield {"type": "site", "domain": name.lower(), "jobs_count": len(jobs_from_api)}

            async for job in _process_raw_jobs(jobs_from_api, st, infer_missing_salary=False):
                yield {"type": "job", "job": job}

        yield _stats_event(st, 0)
        yield _trace("api", f"Free APIs done: {len(st.jobs)} jobs so far")
    else:
        yield _trace("api", "Free APIs disabled (crawl-first mode)")

    # ══════════════════════════════════════════════════════════════════════════
    # Wave 0.5 — Site-specific crawlers (60+ sites, deep pagination)
    # ══════════════════════════════════════════════════════════════════════════

    if META_AGENT and _LLM_AVAILABLE:
        yield await _emit_agent_event(
            session_id,
            _agent_event_payload(
                kind="stage",
                stage="plan",
                title="Meta-agent mode enabled",
                detail="Starting with built-in site crawlers first so the agent has real jobs and URLs to work with.",
            ),
        )

    yield await _emit_agent_event(
        session_id,
        _agent_event_payload(
            kind="stage",
            stage="plan",
            title="Building site crawl plan",
            detail=f"sites_total={len(JOB_SITES)}",
        ),
    )
    yield _trace("plan", f"Building crawl plan for {len(JOB_SITES)} job sites…")

    # Build crawl plan in DB
    crawlable_sites = {
        k: v
        for k, v in JOB_SITES.items()
        if v.get("type") not in ("api",) and (k in SITE_SCRAPER_MAP or v.get("url"))
    }

    uncovered = sorted(
        k
        for k, v in JOB_SITES.items()
        if v.get("type") not in ("api",) and (k not in SITE_SCRAPER_MAP and not v.get("url"))
    )
    if uncovered:
        yield _trace(
            "warn",
            f"{len(uncovered)} JOB_SITES entries have no scraper and no url template; skipping in Wave 0.5",
            site_keys=uncovered[:25],
        )

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

    site_runs: Dict[str, dict] = {
        site_key: {
            "status": "pending",
            "jobs_found": 0,
            "pages_done": 0,
            "max_pages": int(site_cfg.get("max_pages", 20) or 20),
            "runs": 0,
        }
        for site_key, site_cfg in crawlable_sites.items()
    }
    next_site_keys: List[str] = list(crawlable_sites.keys())

    async def _crawl_site_once(site_key: str, max_pages: int) -> AsyncGenerator[dict, None]:
        site_cfg = crawlable_sites.get(site_key)
        if not site_cfg:
            yield await _emit_agent_event(
                session_id,
                _agent_event_payload(
                    kind="tool_result",
                    stage="crawl",
                    title="crawl_site skipped",
                    detail=f"unknown site_key={site_key}",
                    tool={"name": "crawl_site", "status": "warn", "output_summary": "unknown site"},
                ),
            )
            return

        run_state = site_runs.setdefault(site_key, {
            "status": "pending",
            "jobs_found": 0,
            "pages_done": 0,
            "max_pages": int(site_cfg.get("max_pages", 20) or 20),
            "runs": 0,
        })
        bounded_pages = max(1, min(int(max_pages or 1), int(site_cfg.get("max_pages", 20) or 20)))
        run_state["status"] = "running"
        run_state["runs"] += 1
        run_state["max_pages"] = max(run_state.get("max_pages", bounded_pages), bounded_pages)

        await update_crawl_plan(session_id, site_key, status="running")
        yield {
            "type": "site_active",
            "site_key": site_key,
            "site_label": site_cfg.get("label", site_key),
        }
        yield await _emit_agent_event(
            session_id,
            _agent_event_payload(
                kind="tool_call",
                stage="crawl",
                title=f"Site crawl started: {site_cfg.get('label', site_key)}",
                detail=f"site_key={site_key} max_pages={bounded_pages}",
                tool={
                    "name": "crawl_site",
                    "input_summary": f"{site_key} :: pages={bounded_pages}",
                    "output_summary": "",
                    "status": "running",
                },
            ),
        )
        yield {
            "type": "plan_update",
            "site_key": site_key,
            "site_label": site_cfg["label"],
            "status": "running",
            "jobs_found": run_state["jobs_found"],
            "pages_done": run_state["pages_done"],
            "pages_total": run_state["max_pages"],
        }

        try:
            raw_jobs = await scrape_site(site_key, query, max_pages=bounded_pages)
        except Exception as e:
            err = str(e)
            run_state["status"] = "failed"
            await update_crawl_plan(session_id, site_key, status="failed", error_msg=err)
            yield await _emit_agent_event(
                session_id,
                _agent_event_payload(
                    kind="tool_result",
                    stage="crawl",
                    title=f"Site crawl failed: {site_cfg.get('label', site_key)}",
                    detail=err[:220],
                    tool={
                        "name": "crawl_site",
                        "input_summary": site_key,
                        "output_summary": "error",
                        "status": "error",
                    },
                ),
            )
            yield {
                "type": "plan_update",
                "site_key": site_key,
                "site_label": site_cfg["label"],
                "status": "failed",
                "jobs_found": run_state["jobs_found"],
                "pages_done": run_state["pages_done"],
                "pages_total": run_state["max_pages"],
            }
            return

        observed_pages = 1
        st.pages_crawled += observed_pages
        await bump_session_progress(session_id, pages_delta=observed_pages)
        run_state["pages_done"] += observed_pages

        job_count = 0
        async for job in _process_raw_jobs(raw_jobs, st, infer_missing_salary=False):
            yield {"type": "job", "job": job}
            job_count += 1

        run_state["jobs_found"] += job_count
        run_state["status"] = "done"
        await update_crawl_plan(
            session_id,
            site_key,
            status="done",
            jobs_found=job_count,
            pages_done=bounded_pages,
        )
        yield await _emit_agent_event(
            session_id,
            _agent_event_payload(
                kind="tool_result",
                stage="crawl",
                title=f"Site crawl done: {site_cfg.get('label', site_key)}",
                detail=f"jobs={job_count} runs={run_state['runs']}",
                tool={
                    "name": "crawl_site",
                    "input_summary": f"{site_key} :: pages={bounded_pages}",
                    "output_summary": f"{job_count} jobs",
                    "status": "ok",
                },
            ),
        )
        yield {
                "type": "plan_update",
                "site_key": site_key,
                "site_label": site_cfg["label"],
                "status": "done",
                "jobs_found": run_state["jobs_found"],
                "pages_done": run_state["pages_done"],
                "pages_total": run_state["max_pages"],
            }
        if job_count > 0:
            st.sites[site_cfg["label"]] = st.sites.get(site_cfg["label"], 0) + job_count
            yield {"type": "site", "domain": site_cfg["label"], "jobs_count": job_count}
        yield _stats_event(st, queries_used)

    if not (META_AGENT and _LLM_AVAILABLE):
        for site_key in list(next_site_keys):
            site_cfg = crawlable_sites.get(site_key, {})
            async for ev in _crawl_site_once(site_key, int(site_cfg.get("max_pages", 20) or 20)):
                yield ev
            if site_key in next_site_keys:
                next_site_keys.remove(site_key)
        yield _trace("api", f"Site crawl complete: {len(st.jobs)} jobs total across {len(st.sites)} sources")

    # ══════════════════════════════════════════════════════════════════════════
    # Wave 1+ — LLM-planned web crawl
    # ══════════════════════════════════════════════════════════════════════════

    yield await _emit_agent_event(
        session_id,
        _agent_event_payload(
            kind="stage",
            stage="plan",
            title="Planning web crawl strategy",
            detail="LLM query planning" if _LLM_AVAILABLE else "heuristic query planning",
        ),
    )
    yield _trace("plan", "LLM planning search queries…")

    if _LLM_AVAILABLE:
        t0 = time.time()
        pending_queries, reasoning = await _plan_queries(st)
        dt = int((time.time() - t0) * 1000)
        yield _trace("plan",
                     f"Generated {len(pending_queries)} queries",
                     reasoning=reasoning,
                     queries=pending_queries[:6],
                     tokens=st.total_tokens)
        yield await _emit_agent_event(
            session_id,
            _agent_event_payload(
                kind="decision",
                stage="plan",
                title="LLM generated search queries",
                detail=_sentence(
                    str(reasoning or ""),
                    f"I mapped the search into {len(pending_queries)} targeted query variations to cover stronger companies, ATS pages, and role phrasing.",
                )[:480],
                tool={
                    "name": "llm:generate_search_queries",
                    "input_summary": query[:140],
                    "output_summary": f"{len(pending_queries)} queries",
                    "status": "ok",
                    "duration_ms": dt,
                },
            ),
        )
    else:
        parsed = preprocess_query(query, preferences)
        pending_queries = _smart_heuristic_queries(query, preferences, parsed)
        yield _trace("plan", f"Heuristic: {len(pending_queries)} queries",
                     queries=pending_queries[:6])
        yield await _emit_agent_event(
            session_id,
            _agent_event_payload(
                kind="decision",
                stage="plan",
                title="Heuristic generated search queries",
                detail=f"I generated {len(pending_queries)} fallback search queries from your role, experience level, and location preferences.",
                tool={
                    "name": "heuristic:generate_queries",
                    "input_summary": query[:140],
                    "output_summary": f"{len(pending_queries)} queries",
                    "status": "ok",
                },
            ),
        )

    yield {"type": "progress",
           "message": f"Starting deep crawl: {len(pending_queries)} search strategies"}

    queries_used = 0
    satisfied = False
    budget_stop = False
    meta_steps = 0
    crawl_queue: List[str] = []
    candidates: List[dict] = []
    last_search_query = ""
    # Count consecutive waves/queries that returned no search results.
    empty_search_waves = 0
    while True:
        status = await get_session_status(session_id)
        if status == "stopped":
            yield await _emit_agent_event(
                session_id,
                _agent_event_payload(kind="error", stage="done", title="Stopped", detail="User requested stop"),
            )
            budget_stop = True
            break
        if status == "paused":
            async for ev in _pause_gate():
                yield ev
            status2 = await get_session_status(session_id)
            if status2 == "stopped":
                budget_stop = True
                break

        if st.high_quality >= MIN_QUALITY_JOBS:
            satisfied = True

        # Meta-agent mode has two work queues: search queries and crawl URLs.
        no_work_meta = (not pending_queries) and (not crawl_queue) and (not next_site_keys)
        no_work_classic = (not pending_queries)

        if (queries_used >= MAX_QUERIES_PER_SESSION) and (not crawl_queue):
            # Search budget exhausted; only allow draining crawl queue.
            pending_queries = []

        if ((META_AGENT and _LLM_AVAILABLE and no_work_meta) or (not (META_AGENT and _LLM_AVAILABLE) and no_work_classic)) or (
            (queries_used >= MAX_QUERIES_PER_SESSION) and (not crawl_queue)
        ):
            if (
                watch_mode
                and not satisfied
                and not budget_stop
                and (watch_cycles < watch_cycles_limit)
                and (queries_used < MAX_QUERIES_PER_SESSION)
            ):
                watch_cycles += 1
                async for ev in _sleep_with_controls(
                    watch_interval_sec,
                    title="Watch sleeping",
                    detail=f"cycle={watch_cycles}/{watch_cycles_limit or '∞'} interval_sec={watch_interval_sec}",
                ):
                    yield ev

                # Re-plan queries for the next cycle (bounded, safe).
                yield await _emit_agent_event(
                    session_id,
                    _agent_event_payload(
                        kind="stage",
                        stage="plan",
                        title="Watch cycle replanning",
                        detail=f"cycle={watch_cycles}",
                    ),
                )
                if _LLM_AVAILABLE:
                    pending_queries, reasoning = await _plan_queries(st)
                    yield _trace("plan", f"Watch cycle: {len(pending_queries)} queries", reasoning=reasoning)
                else:
                    parsed = preprocess_query(query, preferences)
                    pending_queries = _smart_heuristic_queries(query, preferences, parsed)
                    yield _trace("plan", f"Watch cycle (heuristic): {len(pending_queries)} queries")

                yield {"type": "progress", "message": f"🔁 Watch cycle {watch_cycles}: resuming crawl…"}
                continue
            break

        if META_AGENT and _LLM_AVAILABLE:
            if meta_steps >= META_AGENT_MAX_STEPS:
                yield await _emit_agent_event(
                    session_id,
                    _agent_event_payload(
                        kind="error",
                        stage="done",
                        title="Meta-agent step limit reached",
                        detail=f"steps={meta_steps} limit={META_AGENT_MAX_STEPS}",
                    ),
                )
                budget_stop = True
                break

            # Build compact state for LLM (safe, no hidden reasoning).
            next_qs = pending_queries[:3]
            next_urls = crawl_queue[:3]
            next_sites = [site for site in next_site_keys if site_runs.get(site, {}).get("status") != "running"][:6]
            cand_txt = _format_candidates_for_llm(candidates, limit=10) if candidates else "- (none)"
            site_txt = _format_site_state_for_llm(crawlable_sites, site_runs, limit=18)

            sys = (
                "You are the JobRadar meta-agent. You must control the search by choosing ONE tool call at a time. "
                "Use short, user-readable notes (<= 200 chars). "
                "Avoid blocked aggregators (LinkedIn/Indeed/Glassdoor) and prefer ATS/career pages. "
                "You may use crawl_site multiple times whenever you want more direct site coverage or want to revisit a source. "
                "When you have candidates, call select_urls. When you have a crawl queue, call crawl_url. "
                "Periodically call evaluate_and_plan to decide continue/satisfied and optionally add new queries."
            )
            user = (
                f"USER_QUERY: {st.query}\n"
                f"PREFS: remote={st.prefs.get('remote_preference')} location={st.prefs.get('location')} level={st.prefs.get('experience_level')}\n"
                f"STATS: jobs={len(st.jobs)} high_quality={st.high_quality} pages={st.pages_crawled} tokens={st.total_tokens}\n"
                f"BUDGET: queries_used={queries_used}/{MAX_QUERIES_PER_SESSION} pages={st.pages_crawled}/{MAX_PAGES_TOTAL}\n"
                f"NEXT_QUERIES: {next_qs}\n"
                f"CRAWL_QUEUE: {next_urls}\n"
                f"NEXT_SITES: {next_sites}\n"
                f"LAST_SEARCH_QUERY: {last_search_query or '(none)'}\n"
                f"SITE_STATE:\n{site_txt}\n"
                f"CANDIDATES:\n{cand_txt}\n"
                "Pick the next best action now."
            )

            _assistant_msg, tool_calls, _pt, _ct = await _meta_llm_choose(
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                st=st,
                max_tokens=420,
            )

            # Fallback if the model doesn't call tools.
            if not tool_calls:
                if next_sites:
                    tool_name = "crawl_site"
                    args = {"site_key": "__next__", "max_pages": 5, "note": "Start with a direct site crawler to get real structured jobs."}
                elif crawl_queue:
                    tool_name = "crawl_url"
                    args = {"url": "__next__", "note": "Drain crawl queue (fallback)."}
                elif pending_queries:
                    tool_name = "web_search"
                    args = {"query": "__next__", "max_results": 18, "note": "Find fresh ATS/career pages (fallback)."}
                else:
                    break
            else:
                tc = tool_calls[0]
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}

            note = (args.get("note") or "").strip()[:200]
            reasoning_note = _meta_reasoning_text(tool_name, note, args, st, pending_queries, crawl_queue, crawlable_sites)
            yield await _emit_agent_event(
                session_id,
                _agent_event_payload(
                    kind="decision",
                    stage="plan" if tool_name in ("select_urls", "expand_urls", "evaluate_and_plan") else ("search" if tool_name == "web_search" else "crawl"),
                    title="Meta-agent reasoning",
                    detail=reasoning_note[:480],
                ),
            )
            yield await _emit_agent_event(
                session_id,
                _agent_event_payload(
                    kind="tool_call",
                    stage="plan" if tool_name in ("select_urls", "expand_urls", "evaluate_and_plan") else ("search" if tool_name == "web_search" else "crawl"),
                    title=f"{tool_name}",
                    detail=note or "",
                    tool={
                        "name": tool_name,
                        "input_summary": str(args)[:240],
                        "output_summary": "",
                        "status": "running",
                    },
                ),
            )

            # Execute meta tools.
            if tool_name == "crawl_site":
                site_key = (args.get("site_key") or "").strip()
                if site_key == "__next__":
                    site_key = next((site for site in next_site_keys if site_runs.get(site, {}).get("status") != "running"), "")
                max_pages = int(args.get("max_pages") or 5)
                if not site_key:
                    yield await _emit_agent_event(
                        session_id,
                        _agent_event_payload(
                            kind="tool_result",
                            stage="crawl",
                            title="crawl_site skipped",
                            detail="no site available",
                            tool={"name": "crawl_site", "status": "warn", "output_summary": "skipped"},
                        ),
                    )
                    meta_steps += 1
                    continue
                async for ev in _crawl_site_once(site_key, max_pages):
                    yield ev
                if site_key in next_site_keys:
                    next_site_keys.remove(site_key)

            elif tool_name == "web_search":
                if queries_used >= MAX_QUERIES_PER_SESSION:
                    yield await _emit_agent_event(
                        session_id,
                        _agent_event_payload(
                            kind="tool_result",
                            stage="search",
                            title="web_search skipped",
                            detail="query budget exhausted",
                            tool={"name": "web_search", "status": "warn", "output_summary": "skipped"},
                        ),
                    )
                    meta_steps += 1
                    continue
                q = (args.get("query") or "").strip()
                if q == "__next__":
                    q = pending_queries.pop(0) if pending_queries else st.query
                last_search_query = q
                max_r = int(args.get("max_results") or 18)
                max_r = max(3, min(30, max_r))
                queries_used += 1
                t0 = time.time()
                candidates = await search_web(q, max_results=max_r)
                dt = int((time.time() - t0) * 1000)
                doms = []
                for r in (candidates or [])[:10]:
                    u = (r.get("url") or "")
                    if u:
                        doms.append(urlparse(u).netloc.replace("www.", ""))
                top_doms = ", ".join(list(dict.fromkeys(doms))[:5])
                yield await _emit_agent_event(
                    session_id,
                    _agent_event_payload(
                        kind="tool_result",
                        stage="search",
                        title="web_search results",
                        detail=f"urls={len(candidates)} top={top_doms}",
                        tool={
                            "name": "web_search",
                            "input_summary": q[:180],
                            "output_summary": f"{len(candidates)} urls",
                            "status": "ok",
                            "duration_ms": dt,
                        },
                    ),
                )

                # Guard: repeated empty search results indicate broken search or overly-narrow queries.
                if not candidates:
                    empty_search_waves += 1
                    yield _trace("search", f"No results for query '{q[:80]}' (empty waves={empty_search_waves})")
                    if empty_search_waves >= SEARCH_EMPTY_WAVE_LIMIT:
                        yield await _emit_agent_event(
                            session_id,
                            _agent_event_payload(
                                kind="error",
                                stage="done",
                                title="No search leads",
                                detail=f"No results for {empty_search_waves} consecutive queries; stopping to avoid token waste",
                            ),
                        )
                        budget_stop = True
                        break
                    meta_steps += 1
                    yield _stats_event(st, queries_used)
                    yield {"type": "progress", "message": f"🧠 Agent step {meta_steps} · jobs {len(st.jobs)} · pages {st.pages_crawled}"}
                    continue
                else:
                    empty_search_waves = 0

            elif tool_name == "select_urls":
                sel = args.get("selected_urls") or []
                if not isinstance(sel, list):
                    sel = []
                dedup = []
                seen = set()
                for u in sel:
                    if not isinstance(u, str):
                        continue
                    uu = u.strip()
                    if not uu or uu in seen:
                        continue
                    seen.add(uu)
                    dedup.append(uu)
                    if len(dedup) >= 30:
                        break
                added = 0
                for u in dedup:
                    if u not in crawl_queue and u not in st.visited:
                        crawl_queue.append(u)
                        added += 1
                yield await _emit_agent_event(
                    session_id,
                    _agent_event_payload(
                        kind="tool_result",
                        stage="plan",
                        title="select_urls queued targets",
                        detail=f"added={added} queue={len(crawl_queue)}",
                        tool={
                            "name": "select_urls",
                            "input_summary": f"selected={len(dedup)}",
                            "output_summary": f"queue={len(crawl_queue)}",
                            "status": "ok",
                        },
                    ),
                )

            elif tool_name == "expand_urls":
                base = args.get("urls") or []
                if not isinstance(base, list):
                    base = []
                base2 = [u for u in base if isinstance(u, str) and u.strip()]
                expanded = base2 + _expand_career_candidates(base2)
                added = 0
                seen = set(crawl_queue) | set(st.visited)
                for u in expanded:
                    if u and u not in seen:
                        crawl_queue.append(u)
                        seen.add(u)
                        added += 1
                        if len(crawl_queue) >= 500:
                            break
                yield await _emit_agent_event(
                    session_id,
                    _agent_event_payload(
                        kind="tool_result",
                        stage="plan",
                        title="expand_urls added career targets",
                        detail=f"added={added} queue={len(crawl_queue)}",
                        tool={
                            "name": "expand_urls",
                            "input_summary": f"base={len(base2)}",
                            "output_summary": f"queue={len(crawl_queue)}",
                            "status": "ok",
                        },
                    ),
                )

            elif tool_name == "crawl_url":
                u = (args.get("url") or "").strip()
                if u == "__next__":
                    u = crawl_queue.pop(0) if crawl_queue else ""
                if not u:
                    yield await _emit_agent_event(
                        session_id,
                        _agent_event_payload(
                            kind="tool_result",
                            stage="crawl",
                            title="crawl_url skipped",
                            detail="no URL available",
                            tool={"name": "crawl_url", "status": "warn", "output_summary": "skipped"},
                        ),
                    )
                    meta_steps += 1
                    continue
                yield {"type": "site_active", "domain": urlparse(u).netloc, "url": u}
                t0 = time.time()
                url_done, raw_jobs, page_type, fetch_meta = await _fetch_url_bounded(u, st)
                dt = int((time.time() - t0) * 1000)
                job_count = 0
                async for job in _process_raw_jobs(
                    raw_jobs, st,
                    infer_missing_salary=(_LLM_AVAILABLE and len(st.jobs) < 100),
                ):
                    yield {"type": "job", "job": job}
                    job_count += 1
                engine = (fetch_meta or {}).get("engine") or "unknown"
                yield await _emit_agent_event(
                    session_id,
                    _agent_event_payload(
                        kind="tool_result",
                        stage="crawl",
                        title="crawl_url extracted",
                        detail=f"jobs={job_count} page_type={page_type} engine={engine}",
                        tool={
                            "name": "crawl_url",
                            "input_summary": url_done[:200],
                            "output_summary": f"{job_count} jobs",
                            "status": "ok",
                            "duration_ms": dt,
                        },
                    ),
                )

            elif tool_name == "evaluate_and_plan":
                sat = bool(args.get("satisfied"))
                new_q = args.get("new_queries") or []
                if not isinstance(new_q, list):
                    new_q = []
                new_q2 = [q for q in new_q if isinstance(q, str) and q.strip()][:10]
                if new_q2:
                    pending_queries = new_q2 + pending_queries
                if st.high_quality >= MIN_QUALITY_JOBS:
                    sat = True
                satisfied = sat
                yield await _emit_agent_event(
                    session_id,
                    _agent_event_payload(
                        kind="decision",
                        stage="evaluate",
                        title="evaluate_and_plan",
                        detail=_sentence(
                            note,
                            (
                                "I reviewed the current quality and we already have enough strong matches."
                                if sat else
                                f"I reviewed the current quality and want to keep searching with {len(new_q2)} refined follow-up queries."
                            ),
                        )[:480],
                        tool={
                            "name": "evaluate_and_plan",
                            "input_summary": f"jobs={len(st.jobs)} high_quality={st.high_quality}",
                            "output_summary": "satisfied" if sat else "continue",
                            "status": "ok",
                        },
                    ),
                )
                if satisfied:
                    break

            else:
                yield await _emit_agent_event(
                    session_id,
                    _agent_event_payload(
                        kind="tool_result",
                        stage="plan",
                        title="Unknown tool ignored",
                        detail=tool_name,
                        tool={"name": tool_name, "status": "warn", "output_summary": "ignored"},
                    ),
                )

            meta_steps += 1
            yield _stats_event(st, queries_used)
            yield {"type": "progress", "message": f"🧠 Agent step {meta_steps} · jobs {len(st.jobs)} · pages {st.pages_crawled}"}
            continue

        # Take 3–5 queries per wave and run them in parallel
        wave      = pending_queries[:4]
        pending_queries = pending_queries[4:]
        queries_used   += len(wave)

        direct_urls = [q[4:] for q in wave if isinstance(q, str) and q.startswith("url:")]
        wave_queries = [q for q in wave if not (isinstance(q, str) and q.startswith("url:"))]

        # ── Parallel DDG searches ──────────────────────────────────────────
        if wave_queries:
            yield _trace("search", f"Searching {len(wave_queries)} queries in parallel…",
                         queries=wave_queries)
            yield await _emit_agent_event(
                session_id,
                _agent_event_payload(
                    kind="tool_call",
                    stage="search",
                    title="web_search batch",
                    detail=f"queries={len(wave_queries)}",
                    tool={
                        "name": "web_search",
                        "input_summary": " | ".join([q[:60] for q in wave_queries])[:240],
                        "output_summary": "",
                        "status": "running",
                    },
                ),
            )

        search_batches = []
        if wave_queries:
            search_tasks = [search_web(q, max_results=MAX_URLS_PER_QUERY + 4) for q in wave_queries]
            search_batches = await asyncio.gather(*search_tasks, return_exceptions=True)

        all_results: List[dict] = []
        for q, batch in zip(wave_queries, search_batches):
            if isinstance(batch, Exception):
                yield _trace("warn", f"Search failed: {q[:50]}")
                continue
            yield _trace("search", f"'{q[:55]}' → {len(batch)} URLs",
                         query=q, count=len(batch))
            all_results.extend(batch)

        # Include any direct URLs discovered from earlier frontier expansion.
        if direct_urls:
            all_results.extend([{"url": u, "title": "direct", "snippet": ""} for u in direct_urls if u])

        if not all_results:
            empty_search_waves += 1
            yield _trace("search", f"No results for wave queries (empty waves={empty_search_waves})")
            if empty_search_waves >= SEARCH_EMPTY_WAVE_LIMIT:
                yield await _emit_agent_event(
                    session_id,
                    _agent_event_payload(
                        kind="error",
                        stage="done",
                        title="No search leads",
                        detail=f"No results from {empty_search_waves} consecutive query waves; stopping to avoid token waste",
                    ),
                )
                budget_stop = True
                break
            continue
        yield await _emit_agent_event(
            session_id,
            _agent_event_payload(
                kind="tool_result",
                stage="search",
                title="web_search results",
                detail=f"urls={len(all_results)}",
                tool={
                    "name": "web_search",
                    "input_summary": f"queries={len(wave_queries)}",
                    "output_summary": f"{len(all_results)} urls",
                    "status": "ok",
                },
            ),
        )

        # ── LLM picks best URLs ────────────────────────────────────────────
        if _LLM_AVAILABLE:
            t0 = time.time()
            urls_to_fetch, pick_reason = await _pick_urls(all_results, st)
            dt = int((time.time() - t0) * 1000)
            yield _trace("llm", f"URL prioritization: {pick_reason[:80]}",
                         selected=len(urls_to_fetch),
                         tokens=st.total_tokens)
            yield await _emit_agent_event(
                session_id,
                _agent_event_payload(
                    kind="decision",
                    stage="plan",
                    title="prioritize_urls selected targets",
                    detail=(f"selected={len(urls_to_fetch)} · "
                            f"why={str(pick_reason or '')[:220]}")[:480],
                    tool={
                        "name": "prioritize_urls",
                        "input_summary": f"candidates={len(all_results)}",
                        "output_summary": f"selected={len(urls_to_fetch)}",
                        "status": "ok",
                        "duration_ms": dt,
                    },
                ),
            )
        else:
            blocked = {"indeed.com","linkedin.com","glassdoor.com","naukri.com",
                       "timesjobs.com","monster.com","ziprecruiter.com","shine.com"}
            urls_to_fetch = [
                r["url"] for r in all_results
                if r.get("url") and not any(b in r.get("url","") for b in blocked)
            ][:MAX_URLS_PER_QUERY]

        # ── Filter already-visited ─────────────────────────────────────────
        # Expand into likely company career targets to massively increase crawl surface area.
        expanded = urls_to_fetch + _expand_career_candidates(urls_to_fetch)
        fresh_urls = [u for u in expanded if u not in st.visited]
        if expanded and len(expanded) != len(urls_to_fetch):
            yield await _emit_agent_event(
                session_id,
                _agent_event_payload(
                    kind="tool_result",
                    stage="plan",
                    title="propose_company_sites expanded targets",
                    detail=f"expanded={len(expanded)} base={len(urls_to_fetch)}",
                    tool={
                        "name": "propose_company_sites",
                        "input_summary": f"base_urls={len(urls_to_fetch)}",
                        "output_summary": f"expanded_urls={len(expanded)}",
                        "status": "ok",
                    },
                ),
            )

        if not fresh_urls:
            yield _trace("skip", "All URLs already visited — skipping")
            continue

        # ── Concurrent fetch ───────────────────────────────────────────────
        yield _trace("fetch", f"Fetching {len(fresh_urls)} URLs concurrently…",
                     urls=fresh_urls[:6])
        yield await _emit_agent_event(
            session_id,
            _agent_event_payload(
                kind="tool_call",
                stage="crawl",
                title="crawl_url batch",
                detail=f"urls={len(fresh_urls)}",
                tool={
                    "name": "crawl_url",
                    "input_summary": " | ".join([urlparse(u).netloc for u in fresh_urls[:6]]),
                    "output_summary": "",
                    "status": "running",
                },
            ),
        )

        fetch_tasks = []
        for u in fresh_urls:
            if st.pages_crawled >= MAX_PAGES_TOTAL:
                break
            yield {"type": "site_active", "domain": urlparse(u).netloc, "url": u}
            fetch_tasks.append(_fetch_url_bounded(u, st))
        fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        # Follow-up career discovery frontier (bounded)
        followup: List[str] = []
        wave_jobs_added = 0
        wave_urls_done = 0
        wave_urls_err = 0
        for item in fetch_results:
            if isinstance(item, Exception):
                yield _trace("warn", f"Fetch error: {item}")
                wave_urls_err += 1
                continue
            url, raw_jobs, page_type, fetch_meta = item
            wave_urls_done += 1
            if fetch_meta and fetch_meta.get("rendered"):
                yield await _emit_agent_event(
                    session_id,
                    _agent_event_payload(
                        kind="tool_result",
                        stage="crawl",
                        title="Rendered fetch",
                        detail=f"engine={fetch_meta.get('engine')} domain={urlparse(url).netloc}",
                        tool={
                            "name": "rendered_fetch",
                            "input_summary": url[:200],
                            "output_summary": str(fetch_meta.get("engine") or ""),
                            "status": "ok",
                        },
                    ),
                )
            if (not raw_jobs) and page_type in ("irrelevant", "failed"):
                html = await fetch_html(url, use_playwright_fallback=(urlparse(url).netloc in JS_HEAVY_DOMAINS))
                if html:
                    followup.extend(discover_career_urls(html, url))

            domain   = urlparse(url).netloc
            job_count = 0

            async for job in _process_raw_jobs(
                raw_jobs, st,
                infer_missing_salary=(_LLM_AVAILABLE and len(st.jobs) < 100),
            ):
                yield {"type": "job", "job": job}
                job_count += 1
                wave_jobs_added += 1

            if job_count > 0:
                yield _trace("extract",
                             f"{domain} → {job_count} jobs",
                             url=url, jobs=job_count, page_type=page_type)
                yield {"type": "site", "domain": domain, "jobs_count": job_count}
            else:
                yield _trace("fetch",
                             f"{domain} → no jobs ({page_type})",
                             url=url, page_type=page_type)

        yield await _emit_agent_event(
            session_id,
            _agent_event_payload(
                kind="tool_result",
                stage="crawl",
                title="crawl_url results",
                detail=f"urls_ok={wave_urls_done} urls_err={wave_urls_err} jobs={wave_jobs_added}",
                tool={
                    "name": "crawl_url",
                    "input_summary": f"urls={len(fresh_urls)}",
                    "output_summary": f"jobs={wave_jobs_added}",
                    "status": "ok" if wave_urls_err == 0 else "warn",
                },
            ),
        )

        if followup:
            # Add follow-ups early in the queue for breadth.
            dedup = []
            for u in followup:
                if u not in st.visited:
                    dedup.append(u)
                if len(dedup) >= 80:
                    break
            if dedup:
                yield _trace("plan", f"Discovered {len(dedup)} career targets", urls=dedup[:8])
                # Push as immediate crawl wave by prepending to pending URLs via queries list trick.
                # We treat them as URLs to fetch directly in the next iteration by injecting a synthetic query batch.
                pending_queries = [f"url:{u}" for u in dedup] + pending_queries

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
            budget_stop = True
            break

        # ── Quality evaluation (every 2 waves) ────────────────────────────
        if queries_used % 2 == 0:
            if _LLM_AVAILABLE:
                yield await _emit_agent_event(
                    session_id,
                    _agent_event_payload(
                        kind="stage",
                        stage="evaluate",
                        title="Evaluating quality and next strategy",
                        detail=f"high_quality={st.high_quality} total_jobs={len(st.jobs)}",
                    ),
                )
                yield _trace("eval",
                             f"Evaluating quality ({st.high_quality} high-quality jobs)…")
                t0 = time.time()
                should_go, new_qs, strategy = await _evaluate(st, queries_used)
                dt = int((time.time() - t0) * 1000)

                yield _trace("eval",
                             f"{'Continue' if should_go else 'Satisfied'}: {strategy[:90]}",
                             should_continue=should_go,
                             new_queries=new_qs,
                             tokens=st.total_tokens)
                yield await _emit_agent_event(
                    session_id,
                    _agent_event_payload(
                        kind="decision",
                        stage="evaluate",
                        title="evaluate_and_plan result",
                        detail=_sentence(
                            str(strategy or ""),
                            (
                                "The current results are strong enough to stop."
                                if not should_go else
                                f"I want to continue and branch into {len(new_qs or [])} refined follow-up searches."
                            ),
                        )[:480],
                        tool={
                            "name": "evaluate_and_plan",
                            "input_summary": f"jobs={len(st.jobs)} high_quality={st.high_quality}",
                            "output_summary": "continue" if should_go else "satisfied",
                            "status": "ok",
                            "duration_ms": dt,
                        },
                    ),
                )

                if should_go and new_qs:
                    yield _trace("plan",
                                 f"Adding {len(new_qs)} refined queries",
                                 queries=new_qs)
                    pending_queries = new_qs + pending_queries
                elif not should_go:
                    satisfied = True
                    break
            else:
                if st.high_quality >= MIN_QUALITY_JOBS:
                    satisfied = True
                    break

    # ══════════════════════════════════════════════════════════════════════════
    # Finalize
    # ══════════════════════════════════════════════════════════════════════════

    if st.jobs:
        feedback = await get_feedback_profile(session_id)
        rank_jobs(st.jobs, profile, preferences, feedback)

    await update_session(session_id, "done", len(st.jobs), st.pages_crawled)

    yield await _emit_agent_event(
        session_id,
        _agent_event_payload(
            kind="stage",
            stage="done",
            title="Session finished",
            detail=f"jobs={len(st.jobs)} pages={st.pages_crawled}",
        ),
    )
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
