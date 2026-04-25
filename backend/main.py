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
    get_agent_events,
    get_session,
    init_db,
    insert_feedback,
    set_session_status,
    update_session,
)
from resume_parser import parse_resume_bytes
from resume_inference import resume_profile_from_text
from resume_sections import split_resume_sections, extract_project_bullets, extract_keywords


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
    from config import PLAYWRIGHT_POOL_SIZE, USE_PLAYWRIGHT, SELENIUM_POOL_SIZE, USE_SELENIUM
    if USE_PLAYWRIGHT:
        from playwright_pool import init_playwright_pool
        await init_playwright_pool(size=PLAYWRIGHT_POOL_SIZE)
        log.info("Playwright pool ready")
    if USE_SELENIUM:
        from selenium_pool import init_selenium_pool
        await init_selenium_pool(size=SELENIUM_POOL_SIZE)
        log.info("Selenium pool ready")


@app.on_event("shutdown")
async def shutdown():
    from playwright_pool import close_playwright_pool
    await close_playwright_pool()
    try:
        from selenium_pool import close_selenium_pool
        await close_selenium_pool()
    except Exception:
        pass


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
    # Watch mode (agent keeps scanning on a schedule when results are weak)
    watch_mode:         Optional[bool]      = False
    watch_interval_sec: Optional[int]       = 900
    watch_max_cycles:   Optional[int]       = 0


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
        "watch_mode":         bool(req.watch_mode),
        "watch_interval_sec": int(req.watch_interval_sec or 900),
        "watch_max_cycles":   int(req.watch_max_cycles or 0),
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
            await set_session_status(session_id, "cancelled")
        except Exception as e:
            yield {"data": json.dumps({"type": "error", "message": str(e)})}
            await set_session_status(session_id, "error")

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

    parsed = parse_resume_bytes(
        content=content,
        content_type=file.content_type or "",
        filename=file.filename or "",
    )
    profile_text = parsed.text
    sections = split_resume_sections(profile_text)
    project_bullets = extract_project_bullets(sections)
    keywords = extract_keywords((sections.get("projects") or "") + "\n" + profile_text)

    # Save to disk
    safe_name = file.filename.replace("/", "_").replace("..", "_")
    path      = RESUME_DIR / safe_name
    path.write_bytes(content)

    # Extract skills from resume text (include projects section to improve recall)
    from scraper import extract_skills
    skills   = extract_skills(profile_text + "\n" + (sections.get("projects") or ""))
    exp_meta = resume_profile_from_text(profile_text).get("experience", {})
    exp_lvl  = str(exp_meta.get("level") or "")

    # Very simple summary extraction (first 500 chars of substantial content)
    lines   = [l.strip() for l in profile_text.splitlines() if len(l.strip()) > 40]
    summary = " ".join(lines[:5])[:500]

    return {
        "skills":           skills,
        "experience_level": exp_lvl,
        "summary":          summary,
        "raw_length":       len(profile_text),
        "experience":       exp_meta,
        "sections":         sections,
        "projects":         project_bullets,
        "keywords":         keywords,
        "extraction": {
            "method": parsed.method,
            "warnings": parsed.warnings,
            "page_char_counts": parsed.page_char_counts,
        },
    }


@app.get("/api/health")
async def health():
    from agents import _LLM_AVAILABLE, OPENAI_MODEL

    return {
        "status": "ok",
        "llm_enabled": _LLM_AVAILABLE,
        "llm_model": OPENAI_MODEL if _LLM_AVAILABLE else None,
    }


class ControlRequest(BaseModel):
    action: str  # pause | resume | stop


@app.post("/api/search/{session_id}/control")
async def control_search(session_id: str, req: ControlRequest):
    s = await get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    action = (req.action or "").lower().strip()
    if action == "pause":
        await set_session_status(session_id, "paused")
    elif action == "resume":
        await set_session_status(session_id, "running")
    elif action == "stop":
        await set_session_status(session_id, "stopped")
    else:
        raise HTTPException(400, "Invalid action")
    return {"status": "ok", "session_id": session_id, "action": action}


@app.get("/api/search/{session_id}/events")
async def list_agent_events(session_id: str, after_ts: int = 0, limit: int = 200):
    s = await get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    events = await get_agent_events(session_id, after_ts=after_ts, limit=limit)
    return {"session_id": session_id, "events": events}
