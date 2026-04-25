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