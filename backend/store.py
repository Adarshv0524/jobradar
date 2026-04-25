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