import asyncio
import unittest


def _ensure_backend_on_path():
    import os
    import sys
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)


def _stub_minimal_runtime_deps():
    # Keep these stubs consistent with the repo's existing lightweight tests.
    import sys
    import types

    if "httpx" not in sys.modules:
        httpx = types.ModuleType("httpx")
        class _E(Exception): ...
        httpx.TimeoutException = _E
        httpx.ConnectError = _E
        class _AsyncClient:
            def __init__(self, *a, **k): ...
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def get(self, *a, **k):
                raise _E("stub")
        httpx.AsyncClient = _AsyncClient
        sys.modules["httpx"] = httpx

    if "bs4" not in sys.modules:
        bs4 = types.ModuleType("bs4")
        class BeautifulSoup:  # minimal stub
            def __init__(self, *a, **k): ...
            def get_text(self, *a, **k): return ""
            def find_all(self, *a, **k): return []
            def select(self, *a, **k): return []
            def select_one(self, *a, **k): return None
        bs4.BeautifulSoup = BeautifulSoup
        sys.modules["bs4"] = bs4

    if "aiosqlite" not in sys.modules:
        aiosqlite = types.ModuleType("aiosqlite")
        class _Conn:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def execute(self, *a, **k): raise RuntimeError("stub")
            async def executescript(self, *a, **k): raise RuntimeError("stub")
            async def commit(self): return None
        async def connect(*a, **k):  # pragma: no cover
            return _Conn()
        aiosqlite.connect = connect
        aiosqlite.Row = object
        sys.modules["aiosqlite"] = aiosqlite


class TestAgentEvents(unittest.TestCase):
    def test_emits_agent_event_in_heuristic_mode(self):
        _ensure_backend_on_path()
        _stub_minimal_runtime_deps()

        import importlib
        import agents
        agents = importlib.reload(agents)

        # Force small, deterministic run.
        agents._LLM_AVAILABLE = False
        agents.USE_FREE_APIS = False
        agents.JOB_SITES = {}
        agents.SITE_SCRAPER_MAP = {}
        agents.MAX_QUERIES_PER_SESSION = 1
        agents.MAX_URLS_PER_QUERY = 1
        agents.MAX_PAGES_TOTAL = 1

        async def _noop(*a, **k): return None
        async def _false(*a, **k): return False
        async def _empty_list(*a, **k): return []
        async def _empty_dict(*a, **k): return {}

        agents.insert_agent_event = _noop
        agents.insert_crawl_plan_entry = _noop
        agents.update_crawl_plan = _noop
        agents.update_source = _noop
        agents.update_session = _noop
        agents.mark_visited = _noop
        agents.url_visited = _false
        agents.get_feedback_profile = _empty_dict
        agents.fetch_html = _noop
        agents.search_web = _empty_list

        async def _extract_jobs_from_url(url: str):
            return [], "failed", {"engine": "httpx", "rendered": False}
        agents.extract_jobs_from_url = _extract_jobs_from_url

        agents._smart_heuristic_queries = lambda *a, **k: ["stub query"]
        agents._expand_career_candidates = lambda urls: []

        async def _status(*a, **k): return "running"
        agents.get_session_status = _status

        async def collect():
            out = []
            async for ev in agents.run_search_session(
                session_id="test",
                query="backend engineer",
                preferences={"watch_mode": False},
                profile={},
            ):
                out.append(ev)
                if ev.get("type") == "done":
                    break
            return out

        events = asyncio.run(collect())
        agent_events = [e for e in events if e.get("type") == "agent_event"]
        self.assertTrue(len(agent_events) > 0)
        payload = agent_events[0].get("data") or {}
        for k in ("id", "ts", "kind", "stage", "title"):
            self.assertIn(k, payload)
        importlib.reload(agents)

    def test_watch_mode_emits_wait_and_respects_stop(self):
        _ensure_backend_on_path()
        _stub_minimal_runtime_deps()

        import importlib
        import agents
        agents = importlib.reload(agents)

        agents._LLM_AVAILABLE = False
        agents.USE_FREE_APIS = False
        agents.JOB_SITES = {}
        agents.SITE_SCRAPER_MAP = {}
        agents.MAX_QUERIES_PER_SESSION = 1
        agents.WATCH_HEARTBEAT_SEC = 0.01

        async def _noop(*a, **k): return None
        async def _empty_dict(*a, **k): return {}
        agents.insert_agent_event = _noop
        agents.insert_crawl_plan_entry = _noop
        agents.update_crawl_plan = _noop
        agents.update_source = _noop
        agents.update_session = _noop
        agents.get_feedback_profile = _empty_dict

        agents._smart_heuristic_queries = lambda *a, **k: []

        calls = {"n": 0}
        async def _status(*a, **k):
            calls["n"] += 1
            return "running" if calls["n"] <= 1 else "stopped"
        agents.get_session_status = _status

        async def collect():
            out = []
            async for ev in agents.run_search_session(
                session_id="test-watch",
                query="anything",
                preferences={"watch_mode": True, "watch_interval_sec": 1, "watch_max_cycles": 1},
                profile={},
            ):
                out.append(ev)
                # Stop quickly once the agent acknowledges stop.
                if ev.get("type") == "agent_event" and (ev.get("data") or {}).get("kind") == "error":
                    break
            return out

        events = asyncio.run(collect())
        wait_events = [
            e for e in events
            if e.get("type") == "agent_event" and (e.get("data") or {}).get("kind") == "wait"
        ]
        self.assertTrue(len(wait_events) > 0)
        importlib.reload(agents)
