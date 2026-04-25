import unittest


class TestAgentsHeuristicFallback(unittest.TestCase):
    def test_llm_unavailable_uses_smart_heuristic_queries(self):
        # Import inside the test so we can safely mutate module globals.
        import os
        import sys
        import types

        backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)

        # The repo's runtime deps (httpx/bs4/etc) may not be installed in minimal CI.
        # Stub the modules we don't need for this unit test so importing `agents` works.
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
                async def commit(self): return None
                async def fetchall(self): return []
            async def connect(*a, **k):  # pragma: no cover
                return _Conn()
            aiosqlite.connect = connect
            sys.modules["aiosqlite"] = aiosqlite

        import agents

        agents._LLM_AVAILABLE = False

        query = "Senior backend engineer Python AWS remote"
        prefs = {"location": "Remote", "experience_level": "senior"}
        parsed = agents.preprocess_query(query, prefs)

        queries = agents._smart_heuristic_queries(query, prefs, parsed)
        self.assertIsInstance(queries, list)
        self.assertTrue(len(queries) > 0)
