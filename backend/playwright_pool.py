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