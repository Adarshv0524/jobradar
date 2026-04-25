"""
Selenium browser pool (fallback) for JS-heavy sites.

Notes:
  - Selenium is synchronous; we wrap driver usage with asyncio.to_thread.
  - This is a best-effort fallback behind Playwright.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from config import (
    SELENIUM_CHROME_BINARY,
    SELENIUM_DRIVER_PATH,
    SELENIUM_POOL_SIZE,
    SELENIUM_TIMEOUT_MS,
    USE_SELENIUM,
)

log = logging.getLogger(__name__)

_selenium_available = False

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    _selenium_available = True
except Exception:
    _selenium_available = False


class SeleniumPool:
    def __init__(self, size: int):
        self._size = max(1, int(size))
        self._sem = asyncio.Semaphore(self._size)
        self._drivers: list = []
        self._ready = False

    def _create_driver(self):
        opts = ChromeOptions()
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)
        if SELENIUM_CHROME_BINARY:
            opts.binary_location = SELENIUM_CHROME_BINARY

        service = ChromeService(executable_path=SELENIUM_DRIVER_PATH) if SELENIUM_DRIVER_PATH else None
        driver = webdriver.Chrome(service=service, options=opts) if service else webdriver.Chrome(options=opts)
        try:
            driver.set_page_load_timeout(max(1, int(SELENIUM_TIMEOUT_MS // 1000)))
        except Exception:
            pass
        return driver

    async def start(self):
        if not USE_SELENIUM or not _selenium_available:
            return
        try:
            # Create drivers in background threads (can be slow).
            drivers = await asyncio.gather(
                *[asyncio.to_thread(self._create_driver) for _ in range(self._size)],
                return_exceptions=True,
            )
            self._drivers = [d for d in drivers if not isinstance(d, Exception)]
            self._ready = len(self._drivers) > 0
            if self._ready:
                log.info("Selenium pool ready (size=%d)", len(self._drivers))
            else:
                log.warning("Selenium pool failed to start (no drivers)")
        except Exception as e:
            log.warning("Selenium pool failed to start: %s", e)
            self._ready = False

    async def stop(self):
        if not self._drivers:
            return
        await asyncio.gather(
            *[asyncio.to_thread(self._safe_quit, d) for d in self._drivers],
            return_exceptions=True,
        )
        self._drivers = []
        self._ready = False

    def _safe_quit(self, driver):
        try:
            driver.quit()
        except Exception:
            pass

    @asynccontextmanager
    async def get_driver(self) -> AsyncGenerator[Optional[object], None]:
        if not self._ready or not self._drivers:
            yield None
            return

        async with self._sem:
            driver = None
            try:
                driver = self._drivers.pop() if self._drivers else None
                yield driver
            finally:
                if driver is not None:
                    self._drivers.append(driver)


async def fetch_with_selenium(url: str, wait_selector: str = "body") -> Optional[str]:
    global _pool
    if _pool is None or not _pool._ready:
        return None

    async with _pool.get_driver() as driver:
        if driver is None:
            return None

        def _do_fetch() -> Optional[str]:
            try:
                driver.get(url)
                try:
                    WebDriverWait(driver, max(1, int(SELENIUM_TIMEOUT_MS // 1000))).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector))
                    )
                except Exception:
                    pass
                # Scroll to trigger lazy loading
                try:
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
                except Exception:
                    pass
                return driver.page_source
            except Exception:
                return None

        return await asyncio.to_thread(_do_fetch)


_pool: Optional[SeleniumPool] = None


async def init_selenium_pool(size: int = 1) -> None:
    global _pool
    if not USE_SELENIUM or not _selenium_available:
        return
    _pool = SeleniumPool(size=size)
    await _pool.start()


async def close_selenium_pool() -> None:
    global _pool
    if _pool:
        await _pool.stop()
        _pool = None

