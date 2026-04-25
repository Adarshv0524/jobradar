"""
Rendered fetch helper (Playwright → Selenium → None).

We keep Playwright as the primary engine (fast + async), and Selenium as an
optional fallback for cases where Playwright can't start or fails per-URL.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from config import USE_PLAYWRIGHT, USE_SELENIUM, PLAYWRIGHT_TIMEOUT, SELENIUM_TIMEOUT_MS

log = logging.getLogger(__name__)


async def fetch_rendered_with_engine(url: str, wait_selector: str = "body") -> Tuple[Optional[str], str]:
    # Primary: Playwright pool
    if USE_PLAYWRIGHT:
        try:
            from playwright_pool import fetch_with_playwright
            html = await fetch_with_playwright(url, wait_selector=wait_selector, timeout=PLAYWRIGHT_TIMEOUT)
            if html:
                return html, "playwright"
        except Exception as e:
            log.debug("Playwright rendered fetch failed for %s: %s", url, e)

    # Fallback: Selenium pool
    if USE_SELENIUM:
        try:
            from selenium_pool import fetch_with_selenium
            html = await fetch_with_selenium(url, wait_selector=wait_selector)
            if html:
                return html, "selenium"
        except Exception as e:
            log.debug("Selenium rendered fetch failed for %s: %s", url, e)

    return None, "none"


async def fetch_rendered(url: str, wait_selector: str = "body") -> Optional[str]:
    html, _engine = await fetch_rendered_with_engine(url, wait_selector=wait_selector)
    return html
