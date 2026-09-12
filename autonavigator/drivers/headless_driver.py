"""
Headless Playwright Driver for non-intrusive background browser automation.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional
from PIL import Image
import pyperclip
from playwright.async_api import async_playwright, BrowserContext, Page, Playwright

from autonavigator.config import config
from autonavigator.drivers.base import BaseDriver
from autonavigator.ui import get_overlay

logger = logging.getLogger(__name__)


class HeadlessDriver(BaseDriver):
    """Playwright-based background browser driver that runs without disturbing the user."""

    def __init__(
        self,
        headless: bool = True,
        user_data_dir: Optional[Path] = None,
        viewport_width: int = 1280,
        viewport_height: int = 800,
    ) -> None:
        self.headless = headless
        self.user_data_dir = user_data_dir or config.browser_profile_path
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

        self._playwright: Optional[Playwright] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._step_counter = 0
        self.overlay = get_overlay()

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("Driver has not been started. Call 'await driver.start()' first.")
        return self._page

    async def start(self) -> None:
        """Launch Playwright with persistent context so login sessions are retained."""
        config.ensure_directories()
        self.overlay.start("AUTO-NAVIGATOR ACTIVE")
        self._playwright = await async_playwright().start()

        # Launch persistent context
        self._context = await self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(self.user_data_dir),
            headless=self.headless,
            viewport={"width": self.viewport_width, "height": self.viewport_height},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            permissions=["clipboard-read", "clipboard-write"],
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )

        if self._context.pages:
            self._page = self._context.pages[0]
        else:
            self._page = await self._context.new_page()

        logger.info(f"HeadlessDriver started with persistent profile at: {self.user_data_dir}")

    async def close(self) -> None:
        """Close browser context and stop Playwright."""
        self.overlay.stop()
        if self._context:
            await self._context.close()
            self._context = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        self._page = None
        logger.info("HeadlessDriver closed.")

    async def navigate(self, target: str) -> None:
        """Navigate to URL."""
        if not target.startswith("http://") and not target.startswith("https://"):
            target = f"https://{target}"
        logger.info(f"Navigating to {target}")
        self.overlay.set_status(f"NAVIGATING: {target}")
        await self.page.goto(target, wait_until="domcontentloaded", timeout=45000)
        # Give a brief moment for dynamic elements to hydrate
        await self.page.wait_for_timeout(1500)

    async def screenshot(self, filename: Optional[str] = None) -> Image.Image:
        """Capture screenshot of the page with shutter flash animation."""
        self._step_counter += 1
        
        # Trigger visual screen flash & shutter audio animation
        self.overlay.set_status("CAPTURING SCREENSHOT")
        self.overlay.flash()
        
        png_bytes = await self.page.screenshot(type="png", full_page=False)
        image = Image.open(io.BytesIO(png_bytes))

        if config.save_debug_screenshots:
            fname = filename or f"step_{self._step_counter:03d}.png"
            out_path = config.screenshots_dir / fname
            image.save(out_path)
            logger.debug(f"Saved debug screenshot to {out_path}")

        return image

    async def click(self, x: float, y: float) -> None:
        """Click at (x, y) coordinates on the page."""
        logger.info(f"Clicking at coordinates ({x}, {y})")
        await self.page.mouse.click(x, y)
        await self.page.wait_for_timeout(500)

    async def double_click(self, x: float, y: float) -> None:
        """Double click at (x, y) coordinates."""
        await self.page.mouse.dblclick(x, y)
        await self.page.wait_for_timeout(500)

    async def right_click(self, x: float, y: float) -> None:
        """Right click at (x, y) coordinates."""
        await self.page.mouse.click(x, y, button="right")
        await self.page.wait_for_timeout(500)

    async def type_text(self, text: str, delay_ms: int = 40) -> None:
        """Type text character by character."""
        logger.info(f"Typing text: '{text}'")
        await self.page.keyboard.type(text, delay=delay_ms)
        await self.page.wait_for_timeout(300)

    async def press_key(self, key: str) -> None:
        """Press a keyboard key."""
        logger.info(f"Pressing key: {key}")
        await self.page.keyboard.press(key)
        await self.page.wait_for_timeout(500)

    async def hotkey(self, *keys: str) -> None:
        """Press key combination."""
        combined = "+".join(keys)
        logger.info(f"Pressing hotkey: {combined}")
        await self.page.keyboard.press(combined)
        await self.page.wait_for_timeout(500)

    async def scroll(self, direction: str = "down", amount: int = 300) -> None:
        """Scroll page wheel."""
        delta_y = amount if direction == "down" else -amount
        logger.info(f"Scrolling {direction} by {amount}px")
        await self.page.mouse.wheel(0, delta_y)
        await self.page.wait_for_timeout(800)

    async def get_clipboard(self) -> str:
        """Get clipboard text."""
        try:
            return pyperclip.paste()
        except Exception:
            return ""

    async def set_clipboard(self, text: str) -> None:
        """Set clipboard text."""
        try:
            pyperclip.copy(text)
            logger.info("Copied content to system clipboard.")
        except Exception as e:
            logger.error(f"Failed to set clipboard: {e}")

    async def extract_text_content(self) -> str:
        """Extract visible text on page."""
        try:
            return await self.page.evaluate("() => document.body.innerText")
        except Exception:
            return ""
