"""
Base Driver Interface for Auto-Navigator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple
from PIL import Image
from pydantic import BaseModel


class DriverState(BaseModel):
    """Current state of the driver."""

    is_running: bool = False
    current_url_or_window: str = ""
    viewport_size: Tuple[int, int] = (1280, 800)


class BaseDriver(ABC):
    """Abstract driver interface that works uniformly across headless and desktop execution."""

    @abstractmethod
    async def start(self) -> None:
        """Initialize the driver environment."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up and close driver resources."""
        pass

    @abstractmethod
    async def navigate(self, target: str) -> None:
        """Navigate to URL (web) or switch to application (desktop)."""
        pass

    @abstractmethod
    async def screenshot(self, filename: Optional[str] = None) -> Image.Image:
        """Capture current screenshot as PIL Image."""
        pass

    @abstractmethod
    async def click(self, x: float, y: float) -> None:
        """Perform a single click at coordinates (x, y)."""
        pass

    @abstractmethod
    async def double_click(self, x: float, y: float) -> None:
        """Perform a double click at coordinates (x, y)."""
        pass

    @abstractmethod
    async def right_click(self, x: float, y: float) -> None:
        """Perform a right click at coordinates (x, y)."""
        pass

    @abstractmethod
    async def type_text(self, text: str, delay_ms: int = 40) -> None:
        """Type text into active focused element."""
        pass

    @abstractmethod
    async def press_key(self, key: str) -> None:
        """Press a special key (Enter, Tab, Escape, Backspace, etc.)."""
        pass

    @abstractmethod
    async def hotkey(self, *keys: str) -> None:
        """Press keyboard shortcut combination."""
        pass

    @abstractmethod
    async def scroll(self, direction: str = "down", amount: int = 300) -> None:
        """Scroll page or window up/down."""
        pass

    @abstractmethod
    async def get_clipboard(self) -> str:
        """Read system or driver clipboard text."""
        pass

    @abstractmethod
    async def set_clipboard(self, text: str) -> None:
        """Write text to system or driver clipboard."""
        pass

    @abstractmethod
    async def extract_text_content(self) -> str:
        """Extract visible text content or OCR from the current context."""
        pass
