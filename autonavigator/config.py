"""
Configuration management for Auto-Navigator.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env if present
load_dotenv()


class AppConfig(BaseModel):
    """Global configuration settings for Auto-Navigator."""

    # NVIDIA NIM Configuration
    nvidia_api_key: str = Field(
        default_factory=lambda: os.getenv("NVIDIA_API_KEY", "")
    )
    nvidia_base_url: str = Field(
        default_factory=lambda: os.getenv(
            "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )
    )
    default_vision_model: str = Field(
        default_factory=lambda: os.getenv(
            "DEFAULT_VISION_MODEL", "meta/llama-3.2-11b-vision-instruct"
        )
    )
    default_reasoning_model: str = Field(
        default_factory=lambda: os.getenv(
            "DEFAULT_REASONING_MODEL", "openai/gpt-oss-120b"
        )
    )

    # Execution Mode: "desktop" (live visual macOS desktop control) or "headless"
    default_execution_mode: Literal["headless", "desktop"] = Field(
        default_factory=lambda: os.getenv("DEFAULT_EXECUTION_MODE", "desktop")  # type: ignore
    )

    # Browser storage for persistent sessions (e.g. keep LinkedIn/GitHub login)
    browser_profile_path: Path = Field(
        default_factory=lambda: Path(
            os.getenv("BROWSER_PROFILE_PATH", "./browser_profile")
        ).resolve()
    )

    # Human-In-The-Loop UI
    hitl_interface: Literal["cli", "native"] = Field(
        default_factory=lambda: os.getenv("HITL_INTERFACE", "cli")  # type: ignore
    )

    # Debug & Screenshots
    save_debug_screenshots: bool = Field(
        default_factory=lambda: os.getenv("SAVE_DEBUG_SCREENSHOTS", "true").lower()
        in ("1", "true", "yes")
    )
    screenshots_dir: Path = Field(
        default_factory=lambda: Path(
            os.getenv("SCREENSHOTS_DIR", "./screenshots")
        ).resolve()
    )

    # Navigation & timing parameters
    viewport_width: int = 1280
    viewport_height: int = 800
    step_delay: float = 1.0
    max_steps: int = 30

    def ensure_directories(self) -> None:
        """Ensure necessary runtime directories exist."""
        self.browser_profile_path.mkdir(parents=True, exist_ok=True)
        if self.save_debug_screenshots:
            self.screenshots_dir.mkdir(parents=True, exist_ok=True)


# Global singleton instance
config = AppConfig()
