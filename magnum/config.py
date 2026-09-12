"""
Configuration management for Magnum AI Assistant.
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
    """Global configuration settings for Magnum."""

    # Identity
    assistant_name: str = "Magnum"
    wake_word: str = "magnum"

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
            "DEFAULT_REASONING_MODEL", "nvidia/nemotron-3-super-120b-a12b"
        )
    )

    # Execution Mode
    default_execution_mode: Literal["headless", "desktop"] = Field(
        default_factory=lambda: os.getenv("DEFAULT_EXECUTION_MODE", "desktop")  # type: ignore
    )

    # Browser storage
    browser_profile_path: Path = Field(
        default_factory=lambda: Path(
            os.getenv("BROWSER_PROFILE_PATH", "./browser_profile")
        ).resolve()
    )

    # HITL UI
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

    # Voice settings
    voice_enabled: bool = Field(
        default_factory=lambda: os.getenv("VOICE_ENABLED", "false").lower()
        in ("1", "true", "yes")
    )
    tts_engine: str = Field(
        default_factory=lambda: os.getenv("TTS_ENGINE", "edge-tts")
    )
    voice_silence_timeout: float = Field(
        default_factory=lambda: float(os.getenv("VOICE_SILENCE_TIMEOUT", "3.0"))
    )

    # Daemon settings
    daemon_pid_file: Path = Field(
        default_factory=lambda: Path(
            os.getenv("DAEMON_PID_FILE", "/tmp/magnum.pid")
        )
    )

    # Watcher settings
    watcher_poll_interval: float = Field(
        default_factory=lambda: float(os.getenv("WATCHER_POLL_INTERVAL", "5.0"))
    )

    # Autopilot settings (Antigravity autonomous controller)
    autopilot_poll_interval: float = Field(
        default_factory=lambda: float(os.getenv("AUTOPILOT_POLL_INTERVAL", "3.0"))
    )
    autopilot_mode: str = Field(
        default_factory=lambda: os.getenv("AUTOPILOT_MODE", "full_auto")
    )
    autopilot_step_delay: float = Field(
        default_factory=lambda: float(os.getenv("AUTOPILOT_STEP_DELAY", "2.0"))
    )
    autopilot_max_wait_thinking: int = Field(
        default_factory=lambda: int(os.getenv("AUTOPILOT_MAX_WAIT_THINKING", "300"))
    )
    autopilot_require_confirm: bool = Field(
        default_factory=lambda: os.getenv("AUTOPILOT_REQUIRE_CONFIRM", "false").lower()
        in ("1", "true", "yes")
    )

    # Sentinel settings (Universal App Monitor & Suggestion Engine)
    sentinel_poll_interval: float = Field(
        default_factory=lambda: float(os.getenv("SENTINEL_POLL_INTERVAL", "10.0"))
    )
    sentinel_reasoning_model: str = Field(
        default_factory=lambda: os.getenv(
            "SENTINEL_REASONING_MODEL", "nvidia/nemotron-3-super-120b-a12b"
        )
    )
    suggestion_interval: float = Field(
        default_factory=lambda: float(os.getenv("SUGGESTION_INTERVAL", "45.0"))
    )
    activity_log_interval: float = Field(
        default_factory=lambda: float(os.getenv("ACTIVITY_LOG_INTERVAL", "30.0"))
    )

    # Navigation & timing
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
