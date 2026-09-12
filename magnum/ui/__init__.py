"""
Overlay Controller for Auto-Navigator.
Manages native macOS HUD border, plan checklist, and bottom-center interactive question card.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import subprocess
import logging
import threading
import uuid
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class OverlayController:
    """Manages the visual screen overlay subprocess and bidirectional IPC."""

    def __init__(self) -> None:
        self._process: Optional[subprocess.Popen] = None
        self._pending_questions: Dict[str, asyncio.Future] = {}
        self._reader_thread: Optional[threading.Thread] = None

    def start(self, initial_status: str = "AUTO-NAVIGATOR ACTIVE") -> None:
        """Start the screen overlay subprocess."""
        if self._process is not None and self._process.poll() is None:
            return

        try:
            overlay_script = os.path.join(os.path.dirname(__file__), "overlay.py")
            self._process = subprocess.Popen(
                [sys.executable, overlay_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            self._start_stdout_listener()
            self.set_status(initial_status)
            logger.info("Screen overlay HUD started.")
        except Exception as e:
            logger.warning(f"Could not start screen overlay: {e}")

    def _start_stdout_listener(self) -> None:
        """Listen for user responses from the overlay card via stdout."""
        def reader():
            if not self._process or not self._process.stdout:
                return
            for line in self._process.stdout:
                clean = line.strip()
                if clean.startswith("ANSWER:"):
                    parts = clean.split(":", 2)
                    if len(parts) >= 3:
                        q_id = parts[1]
                        ans = parts[2]
                        fut = self._pending_questions.pop(q_id, None)
                        if fut and not fut.done():
                            fut.get_loop().call_soon_threadsafe(fut.set_result, ans)

        self._reader_thread = threading.Thread(target=reader, daemon=True)
        self._reader_thread.start()

    def set_status(self, text: str) -> None:
        """Update top HUD badge status text."""
        if self._process and self._process.stdin and self._process.poll() is None:
            try:
                self._process.stdin.write(f"STATUS {text}\n")
                self._process.stdin.flush()
            except Exception as e:
                logger.debug(f"Failed to update overlay status: {e}")

    def update_plan(
        self, steps: List[str], active_idx: int = 1, completed: Optional[List[int]] = None
    ) -> None:
        """Update the on-screen Plan Checklist."""
        if self._process and self._process.stdin and self._process.poll() is None:
            try:
                payload = json.dumps({
                    "steps": steps,
                    "active_idx": active_idx,
                    "completed": completed or [],
                })
                self._process.stdin.write(f"PLAN {payload}\n")
                self._process.stdin.flush()
            except Exception as e:
                logger.debug(f"Failed to update plan on overlay: {e}")

    # Backward compatibility alias
    set_plan = update_plan

    def set_watchers(self, watchers: List[str]) -> None:
        """Update active background watchers on HUD overlay."""
        if self._process and self._process.stdin and self._process.poll() is None:
            try:
                payload = json.dumps({"watchers": watchers})
                self._process.stdin.write(f"WATCHERS {payload}\n")
                self._process.stdin.flush()
            except Exception as e:
                logger.debug(f"Failed to update watchers on overlay: {e}")

    def set_autopilot_state(self, state_data: dict) -> None:
        """Update the Antigravity Autopilot monitor panel on HUD overlay."""
        if self._process and self._process.stdin and self._process.poll() is None:
            try:
                payload = json.dumps(state_data)
                self._process.stdin.write(f"AUTOPILOT_STATE {payload}\n")
                self._process.stdin.flush()
            except Exception as e:
                logger.debug(f"Failed to update autopilot state on overlay: {e}")

    def add_autopilot_log(self, action: str) -> None:
        """Add an action entry to the Autopilot log ticker on overlay."""
        if self._process and self._process.stdin and self._process.poll() is None:
            try:
                payload = json.dumps({"action": action})
                self._process.stdin.write(f"AUTOPILOT_LOG {payload}\n")
                self._process.stdin.flush()
            except Exception as e:
                logger.debug(f"Failed to add autopilot log on overlay: {e}")

    async def ask_question_card(
        self, prompt: str, options: Optional[List[str]] = None, timeout: float = 60.0
    ) -> str:
        """Show the bottom-center interactive card and wait for the user's choice."""
        q_id = str(uuid.uuid4())[:8]
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._pending_questions[q_id] = fut

        if self._process and self._process.stdin and self._process.poll() is None:
            try:
                payload = json.dumps({
                    "id": q_id,
                    "prompt": prompt,
                    "options": options or ["Approve", "Cancel"],
                })
                self._process.stdin.write(f"ASK_QUESTION {payload}\n")
                self._process.stdin.flush()
            except Exception as e:
                logger.warning(f"Failed to send question to overlay: {e}")
                return "Approve"

        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self.hide_question_card()
            return "Approve"

    def hide_question_card(self) -> None:
        """Hide the bottom-center interactive card."""
        if self._process and self._process.stdin and self._process.poll() is None:
            try:
                self._process.stdin.write("HIDE_QUESTION\n")
                self._process.stdin.flush()
            except Exception:
                pass

    def flash(self) -> None:
        """Trigger camera flash and shutter audio animation."""
        if self._process and self._process.stdin and self._process.poll() is None:
            try:
                self._process.stdin.write("FLASH\n")
                self._process.stdin.flush()
            except Exception as e:
                logger.debug(f"Failed to send flash command: {e}")

    def stop(self) -> None:
        """Terminate the screen overlay."""
        if self._process and self._process.poll() is None:
            try:
                if self._process.stdin:
                    self._process.stdin.write("QUIT\n")
                    self._process.stdin.flush()
                self._process.terminate()
            except Exception:
                pass
            finally:
                self._process = None
                logger.info("Screen overlay HUD stopped.")


# Global overlay singleton
overlay_controller = OverlayController()


def get_overlay() -> OverlayController:
    return overlay_controller
