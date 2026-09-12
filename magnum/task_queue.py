"""
Magnum Concurrent Task Queue Engine.
Supports background watchers (persistent screen monitors) and foreground tasks
(one-shot goals) running simultaneously via asyncio.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from PIL import Image

import numpy as np
import subprocess

from magnum.config import config
from magnum.intelligence.grounding import ScreenGrounder, TextElement

logger = logging.getLogger(__name__)


def compute_screen_diff(img1: Image.Image, img2: Image.Image) -> float:
    """Fast perceptual difference ratio between two images (0.0 to 1.0)."""
    try:
        t1 = np.array(img1.resize((160, 100), Image.Resampling.BILINEAR).convert("L"), dtype=np.int16)
        t2 = np.array(img2.resize((160, 100), Image.Resampling.BILINEAR).convert("L"), dtype=np.int16)
        return float(np.mean(np.abs(t1 - t2)) / 255.0)
    except Exception:
        return 1.0


def play_system_sound(sound_name: str) -> None:
    """Play native macOS sound asynchronously."""
    path = f"/System/Library/Sounds/{sound_name}.aiff"
    try:
        subprocess.Popen(["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


class TaskStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    FOREGROUND = "foreground"  # One-shot goal: plan → execute → done
    WATCHER = "watcher"       # Background monitor: poll → check → act
    AUTOPILOT = "autopilot"   # Antigravity autonomous controller


class WatcherCondition(BaseModel):
    """Defines what a watcher looks for and what it does."""
    watch_for: str = Field(description="Text to look for on screen (e.g. 'Submit', 'Continue')")
    action: str = Field(default="CLICK", description="Action when found: CLICK, NOTIFY")
    stop_when: Optional[str] = Field(default=None, description="Text that signals watcher should stop (e.g. 'Complete', 'Finished')")
    poll_interval: float = Field(default=5.0, description="Seconds between screen checks")
    region: Optional[str] = Field(default=None, description="Spatial screen region constraint (e.g. 'bottom_right')")


class MagnumTask(BaseModel):
    """Represents a single task (foreground or watcher)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: TaskType = TaskType.FOREGROUND
    instruction: str = ""
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None
    result: Optional[str] = None
    
    # Watcher-specific
    watcher_condition: Optional[WatcherCondition] = None
    watcher_trigger_count: int = 0


class TaskQueue:
    """
    Concurrent task scheduler for Magnum.
    
    Manages background watchers and foreground tasks:
    - Watchers run as asyncio tasks, periodically checking the screen
    - Foreground tasks get exclusive screen control (watchers pause clicks)
    - Watchers resume when foreground completes
    """

    def __init__(self) -> None:
        self.tasks: Dict[str, MagnumTask] = {}
        self._watcher_tasks: Dict[str, asyncio.Task] = {}
        self._foreground_lock = asyncio.Lock()
        self._watcher_pause_event = asyncio.Event()
        self._watcher_pause_event.set()  # Start unpaused
        self._on_speak: Optional[Callable] = None  # TTS callback
        self._on_notify: Optional[Callable] = None  # Notification callback

    def set_callbacks(
        self,
        on_speak: Optional[Callable] = None,
        on_notify: Optional[Callable] = None,
    ) -> None:
        """Set callbacks for voice output and notifications."""
        self._on_speak = on_speak
        self._on_notify = on_notify

    def _speak(self, text: str) -> None:
        """Speak text if TTS callback is set."""
        if self._on_speak:
            try:
                self._on_speak(text)
            except Exception:
                pass
        logger.info(f"🔊 Magnum: {text}")

    def _notify(self, title: str, message: str) -> None:
        """Send native macOS banner notification."""
        try:
            from magnum.notifications import send_notification
            send_notification(title, message)
        except Exception:
            pass
        if self._on_notify:
            try:
                self._on_notify(title, message)
            except Exception:
                pass

    # ── Foreground Tasks ──

    def create_foreground_task(self, instruction: str) -> MagnumTask:
        """Create a new foreground task."""
        task = MagnumTask(
            type=TaskType.FOREGROUND,
            instruction=instruction,
            status=TaskStatus.PENDING,
        )
        self.tasks[task.id] = task
        logger.info(f"Created foreground task [{task.id}]: {instruction}")
        return task

    async def run_foreground(self, task: MagnumTask, execute_fn: Callable) -> bool:
        """
        Run a foreground task with exclusive screen control.
        Pauses all watchers during execution, resumes after.
        """
        task.status = TaskStatus.ACTIVE

        # Pause watchers
        active_watchers = [tid for tid, t in self.tasks.items() if t.type == TaskType.WATCHER and t.status == TaskStatus.ACTIVE]
        if active_watchers:
            logger.info(f"Pausing {len(active_watchers)} background watchers for foreground task")
            self._watcher_pause_event.clear()

        try:
            async with self._foreground_lock:
                result = await execute_fn(task.instruction)
                task.status = TaskStatus.DONE
                task.completed_at = time.time()
                task.result = "Success" if result else "Failed"
                return result
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.result = str(e)
            logger.error(f"Foreground task [{task.id}] failed: {e}")
            return False
        finally:
            # Resume watchers
            if active_watchers:
                self._watcher_pause_event.set()
                logger.info(f"Resumed {len(active_watchers)} background watchers")

    # ── Background Watchers ──

    def create_watcher(
        self,
        instruction: str,
        watch_for: str,
        action: str = "CLICK",
        stop_when: Optional[str] = None,
        poll_interval: float = 5.0,
        region: Optional[str] = None,
    ) -> MagnumTask:
        """Create a background watcher task."""
        condition = WatcherCondition(
            watch_for=watch_for,
            action=action,
            stop_when=stop_when,
            poll_interval=poll_interval,
            region=region,
        )
        task = MagnumTask(
            type=TaskType.WATCHER,
            instruction=instruction,
            status=TaskStatus.PENDING,
            watcher_condition=condition,
        )
        self.tasks[task.id] = task
        logger.info(f"Created watcher [{task.id}]: watch for '{watch_for}', action={action}, region='{region}', stop when '{stop_when}'")
        return task

    async def start_watcher(
        self,
        task: MagnumTask,
        screenshot_fn: Callable,
        click_fn: Callable,
    ) -> None:
        """Start a watcher as a background asyncio task."""
        if not task.watcher_condition:
            return

        task.status = TaskStatus.ACTIVE

        async def _watcher_loop():
            cond = task.watcher_condition
            logger.info(f"Watcher [{task.id}] started: watching for '{cond.watch_for}'")
            last_screenshot: Optional[Image.Image] = None
            last_elements: List[TextElement] = []

            scan_count = 0
            while task.status == TaskStatus.ACTIVE:
                try:
                    # Wait if watchers are paused (foreground task running)
                    await self._watcher_pause_event.wait()

                    # Check if we should stop
                    if task.status != TaskStatus.ACTIVE:
                        break

                    # Take screenshot and extract screen elements
                    screenshot = await screenshot_fn()
                    scan_count += 1
                    elements = ScreenGrounder.extract_screen_text_elements(screenshot)
                    last_screenshot = screenshot
                    last_elements = elements

                    try:
                        from rich.console import Console
                        region_info = f", region: {cond.region.replace('_', ' ').title()}" if cond.region else ""
                        Console().print(f"[dim]📸 Watcher [{task.id[:8]}]: Poll #{scan_count} ({cond.poll_interval}s) — scanned {len(elements)} elements for '{cond.watch_for}'{region_info}...[/dim]")
                    except Exception:
                        pass

                    # Check stop condition first
                    if cond.stop_when:
                        for el in elements:
                            el_text = el.text.strip().lower()
                            if cond.stop_when.lower() in el_text:
                                task.status = TaskStatus.DONE
                                task.completed_at = time.time()
                                task.result = f"Stop condition met: found '{el.text.strip()}'"
                                logger.info(f"Watcher [{task.id}] stopped: found '{cond.stop_when}'")
                                play_system_sound("Glass")
                                self._speak(f"Pranav, the task is finished. I found '{cond.stop_when}' on screen. Stopping the watcher.")
                                self._notify("Watcher Complete", f"Found '{cond.stop_when}' — watcher stopped.")
                                return

                    # Check watch condition with spatial region constraint
                    coords = ScreenGrounder.find_element_by_text(screenshot, cond.watch_for, region=cond.region)
                    if coords:
                        x, y = coords
                        logger.info(f"Watcher [{task.id}] triggered: found '{cond.watch_for}' at ({x:.0f}, {y:.0f})")

                        if cond.action.upper() == "CLICK":
                            # Wait for foreground lock before clicking
                            await self._watcher_pause_event.wait()
                            if task.status == TaskStatus.ACTIVE:
                                await click_fn(x, y)
                                play_system_sound("Tink")
                                task.watcher_trigger_count += 1
                                logger.info(f"Watcher [{task.id}] clicked '{cond.watch_for}' (trigger #{task.watcher_trigger_count})")
                                try:
                                    from rich.console import Console
                                    Console().print(f"\n[bold green]🎯 Watcher [{task.id[:8]}]: Found '{cond.watch_for}' at ({x:.0f}, {y:.0f}) — Clicked! (trigger #{task.watcher_trigger_count})[/bold green]")
                                except Exception:
                                    pass
                                self._notify("Watcher Action", f"Clicked '{cond.watch_for}' at ({x:.0f}, {y:.0f})")
                                # Invalidate cache after click since screen state changed
                                last_screenshot = None
                                last_elements = []
                        elif cond.action.upper() == "NOTIFY":
                            play_system_sound("Ping")
                            self._notify("Watcher Alert", f"'{cond.watch_for}' appeared on screen!")
                            self._speak(f"Pranav, '{cond.watch_for}' has appeared on screen.")

                    # Sleep before next poll
                    await asyncio.sleep(cond.poll_interval)

                except asyncio.CancelledError:
                    task.status = TaskStatus.CANCELLED
                    logger.info(f"Watcher [{task.id}] cancelled")
                    return
                except Exception as e:
                    logger.warning(f"Watcher [{task.id}] error: {e}")
                    await asyncio.sleep(cond.poll_interval)

        # Launch as asyncio background task
        asyncio_task = asyncio.create_task(_watcher_loop())
        self._watcher_tasks[task.id] = asyncio_task

    def cancel_watcher(self, task_id: str) -> bool:
        """Cancel a running watcher."""
        if task_id in self._watcher_tasks:
            self._watcher_tasks[task_id].cancel()
            if task_id in self.tasks:
                self.tasks[task_id].status = TaskStatus.CANCELLED
            play_system_sound("Basso")
            logger.info(f"Watcher [{task_id}] cancelled by user")
            return True
        return False

    def cancel_all_watchers(self) -> int:
        """Cancel all running watchers."""
        count = 0
        for task_id in list(self._watcher_tasks.keys()):
            if self.cancel_watcher(task_id):
                count += 1
        return count

    # ── Autopilot Management ──

    def create_autopilot_task(self, instruction: str = "Antigravity Autopilot") -> MagnumTask:
        """Create an autopilot background task."""
        task = MagnumTask(
            type=TaskType.AUTOPILOT,
            instruction=instruction,
            status=TaskStatus.PENDING,
        )
        self.tasks[task.id] = task
        logger.info(f"Created autopilot task [{task.id}]")
        return task

    def get_active_autopilot(self) -> Optional[MagnumTask]:
        """Get the currently active autopilot task."""
        for t in self.tasks.values():
            if t.type == TaskType.AUTOPILOT and t.status == TaskStatus.ACTIVE:
                return t
        return None

    def cancel_autopilot(self) -> bool:
        """Cancel any active autopilot task."""
        for task_id, t in list(self.tasks.items()):
            if t.type == TaskType.AUTOPILOT and t.status == TaskStatus.ACTIVE:
                t.status = TaskStatus.CANCELLED
                # Cancel the asyncio task if tracked
                if task_id in self._watcher_tasks:
                    self._watcher_tasks[task_id].cancel()
                logger.info(f"Autopilot task [{task_id}] cancelled")
                return True
        return False

    # ── Status & Queries ──

    def get_active_watchers(self) -> List[MagnumTask]:
        """Get all active background watchers."""
        return [t for t in self.tasks.values() if t.type == TaskType.WATCHER and t.status == TaskStatus.ACTIVE]

    def get_active_foreground(self) -> Optional[MagnumTask]:
        """Get the currently active foreground task."""
        for t in self.tasks.values():
            if t.type == TaskType.FOREGROUND and t.status == TaskStatus.ACTIVE:
                return t
        return None

    def get_all_tasks(self) -> List[MagnumTask]:
        """Get all tasks sorted by creation time."""
        return sorted(self.tasks.values(), key=lambda t: t.created_at, reverse=True)

    def get_status_summary(self) -> str:
        """Get a human-readable summary of all tasks."""
        watchers = self.get_active_watchers()
        fg = self.get_active_foreground()
        autopilot = self.get_active_autopilot()
        lines = []

        if autopilot:
            lines.append(f"🤖 Autopilot: ACTIVE [{autopilot.id}]")

        if fg:
            lines.append(f"🎯 Active Task: {fg.instruction}")
        
        if watchers:
            lines.append(f"👁️ Active Watchers ({len(watchers)}):")
            for w in watchers:
                cond = w.watcher_condition
                lines.append(f"  🔄 [{w.id}] Watching for '{cond.watch_for}' (triggered {w.watcher_trigger_count}x)")
        
        done = [t for t in self.tasks.values() if t.status == TaskStatus.DONE]
        if done:
            lines.append(f"✅ Completed: {len(done)} tasks")

        return "\n".join(lines) if lines else "No active tasks."
