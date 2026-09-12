"""
Antigravity Autopilot Engine.
A state-machine-driven autonomous controller that monitors Antigravity IDE,
detects its state, and takes intelligent actions to drive complex workflows.

Modes:
  MONITOR   — Watch only, report state changes (no actions)
  SEMI_AUTO — Auto-click Proceed/Submit, pause between roadmap steps
  FULL_AUTO — Complete autonomous execution: detect → act → loop

The Autopilot runs as a background asyncio task alongside watchers and
foreground tasks, with priority screen control when performing actions.
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Callable, List, Optional, Tuple

from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from magnum.config import config
from magnum.intelligence.antigravity_detector import (
    AntigravityDetector,
    AntigravityState,
    DetectionResult,
    RoadmapStep,
)

logger = logging.getLogger(__name__)
console = Console()


class AutopilotMode(str, Enum):
    """Operating mode for the Autopilot."""
    MONITOR = "monitor"        # Watch only, report state changes
    SEMI_AUTO = "semi_auto"    # Auto-click buttons, pause for roadmaps
    FULL_AUTO = "full_auto"    # Complete autonomous execution


class AutopilotStatus(str, Enum):
    """Current status of the Autopilot."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ACTING = "acting"          # Currently performing an action
    WAITING = "waiting"        # Waiting for AI to finish thinking
    EXECUTING_ROADMAP = "executing_roadmap"
    ERROR = "error"


class AutopilotAction:
    """A single action taken by the Autopilot, for logging."""

    def __init__(self, action_type: str, description: str) -> None:
        self.action_type = action_type
        self.description = description
        self.timestamp = time.time()

    def __str__(self) -> str:
        return f"[{self.action_type}] {self.description}"


class AntigravityAutopilot:
    """
    Autonomous Antigravity IDE controller.
    Continuously monitors the screen, detects Antigravity state,
    and takes actions based on the configured mode.
    """

    def __init__(
        self,
        mode: AutopilotMode = AutopilotMode.FULL_AUTO,
        screenshot_fn: Optional[Callable] = None,
        click_fn: Optional[Callable] = None,
        type_fn: Optional[Callable] = None,
        press_key_fn: Optional[Callable] = None,
        hotkey_fn: Optional[Callable] = None,
        navigate_fn: Optional[Callable] = None,
        overlay_fn: Optional[Callable] = None,
        voice_fn: Optional[Callable] = None,
    ) -> None:
        self.mode = mode
        self.detector = AntigravityDetector()
        self.status = AutopilotStatus.STOPPED

        # Driver function callbacks
        self._screenshot_fn = screenshot_fn
        self._click_fn = click_fn
        self._type_fn = type_fn
        self._press_key_fn = press_key_fn
        self._hotkey_fn = hotkey_fn
        self._navigate_fn = navigate_fn
        self._overlay_fn = overlay_fn
        self._voice_fn = voice_fn

        # State tracking
        self._last_state = AntigravityState.UNKNOWN
        self._last_detection: Optional[DetectionResult] = None
        self._action_log: List[AutopilotAction] = []
        self._scan_count = 0
        self._thinking_start_time: Optional[float] = None
        self._total_actions_taken = 0

        # Roadmap execution tracking
        self._current_roadmap: List[RoadmapStep] = []
        self._roadmap_step_index = 0
        self._roadmap_total = 0

        # Pending prompts queue (for roadmap step injection)
        self._pending_prompts: List[str] = []

        # Control
        self._stop_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Start unpaused
        self._task: Optional[asyncio.Task] = None

        # Timing
        self.poll_interval: float = config.autopilot_poll_interval
        self.step_delay: float = config.autopilot_step_delay
        self.max_wait_thinking: int = config.autopilot_max_wait_thinking

    @property
    def is_active(self) -> bool:
        return self.status not in (AutopilotStatus.STOPPED, AutopilotStatus.ERROR)

    @property
    def action_count(self) -> int:
        return self._total_actions_taken

    @property
    def recent_actions(self) -> List[AutopilotAction]:
        return self._action_log[-5:]

    def _log_action(self, action_type: str, description: str) -> None:
        """Log an action taken by the autopilot."""
        action = AutopilotAction(action_type, description)
        self._action_log.append(action)
        if len(self._action_log) > 100:
            self._action_log = self._action_log[-100:]
        self._total_actions_taken += 1
        logger.info(f"Autopilot [{action_type}]: {description}")
        console.print(f"[bold green]🤖 Autopilot [{action_type}]:[/bold green] {description}")

        # Update overlay
        if self._overlay_fn:
            try:
                self._overlay_fn("add_log", str(action))
            except Exception:
                pass

    def _speak(self, text: str) -> None:
        """Speak via voice engine if available."""
        if self._voice_fn:
            try:
                self._voice_fn(text)
            except Exception:
                pass

    def _update_overlay_state(self, state: AntigravityState, extra: str = "") -> None:
        """Push state update to overlay monitor panel."""
        if self._overlay_fn:
            try:
                self._overlay_fn("set_state", {
                    "state": state.value,
                    "state_display": self.detector.get_state_display(state),
                    "emoji": self.detector.get_state_emoji(state),
                    "mode": self.mode.value,
                    "status": self.status.value,
                    "scan_count": self._scan_count,
                    "actions_taken": self._total_actions_taken,
                    "extra": extra,
                    "roadmap_step": self._roadmap_step_index,
                    "roadmap_total": self._roadmap_total,
                })
            except Exception:
                pass

    # ── Core Autopilot Loop ──

    async def start(self) -> None:
        """Start the autopilot background loop."""
        if self.is_active:
            console.print("[bold yellow]⚠️ Autopilot is already running![/bold yellow]")
            return

        self.status = AutopilotStatus.STARTING
        self._stop_event.clear()
        self._scan_count = 0

        console.print(Panel(
            f"[bold green]🤖 Antigravity Autopilot STARTING[/bold green]\n"
            f"[bold cyan]Mode:[/bold cyan] {self.mode.value.upper()}\n"
            f"[bold cyan]Poll Interval:[/bold cyan] {self.poll_interval}s\n"
            f"[bold cyan]Step Delay:[/bold cyan] {self.step_delay}s\n"
            f"[bold cyan]Max Wait:[/bold cyan] {self.max_wait_thinking}s",
            title="[bold cyan]⚡ AUTOPILOT[/bold cyan]",
            border_style="green",
        ))

        self._speak("Autopilot activated. Monitoring Antigravity.")
        self._log_action("START", f"Autopilot started in {self.mode.value} mode")

        self.status = AutopilotStatus.RUNNING
        self._task = asyncio.create_task(self._main_loop())

    async def stop(self) -> None:
        """Stop the autopilot."""
        if not self.is_active:
            return

        self._stop_event.set()
        self.status = AutopilotStatus.STOPPED

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        self._log_action("STOP", "Autopilot stopped")
        self._speak("Autopilot deactivated.")
        console.print("[bold yellow]⏹️ Autopilot stopped.[/bold yellow]")
        self._update_overlay_state(AntigravityState.UNKNOWN, "STOPPED")

    def pause(self) -> None:
        """Pause the autopilot (stop taking actions but keep monitoring)."""
        self._pause_event.clear()
        self.status = AutopilotStatus.PAUSED
        self._log_action("PAUSE", "Autopilot paused")

    def resume(self) -> None:
        """Resume the autopilot from pause."""
        self._pause_event.set()
        self.status = AutopilotStatus.RUNNING
        self._log_action("RESUME", "Autopilot resumed")

    def add_pending_prompt(self, prompt: str) -> None:
        """Add a prompt to the queue for injection into Antigravity."""
        self._pending_prompts.append(prompt)
        self._log_action("QUEUE", f"Queued prompt: '{prompt[:60]}...'")

    async def _main_loop(self) -> None:
        """Main autopilot polling loop."""
        try:
            while not self._stop_event.is_set():
                # Wait if paused
                await self._pause_event.wait()

                if self._stop_event.is_set():
                    break

                self._scan_count += 1

                try:
                    # 1. Take screenshot
                    screenshot = await self._take_screenshot()
                    if screenshot is None:
                        await asyncio.sleep(self.poll_interval)
                        continue

                    # 2. Detect state
                    detection = self.detector.detect_state(screenshot)
                    self._last_detection = detection

                    # 3. Log state changes
                    if detection.state != self._last_state:
                        self._on_state_change(self._last_state, detection.state, detection)

                    self._last_state = detection.state

                    # 4. Update overlay
                    self._update_overlay_state(detection.state)

                    # 5. Act based on mode and state
                    if self.mode != AutopilotMode.MONITOR:
                        await self._act_on_state(detection)

                    # Console progress (compact)
                    emoji = self.detector.get_state_emoji(detection.state)
                    console.print(
                        f"[dim]🤖 Autopilot #{self._scan_count}: "
                        f"{emoji} {detection.state.value} "
                        f"(conf={detection.confidence:.2f}, "
                        f"{detection.elements_count} els)[/dim]"
                    )

                except Exception as e:
                    logger.error(f"Autopilot loop error: {e}", exc_info=True)
                    console.print(f"[bold red]🤖 Autopilot error: {e}[/bold red]")

                await asyncio.sleep(self.poll_interval)

        except asyncio.CancelledError:
            logger.info("Autopilot loop cancelled")
        finally:
            self.status = AutopilotStatus.STOPPED

    # ── State Change Handler ──

    def _on_state_change(
        self,
        old_state: AntigravityState,
        new_state: AntigravityState,
        detection: DetectionResult,
    ) -> None:
        """Handle a state transition."""
        old_emoji = self.detector.get_state_emoji(old_state)
        new_emoji = self.detector.get_state_emoji(new_state)

        console.print(
            f"\n[bold cyan]🤖 State Change:[/bold cyan] "
            f"{old_emoji} {old_state.value} → {new_emoji} {new_state.value}"
        )

        self._log_action("STATE_CHANGE", f"{old_state.value} → {new_state.value}")

        # Track thinking duration
        if new_state == AntigravityState.THINKING:
            self._thinking_start_time = time.time()
        elif old_state == AntigravityState.THINKING and self._thinking_start_time:
            duration = time.time() - self._thinking_start_time
            console.print(f"[dim]🤖 AI was thinking for {duration:.1f}s[/dim]")
            self._thinking_start_time = None

        # Voice announcements for important transitions
        if new_state == AntigravityState.IMPLEMENTATION_PLAN:
            self._speak("Implementation plan detected.")
        elif new_state == AntigravityState.DONE:
            self._speak("Antigravity has completed the task.")
        elif new_state == AntigravityState.ERROR:
            self._speak("Error detected in Antigravity.")
        elif new_state == AntigravityState.PLAN_WITH_ROADMAP:
            self._speak("Roadmap detected with multiple steps.")

    # ── Action Engine ──

    async def _act_on_state(self, detection: DetectionResult) -> None:
        """Take action based on the detected state and autopilot mode."""
        state = detection.state

        if state == AntigravityState.THINKING or state == AntigravityState.EXECUTING:
            # AI is working — just wait
            self.status = AutopilotStatus.WAITING

            # Check for thinking timeout
            if self._thinking_start_time:
                elapsed = time.time() - self._thinking_start_time
                if elapsed > self.max_wait_thinking:
                    self._log_action("TIMEOUT", f"AI thinking exceeded {self.max_wait_thinking}s")
                    self._speak("Warning: Antigravity appears to be stuck. It has been thinking for too long.")
            return

        elif state == AntigravityState.IMPLEMENTATION_PLAN:
            # Plan is ready — click Proceed!
            if detection.proceed_coords:
                self.status = AutopilotStatus.ACTING
                await self._click_proceed(detection.proceed_coords)
                self.status = AutopilotStatus.RUNNING
            else:
                console.print("[bold yellow]🤖 Plan detected but Proceed button not found — waiting...[/bold yellow]")

        elif state == AntigravityState.PLAN_WITH_ROADMAP:
            # Roadmap detected — extract and execute steps
            if self.mode == AutopilotMode.FULL_AUTO and detection.roadmap_steps:
                self.status = AutopilotStatus.EXECUTING_ROADMAP
                await self._execute_roadmap(detection.roadmap_steps)
                self.status = AutopilotStatus.RUNNING

        elif state in (AntigravityState.IDLE, AntigravityState.AWAITING_INPUT):
            # AI is idle — inject next pending prompt if available
            if self._pending_prompts:
                self.status = AutopilotStatus.ACTING
                prompt = self._pending_prompts.pop(0)
                await self._inject_prompt(prompt, detection.input_coords)
                self.status = AutopilotStatus.RUNNING
            elif self._current_roadmap and self._roadmap_step_index < self._roadmap_total:
                # Continue executing roadmap steps
                self.status = AutopilotStatus.EXECUTING_ROADMAP
                await self._execute_next_roadmap_step()
                self.status = AutopilotStatus.RUNNING

        elif state == AntigravityState.DONE or state == AntigravityState.WALKTHROUGH:
            # Task is done
            if self._current_roadmap and self._roadmap_step_index < self._roadmap_total:
                # Still have roadmap steps — continue after delay
                console.print(
                    f"[bold green]🤖 Step {self._roadmap_step_index}/{self._roadmap_total} "
                    f"completed! Moving to next...[/bold green]"
                )
                await asyncio.sleep(self.step_delay)
                self.status = AutopilotStatus.EXECUTING_ROADMAP
                await self._execute_next_roadmap_step()
                self.status = AutopilotStatus.RUNNING
            else:
                self._log_action("COMPLETE", "Antigravity task completed")
                import subprocess
                try:
                    subprocess.Popen(
                        ["afplay", "/System/Library/Sounds/Glass.aiff"],
                        stderr=subprocess.DEVNULL,
                    )
                except Exception:
                    pass

        elif state == AntigravityState.ERROR:
            self._log_action("ERROR_DETECTED", detection.details)
            # In full-auto, we could try recovery, but for now just notify
            self._speak("Error detected in Antigravity. Please check.")

    # ── Action Implementations ──

    async def _take_screenshot(self) -> Optional[Image.Image]:
        """Take a screenshot via the driver."""
        if not self._screenshot_fn:
            logger.warning("No screenshot function configured for autopilot")
            return None
        try:
            return await self._screenshot_fn()
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return None

    async def _click_proceed(self, coords: Tuple[float, float]) -> None:
        """Click the Proceed button."""
        x, y = coords
        self._log_action("CLICK_PROCEED", f"Clicking Proceed at ({x:.0f}, {y:.0f})")

        if self._click_fn:
            await self._click_fn(x, y)
            await asyncio.sleep(1.0)

            # Play acknowledgment sound
            import subprocess
            try:
                subprocess.Popen(
                    ["afplay", "/System/Library/Sounds/Tink.aiff"],
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

    async def _inject_prompt(
        self, prompt: str, input_coords: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Type a prompt into Antigravity's input field."""
        self._log_action("INJECT_PROMPT", f"Pasting: '{prompt[:60]}...'")

        # 1. Focus Antigravity
        if self._navigate_fn:
            await self._navigate_fn("Antigravity IDE")
            await asyncio.sleep(1.0)

        # 2. Click input area
        if input_coords and self._click_fn:
            await self._click_fn(input_coords[0], input_coords[1])
            await asyncio.sleep(0.3)
        else:
            # Fallback: click near bottom center (typical chat input location)
            if self._click_fn:
                # Use approximate center-bottom position
                await self._click_fn(720.0, 830.0)
                await asyncio.sleep(0.3)

        # 3. Type the prompt
        if self._type_fn:
            await self._type_fn(prompt)
            await asyncio.sleep(0.5)

        # 4. Press Enter to submit
        if self._press_key_fn:
            await self._press_key_fn("return")
            await asyncio.sleep(2.0)

    async def _execute_roadmap(self, steps: List[RoadmapStep]) -> None:
        """Execute a series of roadmap steps sequentially."""
        # Filter out already-completed steps
        pending_steps = [s for s in steps if not s.is_completed]
        if not pending_steps:
            console.print("[bold green]🤖 All roadmap steps already completed![/bold green]")
            return

        self._current_roadmap = pending_steps
        self._roadmap_total = len(pending_steps)
        self._roadmap_step_index = 0

        console.print(Panel(
            f"[bold green]🗺️ Roadmap Execution Starting[/bold green]\n"
            f"[bold cyan]Total Steps:[/bold cyan] {self._roadmap_total}\n"
            f"[bold cyan]Mode:[/bold cyan] Auto-Execute (delay: {self.step_delay}s)",
            title="[bold cyan]🤖 AUTOPILOT ROADMAP[/bold cyan]",
            border_style="green",
        ))

        self._speak(f"Executing roadmap with {self._roadmap_total} steps.")
        self._log_action("ROADMAP_START", f"Starting {self._roadmap_total}-step roadmap")

        # Execute first step
        await self._execute_next_roadmap_step()

    async def _execute_next_roadmap_step(self) -> None:
        """Execute the next step in the current roadmap."""
        if not self._current_roadmap or self._roadmap_step_index >= self._roadmap_total:
            # Roadmap complete
            self._log_action("ROADMAP_COMPLETE", f"All {self._roadmap_total} steps executed")
            self._speak(f"Roadmap completed. All {self._roadmap_total} steps executed.")
            self._current_roadmap = []
            self._roadmap_step_index = 0
            self._roadmap_total = 0
            return

        step = self._current_roadmap[self._roadmap_step_index]
        self._roadmap_step_index += 1

        console.print(
            f"\n[bold yellow]───── Roadmap Step {self._roadmap_step_index}/{self._roadmap_total} "
            f"─────[/bold yellow]"
        )
        console.print(f"[bold white]{step.text}[/bold white]")

        self._log_action(
            "ROADMAP_STEP",
            f"Step {self._roadmap_step_index}/{self._roadmap_total}: {step.text[:80]}",
        )

        # Wait for Antigravity to be idle before injecting
        idle_wait_count = 0
        max_idle_wait = 60  # Max 60 polls (~3 minutes at 3s interval)

        while idle_wait_count < max_idle_wait:
            if self._stop_event.is_set():
                return

            screenshot = await self._take_screenshot()
            if screenshot is None:
                await asyncio.sleep(self.poll_interval)
                idle_wait_count += 1
                continue

            detection = self.detector.detect_state(screenshot)

            if detection.state in (
                AntigravityState.IDLE,
                AntigravityState.AWAITING_INPUT,
                AntigravityState.DONE,
                AntigravityState.WALKTHROUGH,
            ):
                # Antigravity is ready — inject the step
                break
            elif detection.state == AntigravityState.IMPLEMENTATION_PLAN:
                # Plan appeared — click Proceed first
                if detection.proceed_coords:
                    await self._click_proceed(detection.proceed_coords)
                await asyncio.sleep(self.step_delay)
            else:
                # Still busy — wait
                console.print(
                    f"[dim]🤖 Waiting for Antigravity to finish "
                    f"({detection.state.value})... "
                    f"[poll {idle_wait_count + 1}/{max_idle_wait}][/dim]"
                )

            await asyncio.sleep(self.poll_interval)
            idle_wait_count += 1

        if idle_wait_count >= max_idle_wait:
            self._log_action("TIMEOUT", f"Timed out waiting for idle before step {self._roadmap_step_index}")
            return

        # Apply step delay before next injection
        await asyncio.sleep(self.step_delay)

        # Inject the step prompt
        await self._inject_prompt(step.text)

        self._update_overlay_state(
            AntigravityState.EXECUTING,
            f"Step {self._roadmap_step_index}/{self._roadmap_total}",
        )

    # ── Status Reporting ──

    def get_status_summary(self) -> str:
        """Get a human-readable status summary."""
        lines = [
            f"🤖 Autopilot Status: {self.status.value.upper()}",
            f"   Mode: {self.mode.value.upper()}",
            f"   Last State: {self.detector.get_state_display()}",
            f"   Scans: {self._scan_count}",
            f"   Actions Taken: {self._total_actions_taken}",
        ]

        if self._current_roadmap:
            lines.append(
                f"   Roadmap: Step {self._roadmap_step_index}/{self._roadmap_total}"
            )

        if self._action_log:
            lines.append("   Recent Actions:")
            for a in self._action_log[-3:]:
                lines.append(f"     • {a}")

        return "\n".join(lines)

    def get_state_data(self) -> dict:
        """Get current state data as a dict (for overlay/IPC)."""
        return {
            "status": self.status.value,
            "mode": self.mode.value,
            "state": self._last_state.value,
            "state_display": self.detector.get_state_display(),
            "emoji": self.detector.get_state_emoji(),
            "scan_count": self._scan_count,
            "actions_taken": self._total_actions_taken,
            "roadmap_step": self._roadmap_step_index,
            "roadmap_total": self._roadmap_total,
            "recent_actions": [str(a) for a in self._action_log[-3:]],
        }
