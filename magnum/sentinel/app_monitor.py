"""
Universal App Sentinel — Watch any app with AI-driven content rules.
Monitors Gmail, WhatsApp, Slack, or any macOS app by periodically capturing
screenshots (even in background via Quartz) and evaluating content against rules.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from magnum.config import config
from magnum.intelligence.grounding import ScreenGrounder
from magnum.sentinel.content_analyzer import ContentAnalyzer, ContentMatch

logger = logging.getLogger(__name__)
console = Console()


class AppMonitorRule(BaseModel):
    """A monitoring rule for a specific app."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    app_name: str                    # "Gmail", "WhatsApp", "Slack", "Antigravity"
    watch_for: str                   # "emails from Master", "important messages"
    importance: str = "high"         # critical, high, medium, low
    action: str = "notify"           # notify, speak, click, auto_respond
    filter_prompt: str = ""          # Custom AI prompt for content evaluation
    enabled: bool = True
    trigger_count: int = 0
    last_triggered: float = 0.0
    cooldown_seconds: float = 60.0   # Min time between triggers for same rule

    def should_trigger(self) -> bool:
        """Check if enough time has passed since last trigger."""
        if not self.enabled:
            return False
        return (time.time() - self.last_triggered) >= self.cooldown_seconds


class MonitorMatch(BaseModel):
    """A match found by the sentinel."""
    rule_id: str
    rule_app: str
    rule_watch_for: str
    match: ContentMatch
    timestamp: float = 0.0
    notified: bool = False


class AppSentinel:
    """
    Universal background app monitor using OCR + AI reasoning.
    Periodically captures screenshots of target apps and evaluates content
    against monitoring rules.
    """

    def __init__(
        self,
        screenshot_fn: Optional[Callable] = None,
        capture_window_fn: Optional[Callable] = None,
        click_fn: Optional[Callable] = None,
        voice_fn: Optional[Callable] = None,
        overlay_fn: Optional[Callable] = None,
        analyzer: Optional[ContentAnalyzer] = None,
    ) -> None:
        self._screenshot_fn = screenshot_fn
        self._capture_window_fn = capture_window_fn
        self._click_fn = click_fn
        self._voice_fn = voice_fn
        self._overlay_fn = overlay_fn
        self._analyzer = analyzer or ContentAnalyzer()

        # Rules & State
        self._rules: Dict[str, AppMonitorRule] = {}
        self._matches: List[MonitorMatch] = []
        self._scan_count = 0
        self._active = False
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # Timing
        self.poll_interval: float = config.sentinel_poll_interval

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def rules(self) -> List[AppMonitorRule]:
        return list(self._rules.values())

    @property
    def active_rules(self) -> List[AppMonitorRule]:
        return [r for r in self._rules.values() if r.enabled]

    @property
    def recent_matches(self) -> List[MonitorMatch]:
        return self._matches[-10:]

    # ── Rule Management ──

    def add_rule(self, rule: AppMonitorRule) -> str:
        """Add a monitoring rule. Returns rule ID."""
        self._rules[rule.id] = rule
        logger.info(f"Sentinel rule added: [{rule.id}] Watch '{rule.app_name}' for '{rule.watch_for}'")
        console.print(
            f"[bold green]👁️ Sentinel Rule Added:[/bold green] "
            f"Watch [bold cyan]{rule.app_name}[/bold cyan] for "
            f"[bold yellow]'{rule.watch_for}'[/bold yellow] "
            f"(action: {rule.action}, importance: {rule.importance})"
        )
        return rule.id

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a monitoring rule by ID."""
        if rule_id in self._rules:
            rule = self._rules.pop(rule_id)
            logger.info(f"Sentinel rule removed: [{rule_id}] {rule.app_name}")
            return True
        return False

    def remove_rules_for_app(self, app_name: str) -> int:
        """Remove all rules for a specific app. Returns count removed."""
        to_remove = [
            rid for rid, r in self._rules.items()
            if r.app_name.lower() == app_name.lower()
        ]
        for rid in to_remove:
            self._rules.pop(rid)
        return len(to_remove)

    def find_rules_for_app(self, app_name: str) -> List[AppMonitorRule]:
        """Find all rules for a specific app."""
        lower = app_name.lower()
        return [r for r in self._rules.values() if r.app_name.lower() == lower]

    # ── Lifecycle ──

    async def start(self) -> None:
        """Start the sentinel monitoring loop."""
        if self._active:
            console.print("[bold yellow]⚠️ Sentinel is already running![/bold yellow]")
            return

        if not self._rules:
            console.print("[bold yellow]⚠️ No monitoring rules configured. Add rules first.[/bold yellow]")
            return

        self._active = True
        self._stop_event.clear()
        self._scan_count = 0

        active = self.active_rules
        console.print(Panel(
            f"[bold green]👁️ Sentinel ACTIVE[/bold green]\n"
            f"[bold cyan]Monitoring {len(active)} rule(s):[/bold cyan]\n"
            + "\n".join(
                f"  • [yellow]{r.app_name}[/yellow]: {r.watch_for} ({r.importance})"
                for r in active
            ),
            title="[bold cyan]⚡ MAGNUM SENTINEL[/bold cyan]",
            border_style="green",
        ))

        self._task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        """Stop the sentinel."""
        self._active = False
        self._stop_event.set()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        console.print("[bold yellow]⏹️ Sentinel stopped.[/bold yellow]")

    # ── Core Monitor Loop ──

    async def _monitor_loop(self) -> None:
        """Main monitoring loop — checks all rules against their target apps."""
        try:
            while not self._stop_event.is_set():
                self._scan_count += 1

                for rule in self.active_rules:
                    if self._stop_event.is_set():
                        break

                    if not rule.should_trigger():
                        continue

                    try:
                        await self._check_rule(rule)
                    except Exception as e:
                        logger.warning(f"Sentinel check failed for rule [{rule.id}]: {e}")

                await asyncio.sleep(self.poll_interval)

        except asyncio.CancelledError:
            pass
        finally:
            self._active = False

    async def _check_rule(self, rule: AppMonitorRule) -> None:
        """Check a single rule against its target app's current screen."""
        # 1. Capture the app's screen
        screenshot = await self._capture_app(rule.app_name)
        if screenshot is None:
            return

        # 2. OCR extract text
        elements = await asyncio.to_thread(
            ScreenGrounder.extract_screen_text_elements, screenshot
        )
        ocr_text = " ".join(el.text.strip() for el in elements if el.text.strip())

        if not ocr_text or len(ocr_text) < 10:
            return

        # 3. AI content evaluation
        description = rule.filter_prompt or f"Watch for: {rule.watch_for}"
        match_result = await asyncio.to_thread(
            self._analyzer.evaluate_content,
            ocr_text,
            description,
            rule.watch_for,
            rule.app_name,
            screenshot,
        )

        # 4. If matched, take action
        if match_result.matched and match_result.confidence >= 0.5:
            rule.trigger_count += 1
            rule.last_triggered = time.time()

            monitor_match = MonitorMatch(
                rule_id=rule.id,
                rule_app=rule.app_name,
                rule_watch_for=rule.watch_for,
                match=match_result,
                timestamp=time.time(),
            )
            self._matches.append(monitor_match)
            if len(self._matches) > 50:
                self._matches = self._matches[-50:]

            await self._handle_match(rule, match_result)

    async def _capture_app(self, app_name: str) -> Optional[Image.Image]:
        """Capture a screenshot of the specified app (even in background)."""
        # Try background window capture first (via Quartz)
        if self._capture_window_fn:
            try:
                img = await asyncio.to_thread(self._capture_window_fn, app_name)
                if img is not None:
                    return img
            except Exception as e:
                logger.debug(f"Window capture for '{app_name}' failed: {e}")

        # Fallback to full desktop screenshot
        if self._screenshot_fn:
            try:
                result = self._screenshot_fn()
                if asyncio.iscoroutine(result):
                    return await result
                return result
            except Exception:
                pass

        return None

    async def _handle_match(self, rule: AppMonitorRule, match: ContentMatch) -> None:
        """Handle a content match — notify, speak, or take action."""
        importance_emoji = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢",
        }.get(rule.importance, "⚪")

        # Console notification
        console.print(Panel(
            f"[bold white]{match.summary}[/bold white]\n\n"
            f"[dim]Confidence: {match.confidence:.0%}[/dim]\n"
            f"[dim]Importance: {match.importance}/10[/dim]"
            + (f"\n[dim]Items: {', '.join(match.matched_items)}[/dim]" if match.matched_items else "")
            + (f"\n[bold cyan]Suggested: {match.suggested_action}[/bold cyan]" if match.suggested_action else ""),
            title=f"[bold green]{importance_emoji} {rule.app_name}: {rule.watch_for}[/bold green]",
            border_style="green",
        ))

        # macOS notification sound
        import subprocess
        try:
            subprocess.Popen(
                ["afplay", "/System/Library/Sounds/Ping.aiff"],
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

        # Voice notification
        if rule.action in ("notify", "speak") and self._voice_fn:
            try:
                speak_text = f"Alert from {rule.app_name}: {match.summary}"
                self._voice_fn(speak_text)
            except Exception:
                pass

        # Overlay notification
        if self._overlay_fn:
            try:
                self._overlay_fn("sentinel_alert", {
                    "app": rule.app_name,
                    "watch_for": rule.watch_for,
                    "summary": match.summary,
                    "importance": rule.importance,
                    "emoji": importance_emoji,
                })
            except Exception:
                pass

    # ── Status ──

    def get_status_summary(self) -> str:
        """Get a human-readable status summary."""
        lines = [f"👁️ Sentinel: {'ACTIVE' if self._active else 'STOPPED'}"]
        lines.append(f"   Scans: {self._scan_count}")
        lines.append(f"   Rules: {len(self._rules)} ({len(self.active_rules)} active)")

        for r in self.active_rules:
            emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(r.importance, "⚪")
            lines.append(f"   {emoji} [{r.id}] {r.app_name}: '{r.watch_for}' (triggered {r.trigger_count}x)")

        if self._matches:
            lines.append(f"   Recent Matches: {len(self._matches)}")
            for m in self._matches[-3:]:
                lines.append(f"     • {m.rule_app}: {m.match.summary[:60]}")

        return "\n".join(lines)

    def get_rules_table(self) -> Table:
        """Get a rich table of all monitoring rules."""
        table = Table(title="👁️ Sentinel Rules", border_style="cyan")
        table.add_column("ID", style="bold")
        table.add_column("App", style="cyan")
        table.add_column("Watch For", style="yellow")
        table.add_column("Importance", style="green")
        table.add_column("Action", style="magenta")
        table.add_column("Triggers", style="white")
        table.add_column("Status", style="dim")

        for r in self._rules.values():
            emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(r.importance, "⚪")
            table.add_row(
                r.id, r.app_name, r.watch_for,
                f"{emoji} {r.importance}", r.action,
                str(r.trigger_count),
                "✅ ON" if r.enabled else "⏸️ OFF",
            )

        return table
