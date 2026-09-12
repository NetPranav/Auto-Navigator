"""
Magnum Master Agent.
Executes user instructions via dynamic AI planning, OCR-first perception,
concurrent task queue with background watchers, and voice feedback.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from magnum.config import config, AppConfig
from magnum.drivers.base import BaseDriver
from magnum.drivers.desktop_driver import DesktopDriver
from magnum.drivers.headless_driver import HeadlessDriver
from magnum.hitl.hitl_handler import HitlHandler, HitlActionType
from magnum.intelligence.grounding import ScreenGrounder, TextElement
from magnum.intelligence.nim_client import NimClient, GroundedAction
from magnum.intelligence.planner import PlanChecklist, PlanStep
from magnum.task_queue import TaskQueue, MagnumTask, TaskType
from magnum.ui import get_overlay
from magnum.voice import get_voice_engine
from magnum.autopilot import AntigravityAutopilot, AutopilotMode
from magnum.sentinel.app_monitor import AppSentinel, AppMonitorRule
from magnum.sentinel.suggestion_engine import SuggestionEngine
from magnum.sentinel.activity_journal import ActivityJournal, ActivityEntry, get_activity_journal
from magnum.sentinel.content_analyzer import ContentAnalyzer

logger = logging.getLogger(__name__)
console = Console()


class MagnumAgent:
    """Master AI agent with dynamic planning, live HUD, OCR-first grounding, and concurrent tasking."""

    def __init__(
        self,
        mode: Optional[str] = None,
        cfg: Optional[AppConfig] = None,
    ) -> None:
        self.config = cfg or config
        self.mode = mode or self.config.default_execution_mode
        self.nim_client = NimClient()
        self.hitl_handler = HitlHandler()
        self.overlay = get_overlay()
        self.task_queue = TaskQueue()
        self.voice_engine = get_voice_engine()

        # Connect voice output & notifications to task queue
        self.task_queue.set_callbacks(
            on_speak=self.voice_engine.speak,
            on_notify=self.hitl_handler.notify,
        )

        # Initialize persistent logging to magnum.log (and magnm.log)
        from magnum.logger import setup_magnum_logging
        self.log_file = setup_magnum_logging()

        if self.mode == "desktop":
            self.driver: BaseDriver = DesktopDriver()
        else:
            self.driver = HeadlessDriver(
                headless=False,
                user_data_dir=self.config.browser_profile_path,
                viewport_width=self.config.viewport_width,
                viewport_height=self.config.viewport_height,
            )

        # Antigravity Autopilot (initialized lazily, wired to driver on start)
        self._autopilot: Optional[AntigravityAutopilot] = None

        # Universal Sentinel System (initialized lazily)
        self._sentinel: Optional[AppSentinel] = None
        self._suggestion_engine: Optional[SuggestionEngine] = None
        self._activity_journal: ActivityJournal = get_activity_journal()
        self._content_analyzer: Optional[ContentAnalyzer] = None

    async def __aenter__(self) -> MagnumAgent:
        await self.driver.start()
        try:
            from magnum.ui.quick_launcher import get_quick_launcher
            self._quick_launcher = get_quick_launcher(on_command=self.process_instruction)
            self._quick_launcher.start()
        except Exception:
            self._quick_launcher = None
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if getattr(self, "_quick_launcher", None):
            self._quick_launcher.stop()
        # Stop autopilot if running
        if self._autopilot and self._autopilot.is_active:
            await self._autopilot.stop()
        # Stop sentinel if running
        if self._sentinel and self._sentinel.is_active:
            await self._sentinel.stop()
        if self._suggestion_engine and self._suggestion_engine.is_active:
            await self._suggestion_engine.stop()
        self.voice_engine.stop()
        self.task_queue.cancel_all_watchers()
        self.task_queue.cancel_autopilot()
        await self.driver.close()

    def _extract_stop_condition(self, lower: str) -> Optional[str]:
        """Extract stop condition from natural speech. Returns None for manual stop."""
        # 1. Manual stop phrases: do NOT stop automatically on screen text!
        manual_stop_phrases = (
            "until i tell you", "until i say", "until i stop", "tell you to stop",
            "tell it to stop", "unitl i tell", "unitl i say",
            "manually close", "manually stop", "until i close", "until stopped",
            "until you to stop", "until i tell",
        )
        if any(p in lower for p in manual_stop_phrases):
            return None

        # 2. Match automated screen-text stop conditions (e.g. "until running is complete")
        stop_match = re.search(r"(?:until|till)\s+(?:the\s+)?(?:ai\s+)?(?:is\s+)?(?:running\s+is\s+)?(.+?)(?:\.|$)", lower)
        if stop_match:
            stop_text = stop_match.group(1).strip()
            if any(w in stop_text for w in ("i tell you", "i say", "you to stop", "manually")):
                return None
            if any(w in stop_text for w in ("complete", "finish", "done", "end")):
                return "complete"
            return stop_text
        return None

    def _extract_region(self, text: str) -> Optional[str]:
        """Extract spatial screen region constraint from instruction."""
        lower = text.lower()
        if "bottom right" in lower or "lower right" in lower or "bottom-right" in lower:
            return "bottom_right"
        elif "bottom left" in lower or "lower left" in lower or "bottom-left" in lower:
            return "bottom_left"
        elif "top right" in lower or "upper right" in lower or "top-right" in lower:
            return "top_right"
        elif "top left" in lower or "upper left" in lower or "top-left" in lower:
            return "top_left"
        elif "bottom" in lower:
            return "bottom"
        elif "top" in lower:
            return "top"
        elif "right" in lower:
            return "right"
        elif "left" in lower:
            return "left"
        return None

    def _clean_watcher_target(self, raw_target: str, full_sentence: Optional[str] = None) -> str:
        """Clean natural language noise from extracted watcher button target."""
        # 1. Check if the full sentence or raw target mentions a known button
        search_space = f"{full_sentence or ''} {raw_target}".lower().strip()
        for known in ("submit", "proceed", "continue", "confirm", "approve", "allow", "done", "next", "ok", "yes"):
            if known in search_space:
                return known.title()

        # 2. Strip filler words
        lower = raw_target.lower().strip()
        clean = re.sub(
            r"\b(button|in|the|anti\s*gravity|integrity|whenever|once|comes|appears|shows|up|on|it|again|and)\b",
            " ",
            lower,
            flags=re.IGNORECASE,
        )
        clean = " ".join(clean.split()).strip()
        return clean.title() if clean else raw_target.title()

    def _parse_watcher_instruction(self, instruction: str) -> Optional[dict]:
        """
        Parse natural language to detect watcher-type instructions.
        Returns watcher config or None if it's a regular task.
        """
        lower = instruction.lower().strip()

        # 1. Direct teaching: "learn that ..."
        if lower.startswith("learn that ") or lower.startswith("remember that "):
            from magnum.intelligence.memory import get_memory
            rule = instruction.split("that ", 1)[1].strip()
            get_memory().learn_user_rule(rule)
            self.voice_engine.speak("Learned. I have committed this to memory.")
            console.print(f"[bold green]🧠 Learned rule:[/bold green] '{rule}'")
            return None

        # 2. Semantic Persistent Watcher: Catch conversational multi-phrase instructions FIRST!
        # e.g. "in the right side of the integrity there is a submit button click on it again and again whenever it appears until i tell you to stop"
        # e.g. "turn the watcher on to click on the submit button again and again until I tell you to stop"
        watcher_keywords = [
            "start a watcher",
            "start watcher",
            "start the watcher",
            "watcher to look for",
            "until i tell you to stop",
            "until i tell it to stop",
            "unitl i tell it to stop",
            "tell it to stop",
            "until i say stop",
            "until stopped",
            "tell you to stop",
            "again and again",
            "multiple times",
            "as soon as you find",
            "as soon as it appears",
            "every time it appears",
            "each time it appears",
            "repeating until",
            "turn the watcher on",
            "turn on the watcher",
            "turn watcher on",
            "whenever",
        ]
        if any(kw in lower for kw in watcher_keywords):
            target = self._clean_watcher_target(lower, full_sentence=lower)
            return {
                "watch_for": target,
                "action": "CLICK",
                "stop_when": self._extract_stop_condition(lower),
                "poll_interval": self.config.watcher_poll_interval,
                "region": self._extract_region(lower),
            }

        # 3. Match all variations of watching for an appearance:
        # e.g. "execute the submit button when it appears", "click submit button once it shows", "press continue if it appears"
        m1 = re.search(r'(?:click|press|hit|execute|tap)\s+(?:on\s+)?(?:the\s+)?(.+?)\s+(?:button\s+)?(?:when(?:ever)?|once|if|as soon as)\s+(?:it\s+)?(?:appears|is visible|shows|pops up)', lower)
        if m1:
            target = self._clean_watcher_target(m1.group(1), full_sentence=lower)
            return {
                "watch_for": target,
                "action": "CLICK",
                "stop_when": self._extract_stop_condition(lower),
                "poll_interval": self.config.watcher_poll_interval,
                "region": self._extract_region(lower),
            }

        # 4. "wait for [X] and click [it]"
        m2 = re.search(r'wait\s+for\s+(?:the\s+)?(.+?)\s+(?:button\s+)?(?:and\s+)?(?:click|press|execute|hit)', lower)
        if m2:
            target = self._clean_watcher_target(m2.group(1), full_sentence=lower)
            return {
                "watch_for": target,
                "action": "CLICK",
                "stop_when": self._extract_stop_condition(lower),
                "poll_interval": self.config.watcher_poll_interval,
                "region": self._extract_region(lower),
            }

        # 5. "whenever/once/if [X] appears, click it"
        m3 = re.search(r'(?:when(?:ever)?|once|if)\s+(?:the\s+)?(.+?)\s+(?:button\s+)?(?:appears|shows|is visible|pops up),\s*(?:click|press|hit|execute)', lower)
        if m3:
            target = self._clean_watcher_target(m3.group(1), full_sentence=lower)
            return {
                "watch_for": target,
                "action": "CLICK",
                "stop_when": self._extract_stop_condition(lower),
                "poll_interval": self.config.watcher_poll_interval,
                "region": self._extract_region(lower),
            }

        # 6. "keep clicking [X] until [Y]"
        m4 = re.search(r'keep\s+(?:clicking|pressing|hitting|executing)\s+(?:the\s+)?(.+?)\s+(?:button\s+)?(?:until|till)\s+(.+)', lower)
        if m4:
            target = self._clean_watcher_target(m4.group(1), full_sentence=lower)
            return {
                "watch_for": target,
                "action": "CLICK",
                "stop_when": self._extract_stop_condition(lower),
                "poll_interval": self.config.watcher_poll_interval,
                "region": self._extract_region(lower),
            }

        # 7. "notify me when [X] appears"
        if "notify" in lower and ("when" in lower or "if" in lower):
            m5 = re.search(r'notify\s+(?:me\s+)?(?:when|if)\s+(.+?)(?:\.|$)', lower)
            if m5:
                return {
                    "watch_for": m5.group(1).strip(),
                    "action": "NOTIFY",
                    "stop_when": None,
                    "poll_interval": self.config.watcher_poll_interval,
                    "region": self._extract_region(lower),
                }

        return None

    def _should_auto_skip_app_open(
        self,
        current_step: Any,
        active_app: str,
        frontmost_app: str,
        top_bar_texts: List[str],
    ) -> Optional[str]:
        """
        Determine if current step is solely an app-launch step for an application
        that is already active and focused on screen.
        Prevents false-skipping of action steps like 'Search for Rishika in WhatsApp'.
        """
        title_lower = (getattr(current_step, "title", "") or "").lower().strip()
        open_prefixes = ("open ", "launch ", "switch to ", "focus ")

        # 1. Must start with an app opening command
        if not any(title_lower.startswith(p) for p in open_prefixes):
            return None

        # 2. Must NOT contain inner sub-actions (search, message, chat, etc.)
        subactions = (
            "search", "chat", "message", "text", "send", "type", "click", "find",
            "select", "reply", "compose", "call", "play", "tab", "file", "folder",
            "url", "link", "box", "field", "button", "input", "write", "post", "enter"
        )
        if any(act in title_lower for act in subactions):
            return None

        # 3. Extract target app from title
        target = title_lower
        for p in open_prefixes:
            if target.startswith(p):
                target = target[len(p):].strip()
                break

        clean_target = re.sub(r"\b(the|desktop|application|app|window)\b", "", target).strip()
        if not clean_target:
            return None

        candidates = [active_app.lower(), frontmost_app.lower()] + [t.lower() for t in top_bar_texts]
        for cand in candidates:
            cand = cand.strip()
            if not cand:
                continue
            if clean_target == cand or clean_target in cand or cand in clean_target:
                return cand.title()

        return None

    async def _wait_for_obstacle_resolution(
        self,
        baseline_ocr: List[Any],
        prompt_msg: str,
        max_wait_seconds: int = 120,
        poll_interval: float = 2.0,
    ) -> bool:
        """
        Closed-loop visual monitoring that waits for the user to resolve an on-screen obstacle
        (e.g. scanning a QR code with a phone, entering 2FA/OTP, solving a CAPTCHA, logging in).
        Detects real-life visual state transition when the obstacle screen is dismissed and
        the authenticated destination UI loads.
        """
        baseline_texts = set(el.text.strip().lower() for el in baseline_ocr if len(el.text.strip()) > 2)
        start_time = time.time()

        while time.time() - start_time < max_wait_seconds:
            await asyncio.sleep(poll_interval)
            try:
                poll_screenshot = await self.driver.screenshot("obstacle_poll.png")
                poll_ocr = ScreenGrounder.extract_screen_text_elements(poll_screenshot)
                current_texts = set(el.text.strip().lower() for el in poll_ocr if len(el.text.strip()) > 2)

                if baseline_texts:
                    still_present = baseline_texts.intersection(current_texts)
                    overlap_ratio = len(still_present) / len(baseline_texts)
                    new_elements_count = len(current_texts - baseline_texts)

                    # State transition: obstacle texts cleared (<45% overlap) OR significant new UI loaded (>6 new elements)
                    if overlap_ratio < 0.45 or (new_elements_count >= 6 and overlap_ratio < 0.75):
                        logger.info(f"Visual state change detected! Overlap={overlap_ratio:.2f}, NewElements={new_elements_count}")
                        return True
                else:
                    if len(current_texts) >= 5:
                        return True
            except Exception as e:
                logger.debug(f"Error in obstacle polling: {e}")

        return False

    async def process_instruction(self, instruction: str, is_workflow_step: bool = False) -> bool:
        """
        Process any instruction — auto-detects system commands, watchers, or foreground tasks.
        Automatically detects Hindi/Hinglish and translates into clear English instructions.
        """
        raw_input = instruction.strip()
        from magnum.voice.translator import get_hindi_translator
        translator = get_hindi_translator()
        trans_res = translator.translate(raw_input)
        if trans_res.was_translated:
            console.print(f"[dim]🇮🇳 Hindi/Hinglish Input ({trans_res.detected_language}): '{raw_input}'[/dim]")
            console.print(f"[bold cyan]🌐 Translated to English:[/bold cyan] '{trans_res.translated_text}'")
            instruction = trans_res.translated_text

        lower = instruction.lower().strip()

        from magnum.logger import log_instruction, get_recent_logs, get_log_file_path, get_latest_run_summary, get_latest_run_file_path
        log_instruction(
            instruction,
            mode=getattr(self, "mode", "desktop"),
            raw_input=raw_input if trans_res.was_translated else None,
            language=trans_res.detected_language if trans_res.was_translated else None,
        )

        # 0. Show comprehensive flight run / plan / execution audit / failure query
        if any(lower.startswith(k) for k in (
            "show run", "latest run", "what did you plan", "how did you try", "why did it fail",
            "what failed", "show plan", "flight recorder", "execution breakdown", "show log",
            "view log", "check log", "check logs", "audit", "show audit"
        )):
            summary = get_latest_run_summary()
            console.print(f"\n[bold cyan]🛸 Magnum Flight Recorder ({get_latest_run_file_path()}):[/bold cyan]\n")
            console.print(summary)
            self.voice_engine.speak("Displaying complete task execution breakdown.")
            return True

        # 1. Stop / Cancel all background watchers (robust natural language detection)
        is_start = any(lower.startswith(k) for k in ("start", "create", "launch", "run", "turn on", "set up", "make")) or "start a watcher" in lower or "create a watcher" in lower
        stop_verbs = ("stop", "cancel", "kill", "clear", "terminate", "disable", "turn off", "shut down", "end")
        watcher_nouns = ("watcher", "watchers", "watching", "background task", "background tasks", "monitor", "monitoring")

        is_stop_watcher = False
        if not is_start:
            if any(v in lower for v in stop_verbs) and any(n in lower for n in watcher_nouns):
                is_stop_watcher = True
            elif lower in ("stop", "cancel", "halt", "stop now", "stop it", "stop please") and bool(self.task_queue.get_active_watchers()):
                is_stop_watcher = True

        if is_stop_watcher:
            active = self.task_queue.get_active_watchers()
            self.task_queue.cancel_all_watchers()
            self.overlay.set_watchers([])
            self.voice_engine.speak("All background watchers stopped.")
            console.print(f"[bold yellow]⏹️ Cancelled {len(active)} active background watcher(s).[/bold yellow]")
            self.overlay.set_status("WATCHERS STOPPED")
            return True

        # 2. Watcher status query
        if lower in ("what are you watching", "watcher status", "status", "active watchers", "list watchers"):
            active = self.task_queue.get_active_watchers()
            autopilot_info = ""
            if self._autopilot and self._autopilot.is_active:
                autopilot_info = f"\n{self._autopilot.get_status_summary()}"
            if not active and not autopilot_info:
                msg = "No background watchers or autopilot currently active."
            else:
                targets = ", ".join(f"'{w.watcher_condition.watch_for}'" for w in active if w.watcher_condition)
                msg = f"Currently watching for {targets} in the background." if targets else ""
                if autopilot_info:
                    msg += autopilot_info
            self.voice_engine.speak(msg)
            console.print(f"[bold cyan]ℹ️ Status:[/bold cyan] {msg}")
            return True

        # 2b. Antigravity Autopilot Commands
        autopilot_result = await self._handle_autopilot_command(lower, instruction)
        if autopilot_result is not None:
            return autopilot_result

        # 2c. Universal Sentinel Commands (watch, recall, suggest)
        sentinel_result = await self._handle_sentinel_command(lower, instruction)
        if sentinel_result is not None:
            return sentinel_result

        # 3. What have you learned / Memory query
        if lower in ("what have you learned", "show memory", "list memory", "what do you know"):
            from magnum.intelligence.memory import get_memory
            mem = get_memory()
            rules = mem.user_rules
            rule_summary = f"I have learned {len(rules)} custom rules."
            self.voice_engine.speak(rule_summary)
            console.print(f"\n[bold green]🧠 Magnum Learned Memory:[/bold green]")
            console.print(mem.get_prompt_context())
            return True

        # 4. Custom Workflow Commands (only when not already executing a workflow step):
        if not is_workflow_step:
            # 4a. Create / Record Workflow (robust phonetic and keyword detection)
            workflow_synonyms = ("workflow", "work flow", "voucher", "workshop", "macro", "sequence")
            create_verbs = ("create", "start", "record", "make", "new", "begin", "build", "set up", "setup")
            if any(syn in lower for syn in workflow_synonyms) and any(verb in lower for verb in create_verbs):
                return await self._interactive_create_workflow()

            # 4b. List Workflows
            if lower in ("list workflows", "show workflows", "my workflows", "what workflows do i have", "list workflow"):
                from magnum.workflows import get_workflow_manager
                mgr = get_workflow_manager()
                wfs = mgr.list_workflows()
                if not wfs:
                    msg = "You have no saved workflows. Say 'start a workflow' to record one."
                else:
                    names = ", ".join(f"'{w.name}' ({len(w.steps)} steps)" for w in wfs)
                    msg = f"You have {len(wfs)} saved workflow(s): {names}."
                self.voice_engine.speak(msg)
                console.print(f"[bold cyan]📋 Workflows:[/bold cyan] {msg}")
                return True

            # 4c. Delete Workflow
            if lower.startswith("delete ") or lower.startswith("remove "):
                from magnum.workflows import get_workflow_manager
                wf_name = (
                    instruction.replace("delete workflow ", "")
                    .replace("delete ", "")
                    .replace("remove workflow ", "")
                    .replace("remove ", "")
                    .strip()
                )
                mgr = get_workflow_manager()
                found = mgr.find_workflow(wf_name)
                if found and mgr.delete_workflow(found.name):
                    msg = f"Deleted workflow '{found.name}'."
                    self.voice_engine.speak(msg)
                    console.print(f"[bold yellow]🗑️ {msg}[/bold yellow]")
                    return True

            # 4d. Run Workflow by Name
            from magnum.workflows import get_workflow_manager
            mgr = get_workflow_manager()
            matched_wf = None
            if lower.startswith("run workflow ") or lower.startswith("execute workflow ") or lower.startswith("start workflow "):
                wf_query = instruction.split("workflow ", 1)[1].strip()
                matched_wf = mgr.find_workflow(wf_query)
            elif lower.startswith("run the ") or lower.startswith("run "):
                wf_query = instruction.replace("run the ", "").replace("run ", "").strip()
                matched_wf = mgr.find_workflow(wf_query)
            else:
                # Check if instruction directly matches a saved workflow name
                if not any(lower.startswith(skip) for skip in ("delete", "remove", "stop", "cancel", "kill", "close", "quit", "exit")):
                    if len(lower.split()) in (2, 3, 4) and not any(w in lower for w in ("look for", "submit", "button", "corner", "wait", "click")):
                        matched_wf = mgr.find_workflow(instruction)

            if matched_wf:
                return await self._execute_workflow(matched_wf)

        # 4e. Phased Pipeline Autopilot (for Antigravity multi-phase development)
        if "pipeline" in lower or ("phase" in lower and any(act in lower for act in ("run", "start", "execute", "play"))):
            from magnum.pipeline import get_pipeline_manager, ProjectPipeline, PipelineStep
            pipe_mgr = get_pipeline_manager()
            
            words = instruction.split()
            target_file: Optional[Path] = None
            for w in words:
                cleaned_w = w.strip("',\"")
                if "." in cleaned_w:
                    try:
                        cand = Path(cleaned_w)
                        if cand.exists():
                            target_file = cand
                            break
                    except Exception:
                        pass
            
            # If no target file was explicitly named, look for standard pipeline / roadmap files
            if not target_file:
                for default_name in ("pipeline.md", "phases.md", "ROADMAP.md", "task.md", "implementation_plan.md"):
                    p = Path(default_name)
                    if p.exists():
                        target_file = p
                        break

            if target_file:
                phases = pipe_mgr.parse_phases_file(target_file)
                if phases:
                    pipe = ProjectPipeline(
                        name=target_file.stem.replace("_", " ").title(),
                        steps=[PipelineStep(phase_number=i, prompt=p) for i, p in enumerate(phases, start=1)],
                    )
                    return await pipe_mgr.run_pipeline(pipe, self)
                else:
                    console.print(f"[yellow]⚠️ No phases found in pipeline file: {target_file}[/yellow]")
                    self.voice_engine.speak(f"No phases found in {target_file.name}")
                    return False
            else:
                console.print("[yellow]⚠️ Pipeline command detected, but no valid pipeline file or phase file was found.[/yellow]")
                self.voice_engine.speak("Please specify a valid pipeline file to execute.")
                return False

        # 4f. Click-to-Teach / Learn Button
        if any(kw in lower for kw in ("learn this button", "click to teach", "teach button", "teach me where", "learn button")):
            label = "Target Button"
            for kw in ("teach button", "learn button", "teach me where", "learn this button", "click to teach"):
                if kw in lower:
                    rem = instruction.lower().split(kw, 1)[1].strip()
                    rem = rem.replace("is", "").replace("the", "").replace("button", "").strip()
                    if rem:
                        label = rem.title()
                    break
            return await self._interactive_teach_button(label)

        # 4g. Jarvis Native Browser Controls (Zero-latency AppleScript + DOM execution):
        from magnum.browser import get_browser_controller
        browser_ctrl = get_browser_controller()

        # 4g-1. Summarize webpage / Explain page
        if any(kw in lower for kw in ("summarize this page", "summarize the page", "what is this page about", "what does this page say", "explain this page", "explain the page", "page summary")):
            console.print("[bold cyan]🌐 Jarvis Web Summarizer:[/bold cyan] Reading active page...")
            self.overlay.set_status("SUMMARIZING WEBPAGE")
            summary = await browser_ctrl.summarize_active_page(self.nim_client)
            console.print(Panel(summary, title="[bold green]📄 Webpage Executive Summary[/bold green]", border_style="green"))
            first_sentence = summary.split("\n")[0].strip() if summary else "Summary generated."
            self.voice_engine.speak(first_sentence, blocking=False)
            try:
                from magnum.notifications import send_notification
                send_notification("Jarvis Web Summary", first_sentence[:80])
            except Exception:
                pass
            return True

        # 4g-2. Browser Tab Management
        if any(kw in lower for kw in ("switch to tab", "go to tab", "switch tab", "focus tab")):
            tab_query = re.sub(r"(?i)(?:switch\s+to\s+tab|go\s+to\s+tab|switch\s+tab|focus\s+tab)\s+", "", instruction).strip()
            tab_query = tab_query.replace("the", "").replace("tab", "").strip()
            matched = browser_ctrl.switch_to_tab(tab_query)
            if matched:
                msg = f"Switched to tab: '{matched.title}'"
                self.voice_engine.speak(msg, blocking=False)
                console.print(f"[bold green]✓ {msg}[/bold green]")
                return True
            else:
                msg = f"No open tab found matching '{tab_query}'."
                self.voice_engine.speak(msg, blocking=False)
                console.print(f"[bold yellow]⚠️ {msg}[/bold yellow]")
                return False

        if lower in ("close tab", "close this tab", "close active tab", "kill tab"):
            success = browser_ctrl.close_tab()
            if success:
                self.voice_engine.speak("Closed tab.", blocking=False)
                console.print("[bold green]✓ Closed active tab.[/bold green]")
                return True
            return False

        if lower in ("list tabs", "show tabs", "open tabs", "what tabs are open", "list open tabs"):
            tabs = browser_ctrl.list_tabs()
            if not tabs:
                msg = "No open browser windows or tabs found."
                console.print(f"[dim]{msg}[/dim]")
                self.voice_engine.speak(msg, blocking=False)
            else:
                table = Table(title=f"🌐 Open Browser Tabs ({len(tabs)})", border_style="cyan")
                table.add_column("#", style="bold yellow", width=4)
                table.add_column("Browser", style="dim", width=14)
                table.add_column("Title", style="bold white")
                table.add_column("URL", style="cyan")
                for i, t in enumerate(tabs, start=1):
                    table.add_row(str(i), t.browser, t.title[:45], t.url[:50])
                console.print(table)
                self.voice_engine.speak(f"You have {len(tabs)} open tabs.", blocking=False)
            return True

        if lower in ("new tab", "open new tab", "create new tab"):
            browser_ctrl.open_url("about:blank", new_tab=True)
            self.voice_engine.speak("Opened new tab.", blocking=False)
            console.print("[bold green]✓ Opened new tab.[/bold green]")
            return True

        # 4g-3. Fluid Page Navigation & Scrolling
        if any(lower.startswith(prefix) for prefix in ("scroll down", "scroll up", "scroll to bottom", "scroll to top", "page down", "page up")):
            direction = "down"
            if "up" in lower or "top" in lower:
                direction = "top" if "to top" in lower else "up"
            elif "bottom" in lower:
                direction = "bottom"
            browser_ctrl.scroll(direction=direction)
            return True

        if lower in ("reload", "reload page", "refresh", "refresh page"):
            browser_ctrl.navigate("reload")
            self.voice_engine.speak("Page refreshed.", blocking=False)
            return True

        if lower in ("go back", "previous page", "back page"):
            browser_ctrl.navigate("back")
            return True

        if lower in ("go forward", "next page", "forward page"):
            browser_ctrl.navigate("forward")
            return True

        # 4g-4. Direct Smart Search Shortcuts
        search_match = re.search(r"(?:search\s+for|search|look\s+up|google)\s+(.+?)\s+(?:on|in)\s+(google|youtube|github|reddit|amazon|wikipedia|duckduckgo)", lower)
        if search_match:
            query = search_match.group(1).strip()
            engine = search_match.group(2).strip()
            browser_ctrl.search(query=query, engine=engine)
            msg = f"Searching {engine.title()} for '{query}'"
            self.voice_engine.speak(msg, blocking=False)
            console.print(f"[bold cyan]🔍 {msg}[/bold cyan]")
            return True

        # 4h. Jarvis Visual Screen Q&A ("look at my screen", "what is on my screen", "what error is showing", "explain this screen")
        visual_qa_phrases = (
            "look at my screen",
            "look at the screen",
            "what is on my screen",
            "what's on my screen",
            "what is on the screen",
            "what do you see",
            "explain my screen",
            "explain what is on my screen",
            "explain this screen",
            "what error is showing",
            "what error is this",
            "why is this failing",
            "how do i fix this error",
            "read my screen",
            "analyze my screen",
            "scan my screen",
        )
        if any(kw in lower for kw in visual_qa_phrases):
            from magnum.intelligence.screen_qa import ScreenQARunner
            console.print("\n[bold cyan]👁️ Jarvis Visual Screen Analyst:[/bold cyan] Inspecting display...")
            self.overlay.set_status("ANALYZING SCREEN")

            target_app = None
            for app in ("antigravity", "chrome", "safari", "terminal", "slack"):
                if app in lower:
                    target_app = app.title()
                    break

            qa_res = await ScreenQARunner.capture_and_analyze(
                driver=self.driver,
                nim_client=self.nim_client,
                user_query=instruction,
                target_app=target_app,
            )

            from rich.panel import Panel
            from rich.markdown import Markdown
            console.print(Panel(
                Markdown(qa_res.analysis),
                title="[bold green]👁️ Jarvis Screen Visual Analysis[/bold green]",
                subtitle=f"[dim]{qa_res.headline[:60]}[/dim]",
                border_style="green",
            ))

            self.voice_engine.speak(qa_res.speak_text, blocking=False)
            try:
                from magnum.notifications import send_notification
                send_notification("Jarvis Screen Analysis", qa_res.headline[:80])
            except Exception:
                pass

            return True

        # 5. Check if this is a watcher instruction
        watcher_config = self._parse_watcher_instruction(instruction)
        if watcher_config:
            return await self._start_watcher(instruction, watcher_config)
        
        # 6. Regular foreground task
        task = self.task_queue.create_foreground_task(instruction)
        res = await self.task_queue.run_foreground(task, self.execute)

        # After foreground task finishes: if the instruction also asked for continuous watching, turn it on!
        watcher_config = self._parse_watcher_instruction(instruction)
        if watcher_config and not any(w.watcher_condition and w.watcher_condition.watch_for == watcher_config['watch_for'] for w in self.task_queue.get_active_watchers()):
            console.print(f"[bold green]👁️ Turning on persistent background watcher for '{watcher_config['watch_for']}'...[/bold green]")
            await self._start_watcher(instruction, watcher_config)

        return res

    # ── Antigravity Autopilot Integration ──

    def _get_or_create_autopilot(self, mode: Optional[AutopilotMode] = None) -> AntigravityAutopilot:
        """Get or create the Antigravity Autopilot, wired to the driver."""
        if self._autopilot is None:
            ap_mode = mode or AutopilotMode(self.config.autopilot_mode)

            def _overlay_callback(cmd: str, data=None):
                if cmd == "set_state" and isinstance(data, dict):
                    self.overlay.set_autopilot_state(data)
                elif cmd == "add_log" and isinstance(data, str):
                    self.overlay.add_autopilot_log(data)

            self._autopilot = AntigravityAutopilot(
                mode=ap_mode,
                screenshot_fn=lambda: asyncio.to_thread(self.driver.take_screenshot, "Antigravity"),
                click_fn=self.driver.click,
                type_fn=self.driver.type_text,
                press_key_fn=self.driver.press_key,
                hotkey_fn=self.driver.hotkey,
                navigate_fn=self.driver.navigate,
                overlay_fn=_overlay_callback,
                voice_fn=self.voice_engine.speak,
            )
        elif mode and self._autopilot.mode != mode:
            self._autopilot.mode = mode
        return self._autopilot

    async def _handle_autopilot_command(self, lower: str, instruction: str) -> Optional[bool]:
        """
        Detect and handle Antigravity Autopilot commands.
        Returns True/False if handled, None if not an autopilot command.
        """
        # ── Start Autopilot ──
        autopilot_start_phrases = (
            "automate antigravity", "autopilot on", "autopilot start",
            "start autopilot", "turn on autopilot", "launch autopilot",
            "fully automate", "fully automate it", "completely automate",
            "completely automate it", "automate it completely",
            "run autopilot", "activate autopilot",
            "automate the antigravity", "start the autopilot",
            "begin autopilot", "engage autopilot",
        )
        if any(phrase in lower for phrase in autopilot_start_phrases):
            # Determine mode from instruction
            mode = AutopilotMode.FULL_AUTO
            if "monitor" in lower and "only" in lower:
                mode = AutopilotMode.MONITOR
            elif "semi" in lower:
                mode = AutopilotMode.SEMI_AUTO

            autopilot = self._get_or_create_autopilot(mode)

            if autopilot.is_active:
                console.print("[bold yellow]🤖 Autopilot is already running![/bold yellow]")
                self.voice_engine.speak("Autopilot is already active.")
                return True

            # Create task queue entry
            task = self.task_queue.create_autopilot_task(instruction)
            task.status = __import__('magnum.task_queue', fromlist=['TaskStatus']).TaskStatus.ACTIVE

            self.overlay.set_status("🤖 AUTOPILOT ACTIVE")
            await autopilot.start()
            return True

        # ── Stop Autopilot ──
        autopilot_stop_phrases = (
            "stop autopilot", "autopilot off", "autopilot stop",
            "turn off autopilot", "disable autopilot", "kill autopilot",
            "cancel autopilot", "deactivate autopilot",
            "stop automating", "stop the autopilot",
        )
        if any(phrase in lower for phrase in autopilot_stop_phrases):
            if self._autopilot and self._autopilot.is_active:
                await self._autopilot.stop()
                self.task_queue.cancel_autopilot()
                self.overlay.set_status("AUTOPILOT STOPPED")
                return True
            else:
                console.print("[dim]No autopilot is currently running.[/dim]")
                self.voice_engine.speak("No autopilot is active.")
                return True

        # ── Pause/Resume Autopilot ──
        if lower in ("pause autopilot", "autopilot pause"):
            if self._autopilot and self._autopilot.is_active:
                self._autopilot.pause()
                self.voice_engine.speak("Autopilot paused.")
                return True
            return None

        if lower in ("resume autopilot", "autopilot resume", "unpause autopilot"):
            if self._autopilot and self._autopilot.is_active:
                self._autopilot.resume()
                self.voice_engine.speak("Autopilot resumed.")
                return True
            return None

        # ── Check Antigravity State (one-shot scan) ──
        check_phrases = (
            "check antigravity", "check the antigravity",
            "what is antigravity doing", "what's antigravity doing",
            "antigravity status", "is antigravity running",
            "is antigravity done", "is the ai done",
            "check on antigravity", "check on the antigravity",
            "what is the antigravity doing",
        )
        if any(phrase in lower for phrase in check_phrases):
            return await self._check_antigravity_state()

        # ── Autopilot Status ──
        if lower in ("autopilot status", "autopilot info", "show autopilot"):
            if self._autopilot:
                summary = self._autopilot.get_status_summary()
                console.print(Panel(summary, title="[bold cyan]🤖 Autopilot Status[/bold cyan]", border_style="cyan"))
                self.voice_engine.speak(f"Autopilot is {self._autopilot.status.value}.")
            else:
                console.print("[dim]Autopilot has not been started yet.[/dim]")
            return True

        # Not an autopilot command
        return None

    async def _check_antigravity_state(self) -> bool:
        """Perform a one-shot Antigravity state detection scan."""
        from magnum.intelligence.antigravity_detector import AntigravityDetector

        console.print("\n[bold cyan]🤖 Checking Antigravity state...[/bold cyan]")
        self.overlay.set_status("CHECKING ANTIGRAVITY")

        # Take screenshot of Antigravity
        screenshot = self.driver.take_screenshot("Antigravity")
        detector = AntigravityDetector()
        result = detector.detect_state(screenshot)

        # Display results
        emoji = detector.get_state_emoji(result.state)
        display = detector.get_state_display(result.state)

        console.print(Panel(
            f"[bold white]{display}[/bold white]\n\n"
            f"[dim]Confidence: {result.confidence:.0%}[/dim]\n"
            f"[dim]Elements detected: {result.elements_count}[/dim]\n"
            f"[dim]{result.details}[/dim]"
            + (f"\n[bold green]✓ Proceed button found[/bold green]" if result.proceed_coords else "")
            + (f"\n[bold green]✓ Submit button found[/bold green]" if result.submit_coords else "")
            + (f"\n[bold cyan]📋 {len(result.roadmap_steps)} roadmap steps detected[/bold cyan]" if result.roadmap_steps else ""),
            title=f"[bold cyan]{emoji} Antigravity State[/bold cyan]",
            border_style="cyan",
        ))

        self.voice_engine.speak(display)
        self.overlay.set_status(f"ANTIGRAVITY: {result.state.value.upper()}")
        return True

    # ── Universal Sentinel Integration ──

    def _get_or_create_sentinel(self) -> AppSentinel:
        """Get or create the AppSentinel instance, wired to driver."""
        if self._sentinel is None:
            self._content_analyzer = ContentAnalyzer(nim_client=self.nim_client)

            # Background window capture function
            capture_window_fn = None
            if hasattr(self.driver, 'take_screenshot'):
                capture_window_fn = self.driver.take_screenshot

            self._sentinel = AppSentinel(
                screenshot_fn=lambda: self.driver.take_screenshot(),
                capture_window_fn=capture_window_fn,
                click_fn=self.driver.click if hasattr(self.driver, 'click') else None,
                voice_fn=self.voice_engine.speak,
                overlay_fn=self._sentinel_overlay_callback,
                analyzer=self._content_analyzer,
            )
        return self._sentinel

    def _get_or_create_suggestion_engine(self) -> SuggestionEngine:
        """Get or create the SuggestionEngine instance."""
        if self._suggestion_engine is None:
            if self._content_analyzer is None:
                self._content_analyzer = ContentAnalyzer(nim_client=self.nim_client)

            self._suggestion_engine = SuggestionEngine(
                screenshot_fn=lambda: self.driver.take_screenshot(),
                overlay_fn=self._sentinel_overlay_callback,
                voice_fn=self.voice_engine.speak,
                analyzer=self._content_analyzer,
                journal=self._activity_journal,
            )
        return self._suggestion_engine

    def _sentinel_overlay_callback(self, cmd: str, data=None):
        """Overlay callback for sentinel events."""
        if cmd == "sentinel_alert" and isinstance(data, dict):
            self.overlay.set_status(
                f"👁️ {data.get('emoji', '')} {data.get('app', '')}: {data.get('watch_for', '')}"
            )
        elif cmd == "suggestion" and isinstance(data, dict):
            self.overlay.set_status(
                f"💡 {data.get('emoji', '')} {data.get('text', '')[:60]}"
            )

    async def _handle_sentinel_command(self, lower: str, instruction: str) -> Optional[bool]:
        """
        Detect and handle Universal Sentinel commands:
        - Watch/monitor: "watch email for messages from Master"
        - Stop watching: "stop watching email", "remove all monitors"
        - Recall: "what was I doing", "what was the most important thing"
        - Suggest: "suggest", "what should I do next"
        - Activity: "show my activity", "activity log"
        - Sentinel status: "sentinel status"
        """

        # ── Start Watching an App ──
        watch_verbs = ("watch", "monitor", "keep an eye on", "keep a watch", "keep watch")
        is_watch = any(lower.startswith(v) or f" {v} " in f" {lower} " for v in watch_verbs)

        # Avoid false positives with "stop watching"
        stop_words = ("stop", "cancel", "disable", "remove", "turn off", "kill")
        if is_watch and any(lower.startswith(sw) for sw in stop_words):
            is_watch = False

        if is_watch:
            return await self._start_sentinel_watch(instruction)

        # ── Stop Watching ──
        stop_watch_phrases = (
            "stop watching", "stop monitoring", "cancel monitor",
            "remove monitor", "remove all monitors", "stop sentinel",
            "sentinel off", "turn off sentinel", "disable sentinel",
            "stop all monitors", "cancel all monitors",
        )
        if any(phrase in lower for phrase in stop_watch_phrases):
            return await self._stop_sentinel_watch(lower, instruction)

        # ── Sentinel Status ──
        if lower in ("sentinel status", "sentinel info", "show sentinel", "show monitors",
                      "list monitors", "what are you watching", "monitoring status"):
            sentinel = self._get_or_create_sentinel()
            if sentinel.rules:
                console.print(sentinel.get_rules_table())
                console.print(f"\n{sentinel.get_status_summary()}")
            else:
                console.print("[dim]No monitoring rules configured.[/dim]")
            return True

        # ── Activity Recall ──
        recall_phrases = (
            "what was i doing", "what have i been doing",
            "what was the most important", "most important thing",
            "what was i working on", "what did i do",
            "recall my activity", "recall activity",
            "what did i spend time on", "time summary",
            "what were the important things",
        )
        if any(phrase in lower for phrase in recall_phrases):
            return await self._recall_activity(instruction)

        # ── Show Activity Log ──
        if lower in ("show my activity", "activity log", "show activity",
                      "my activity", "activity history", "show log"):
            return self._show_activity_log()

        # ── On-demand Suggestion ──
        suggest_phrases = (
            "suggest", "what should i do", "what should i do next",
            "what next", "give me a suggestion", "recommend",
            "what do you recommend", "next task", "priority",
        )
        if any(phrase in lower for phrase in suggest_phrases):
            return await self._get_suggestion()

        # ── Start/Stop Suggestion Engine ──
        if lower in ("start suggestions", "enable suggestions", "suggestions on",
                      "turn on suggestions"):
            engine = self._get_or_create_suggestion_engine()
            await engine.start()
            self.voice_engine.speak("Suggestion engine started. I'll observe and suggest tasks.")
            self.overlay.set_status("💡 SUGGESTIONS ACTIVE")
            return True

        if lower in ("stop suggestions", "disable suggestions", "suggestions off",
                      "turn off suggestions"):
            if self._suggestion_engine and self._suggestion_engine.is_active:
                await self._suggestion_engine.stop()
                self.voice_engine.speak("Suggestion engine stopped.")
                return True
            return None

        return None

    async def _start_sentinel_watch(self, instruction: str) -> bool:
        """Parse a natural language watch command and create a monitoring rule."""
        rule = self._parse_watch_instruction(instruction)

        sentinel = self._get_or_create_sentinel()
        sentinel.add_rule(rule)

        # Auto-start sentinel if not already running
        if not sentinel.is_active:
            await sentinel.start()

        # Also start suggestion engine for background activity logging
        engine = self._get_or_create_suggestion_engine()
        if not engine.is_active:
            await engine.start()

        self.voice_engine.speak(
            f"Now watching {rule.app_name} for {rule.watch_for}. "
            f"I'll alert you when there's a match."
        )
        return True

    def _parse_watch_instruction(self, instruction: str) -> AppMonitorRule:
        """Parse natural language into an AppMonitorRule."""
        lower = instruction.lower()

        # Detect app name from instruction
        app_mappings = {
            "email": "Mail", "gmail": "Google Chrome",
            "mail": "Mail", "outlook": "Microsoft Outlook",
            "whatsapp": "WhatsApp", "whats app": "WhatsApp",
            "slack": "Slack", "discord": "Discord",
            "telegram": "Telegram", "messages": "Messages",
            "chrome": "Google Chrome", "safari": "Safari",
            "antigravity": "Antigravity IDE",
            "vscode": "Visual Studio Code", "vs code": "Visual Studio Code",
            "terminal": "Terminal", "finder": "Finder",
            "teams": "Microsoft Teams", "zoom": "zoom.us",
        }

        app_name = ""
        for keyword, real_name in app_mappings.items():
            if keyword in lower:
                app_name = real_name
                break

        if not app_name:
            # Try to extract app name from the instruction
            words = instruction.split()
            for i, w in enumerate(words):
                if w.lower() in ("on", "in", "from") and i + 1 < len(words):
                    app_name = words[i + 1].strip().title()
                    break
            if not app_name:
                app_name = "Desktop"  # Monitor full desktop

        # Extract what to watch for
        watch_for = instruction
        # Remove common prefixes
        for prefix in ("watch ", "monitor ", "keep an eye on ", "keep a watch on ",
                       "keep watch on ", "look for ", "check for "):
            if lower.startswith(prefix):
                watch_for = instruction[len(prefix):]
                break

        # Remove "on/in/from [app]" from watch_for to get the actual target
        for keyword in app_mappings:
            patterns = [f" on {keyword}", f" in {keyword}", f" from {keyword}",
                        f" on the {keyword}", f" in the {keyword}"]
            for pat in patterns:
                if pat in watch_for.lower():
                    idx = watch_for.lower().index(pat)
                    watch_for = watch_for[:idx].strip()
                    break

        # Determine importance
        importance = "high"
        if any(w in lower for w in ("critical", "urgent", "emergency", "asap")):
            importance = "critical"
        elif any(w in lower for w in ("important",)):
            importance = "high"
        elif any(w in lower for w in ("only if important", "if important")):
            importance = "high"

        # Determine action
        action = "notify"
        if any(w in lower for w in ("tell me", "notify", "alert", "let me know")):
            action = "notify"
        elif any(w in lower for w in ("click", "press", "tap")):
            action = "click"

        # Build filter prompt
        filter_prompt = (
            f"Check if the screen content shows: {watch_for}. "
            f"Context: User is monitoring {app_name}. "
            f"Only return matched=true if there is a genuine match."
        )

        return AppMonitorRule(
            app_name=app_name,
            watch_for=watch_for,
            importance=importance,
            action=action,
            filter_prompt=filter_prompt,
            cooldown_seconds=60.0,
        )

    async def _stop_sentinel_watch(self, lower: str, instruction: str) -> bool:
        """Stop watching specific apps or all monitoring."""
        sentinel = self._get_or_create_sentinel()

        if "all" in lower:
            count = len(sentinel.rules)
            for r in list(sentinel.rules):
                sentinel.remove_rule(r.id)
            if sentinel.is_active:
                await sentinel.stop()
            self.voice_engine.speak(f"Stopped all {count} monitoring rules.")
            console.print(f"[bold yellow]⏹️ Removed {count} monitoring rule(s).[/bold yellow]")
            return True

        # Try to find which app to stop watching
        app_mappings = {
            "email": "Mail", "gmail": "Google Chrome", "mail": "Mail",
            "whatsapp": "WhatsApp", "slack": "Slack", "discord": "Discord",
            "chrome": "Google Chrome", "antigravity": "Antigravity IDE",
        }

        for keyword, real_name in app_mappings.items():
            if keyword in lower:
                count = sentinel.remove_rules_for_app(real_name)
                if count > 0:
                    self.voice_engine.speak(f"Stopped monitoring {real_name}.")
                    console.print(f"[bold yellow]⏹️ Removed {count} rule(s) for {real_name}.[/bold yellow]")
                    # Stop sentinel if no rules left
                    if not sentinel.rules and sentinel.is_active:
                        await sentinel.stop()
                    return True

        # Fallback: stop everything
        if sentinel.is_active:
            await sentinel.stop()
        self.voice_engine.speak("Sentinel monitoring stopped.")
        return True

    async def _recall_activity(self, question: str) -> bool:
        """Recall past activity using the activity journal + AI reasoning."""
        journal_text = self._activity_journal.get_journal_text(hours=24, max_entries=50)

        if "(No activity" in journal_text:
            self.voice_engine.speak("I don't have any activity records yet. Start the suggestion engine to begin tracking.")
            console.print("[dim]No activity recorded. Use 'start suggestions' to begin tracking.[/dim]")
            return True

        console.print("\n[bold cyan]🧠 Analyzing your activity history...[/bold cyan]")
        self.overlay.set_status("RECALLING ACTIVITY...")

        analyzer = self._content_analyzer or ContentAnalyzer(nim_client=self.nim_client)
        answer = await asyncio.to_thread(
            analyzer.recall_from_journal,
            journal_text,
            question,
        )

        console.print(Panel(
            f"[bold white]{answer}[/bold white]",
            title="[bold cyan]🧠 Activity Recall[/bold cyan]",
            border_style="cyan",
        ))

        # Speak a condensed version
        speak_text = answer[:200] if len(answer) > 200 else answer
        self.voice_engine.speak(speak_text)
        self.overlay.set_status("RECALL COMPLETE")
        return True

    def _show_activity_log(self) -> bool:
        """Display the recent activity log in the console."""
        journal_text = self._activity_journal.get_journal_text(hours=24, max_entries=30)
        stats = self._activity_journal.get_stats()

        console.print(Panel(
            f"[bold white]{journal_text}[/bold white]\n\n"
            f"[dim]Total entries: {stats.get('total_entries', 0)}[/dim]\n"
            f"[dim]Session duration: {stats.get('session_duration', 0) // 60}min[/dim]",
            title="[bold cyan]📋 Activity Log (Last 24h)[/bold cyan]",
            border_style="cyan",
        ))

        if stats.get("top_apps"):
            console.print(f"[bold cyan]Top Apps:[/bold cyan] " +
                          ", ".join(f"{app}({count})" for app, count in stats["top_apps"]))

        return True

    async def _get_suggestion(self) -> bool:
        """Generate an on-demand suggestion."""
        engine = self._get_or_create_suggestion_engine()

        console.print("\n[bold cyan]💡 Generating suggestion...[/bold cyan]")
        self.overlay.set_status("THINKING...")

        suggestion = await engine.suggest_now()

        if suggestion:
            console.print(Panel(
                f"[bold white]{suggestion.priority_emoji} {suggestion.suggestion}[/bold white]\n\n"
                f"[dim]Reason: {suggestion.reason}[/dim]\n"
                f"[dim]Category: {suggestion.category}[/dim]",
                title=f"[bold cyan]💡 Suggestion ({suggestion.priority.upper()})[/bold cyan]",
                border_style="cyan",
            ))
            self.voice_engine.speak(suggestion.suggestion)
        else:
            console.print("[dim]Couldn't generate a suggestion right now.[/dim]")

        return True

    async def _start_watcher(self, instruction: str, watcher_config: dict) -> bool:
        """Start a background watcher."""
        task = self.task_queue.create_watcher(
            instruction=instruction,
            **watcher_config,
        )
        
        console.print(f"\n[bold green]👁️ Watcher Created [{task.id}]:[/bold green]")
        console.print(f"  Watching for: [bold yellow]'{watcher_config['watch_for']}'[/bold yellow]")
        console.print(f"  Action: [bold cyan]{watcher_config['action']}[/bold cyan]")
        if watcher_config.get('stop_when'):
            console.print(f"  Stop when: [bold red]'{watcher_config['stop_when']}'[/bold red]")
        if watcher_config.get('region'):
            console.print(f"  Region: [bold magenta]{watcher_config['region'].replace('_', ' ').title()}[/bold magenta]")
        console.print(f"  Poll interval: [dim]{watcher_config['poll_interval']}s[/dim]")

        self.overlay.set_status(f"WATCHING: {watcher_config['watch_for'].upper()}")
        active_watchers = self.task_queue.get_active_watchers()
        watcher_names = [w.watcher_condition.watch_for for w in active_watchers if w.watcher_condition]
        self.overlay.set_watchers(watcher_names)

        target_app = None
        if "antigravity" in instruction.lower():
            target_app = "Antigravity"

        await self.task_queue.start_watcher(
            task=task,
            screenshot_fn=lambda: asyncio.to_thread(self.driver.take_screenshot, target_app),
            click_fn=self.driver.click,
        )
        
        console.print(f"[bold green]✓ Watcher is running in the background. You can give me other tasks.[/bold green]")
        return True

    async def _interactive_create_workflow(self) -> bool:
        """Interactively record a custom multi-step workflow by voice and terminal."""
        from magnum.workflows import get_workflow_manager
        mgr = get_workflow_manager()
        loop = asyncio.get_running_loop()
        is_voice_active = bool(self.voice_engine and self.voice_engine.is_listening)

        prompt_msg = "What would you like to call this workflow?"
        self.voice_engine.speak(prompt_msg, blocking=is_voice_active)
        console.print(f"\n[bold green]🎙️ Magnum:[/bold green] {prompt_msg}")

        # Get workflow name (Voice or CLI Terminal)
        name: Optional[str] = None
        if is_voice_active:
            name = await self.voice_engine.listen_for_instruction(silence_timeout=4.0)

        if not name or not name.strip():
            # Prompt via terminal input
            try:
                name = await loop.run_in_executor(None, lambda: console.input("[bold cyan]👉 Workflow Name: [/bold cyan]").strip())
            except Exception:
                pass

        if not name or not name.strip():
            self.voice_engine.speak("Workflow creation cancelled.")
            console.print("[dim]Workflow creation cancelled (no name provided).[/dim]")
            return False

        clean_name = name.strip().title()
        console.print(f"[bold cyan]📋 Recording Workflow:[/bold cyan] '{clean_name}'")
        step_msg = f"Got it, '{clean_name}'. What is step 1?"
        self.voice_engine.speak(step_msg, blocking=is_voice_active)
        console.print(f"[bold green]🎙️ Magnum:[/bold green] {step_msg}")

        steps: List[str] = []
        step_num = 1
        stop_keywords = (
            "that's it", "thats it", "complete", "done", "finish", "finished",
            "stop now", "stop recording", "exit", "no more", "that is all"
        )

        while True:
            step_text: Optional[str] = None
            if is_voice_active:
                console.print(f"[dim]🎤 Listening for Step {step_num} (say 'That\'s it' or 'Done' to finish, or type below)...[/dim]")
                step_text = await self.voice_engine.listen_for_instruction(silence_timeout=4.0)

            if not step_text or not step_text.strip():
                try:
                    step_text = await loop.run_in_executor(
                        None,
                        lambda: console.input(f"[bold cyan]👉 Step {step_num} (or 'done'): [/bold cyan]").strip()
                    )
                except Exception:
                    pass

            if not step_text or not step_text.strip():
                await asyncio.sleep(0.1)
                continue

            lower_step = step_text.lower().strip()
            if any(lower_step == kw or lower_step.startswith(kw) for kw in stop_keywords):
                break

            steps.append(step_text.strip())
            console.print(f"  [bold green]✓ Step {step_num}:[/bold green] '{step_text.strip()}'")
            step_num += 1

            next_prompt = f"Added. What is step {step_num}?"
            self.voice_engine.speak(next_prompt, blocking=is_voice_active)
            console.print(f"[bold green]🎙️ Magnum:[/bold green] {next_prompt}")

        if not steps:
            self.voice_engine.speak("No steps were recorded. Workflow cancelled.", blocking=True)
            console.print("[yellow]No steps recorded. Workflow cancelled.[/yellow]")
            return False

        mgr.save_workflow(clean_name, steps)
        success_msg = f"Workflow '{clean_name}' saved with {len(steps)} steps. Say 'Run {clean_name}' anytime to execute it."
        self.voice_engine.play_chime("/System/Library/Sounds/Glass.aiff")
        self.voice_engine.speak(success_msg, blocking=False)
        console.print(f"\n[bold green]🎉 {success_msg}[/bold green]\n")
        return True

    async def _interactive_teach_button(self, label: str) -> bool:
        """Guide user to click on a button to teach Magnum its exact coordinates and appearance."""
        from magnum.intelligence.memory import get_memory
        mem = get_memory()
        loop = asyncio.get_running_loop()

        clean_label = label.strip() or "target button"
        prompt = f"Please click on '{clean_label}' with your mouse now to teach me."
        self.voice_engine.speak(prompt, blocking=False)
        console.print(f"\n[bold green]🎯 Click-to-Teach:[/bold green] {prompt}")
        self.overlay.set_status(f"TEACH: CLICK '{clean_label.upper()}'")

        def _listen_mouse_click():
            from pynput import mouse
            clicked_pos = []

            def on_click(x, y, button, pressed):
                if pressed:
                    clicked_pos.append((x, y))
                    return False  # Stop listener

            try:
                with mouse.Listener(on_click=on_click) as listener:
                    listener.join(timeout=15.0)
            except Exception:
                pass

            return clicked_pos[0] if clicked_pos else None

        click_coords = await loop.run_in_executor(None, _listen_mouse_click)

        if not click_coords:
            console.print("[yellow]No click detected within 15 seconds. Learning cancelled.[/yellow]")
            self.voice_engine.speak("Learning cancelled.")
            return False

        cx, cy = click_coords
        mem.learn_visual_anchor(clean_label, cx, cy)
        self.voice_engine.play_chime("/System/Library/Sounds/Glass.aiff")
        success_msg = f"Learned '{clean_label}' at ({cx:.0f}, {cy:.0f}). I will remember this next time!"
        self.voice_engine.speak(success_msg, blocking=False)
        console.print(f"\n[bold green]🎉 {success_msg}[/bold green]\n")
        self.overlay.set_status(f"LEARNED: {clean_label.upper()}")
        return True

    async def _execute_workflow(self, workflow) -> bool:
        """Execute each step in a recorded multi-step workflow sequentially."""
        from magnum.ui import get_overlay
        overlay = get_overlay()

        console.print(f"\n[bold cyan]🚀 Executing Workflow:[/bold cyan] '{workflow.name}' ({len(workflow.steps)} steps)")
        self.voice_engine.speak(f"Starting workflow: {workflow.name}.", blocking=False)
        overlay.set_status(f"WORKFLOW: {workflow.name.upper()}")

        step_descriptions = [s.instruction for s in workflow.steps]
        overlay.update_plan(step_descriptions, active_idx=1, completed=[])

        completed_indices: List[int] = []
        for i, step in enumerate(workflow.steps, start=1):
            overlay.update_plan(step_descriptions, active_idx=i, completed=completed_indices)
            console.print(f"\n[bold yellow]─── Workflow Step {i}/{len(workflow.steps)}: {step.instruction} ───[/bold yellow]")

            # Execute the step instruction (guaranteed no recursive workflow lookup)
            success = await self.process_instruction(step.instruction, is_workflow_step=True)
            if not success:
                console.print(f"[bold red]❌ Step {i} failed in workflow '{workflow.name}'.[/bold red]")
                self.voice_engine.play_chime("/System/Library/Sounds/Basso.aiff")
                self.voice_engine.speak(f"Step {i} failed in workflow {workflow.name}.")
                return False

            completed_indices.append(i)
            await asyncio.sleep(0.5)

        overlay.update_plan(step_descriptions, active_idx=len(workflow.steps), completed=completed_indices)
        self.voice_engine.play_chime("/System/Library/Sounds/Glass.aiff")
        finish_msg = f"Workflow '{workflow.name}' completed successfully!"
        try:
            from magnum.notifications import send_notification
            send_notification("⚡ Magnum: Workflow Complete", finish_msg, sound="Glass")
        except Exception:
            pass
        self.voice_engine.speak(finish_msg, blocking=False)
        console.print(f"\n[bold green]🎉 {finish_msg}[/bold green]\n")
        return True

    async def execute(self, instruction: str) -> bool:
        """Dynamically plan and execute the user's task step-by-step."""
        import time
        start_exec_time = time.time()
        from magnum.logger import (
            flight_recorder,
            log_plan,
            log_step_start,
            log_perception,
            log_action_execution,
            log_failure,
            log_task_complete,
            log_user_interaction,
        )

        flight_recorder.start_task(instruction, mode=getattr(self, "mode", "desktop"))
        console.print(f"\n[bold cyan]🎯 Goal:[/bold cyan] {instruction}")
        console.print("[dim]🧠 Generating dynamic AI plan checklist...[/dim]")

        # Show active watchers
        active_watchers = self.task_queue.get_active_watchers()
        if active_watchers:
            console.print(f"[dim]👁️ {len(active_watchers)} background watcher(s) paused during this task[/dim]")

        try:
            # 1. Generate dynamic plan
            plan: PlanChecklist = self.nim_client.generate_plan(instruction)
            flight_recorder.record_plan(instruction, plan.steps)
            log_plan(instruction, [f"{s.title}: {s.description}" for s in plan.steps])

            table = Table(title="📋 MAGNUM PLAN", border_style="cyan")
            table.add_column("Step", style="bold yellow", width=6)
            table.add_column("Title", style="bold white")
            table.add_column("Description", style="dim")
            for s in plan.steps:
                table.add_row(str(s.step_index), s.title, s.description)
            console.print(table)

            # 2. HUD
            completed_indices: List[int] = []
            self.overlay.update_plan(
                steps=plan.get_titles_list(),
                active_idx=plan.active_index,
                completed=completed_indices,
            )

            total_steps = len(plan.steps)
            history: List[str] = []
            all_steps_successful = True

            # 3. Execute step-by-step
            while not plan.is_finished:
                current_step = plan.current_step
                if not current_step:
                    break

                flight_recorder.start_step(current_step.step_index, total_steps, current_step.title, current_step.description)
                log_step_start(current_step.step_index, total_steps, current_step.title, current_step.description)
                console.print(
                    f"\n[bold blue]─── Step {current_step.step_index}/{total_steps}: {current_step.title} ───[/bold blue]"
                )
                self.overlay.set_status(f"STEP {current_step.step_index}/{total_steps}: {current_step.title.upper()}")
                self.overlay.update_plan(
                    steps=plan.get_titles_list(),
                    active_idx=current_step.step_index,
                    completed=completed_indices,
                )

                step_attempts = 0
                max_step_attempts = 8
                step_completed = False
                last_action_key = ""  # Track last action to detect repeats
                actions_performed = []  # Track what we've done this step

                while step_attempts < max_step_attempts and not step_completed:
                    step_attempts += 1

                    screenshot = await self.driver.screenshot(
                        f"step_{current_step.step_index}_attempt_{step_attempts}.png"
                    )
                    screenshot_b64 = NimClient.pil_to_base64(screenshot)

                    # OCR & Astra a11y Tree Extraction
                    ocr_elements = ScreenGrounder.extract_screen_text_elements(screenshot)
                    element_count = len(ocr_elements)

                    from magnum.intelligence.a11y_tree import A11yEngine
                    from magnum.browser import get_browser_controller
                    browser_ctrl = get_browser_controller()

                    active_app = "Desktop"
                    try:
                        from magnum.driver.macos_ax import MacOSAccessibilityDriver
                        frontmost = MacOSAccessibilityDriver.get_frontmost_app()
                        if frontmost and frontmost.get("name"):
                            active_app = frontmost["name"]
                    except Exception:
                        pass
                    if active_app == "Desktop" and ocr_elements:
                        active_app = ocr_elements[0].text.strip()
                    a11y_tree = A11yEngine.build_tree(
                        image=screenshot,
                        active_app=active_app,
                        browser_controller=browser_ctrl,
                        ocr_elements=ocr_elements,
                    )
                    log_perception(active_app, element_count, len(a11y_tree.elements))

                    # Astra Set-of-Marks visual overlay
                    if a11y_tree.elements:
                        som_screenshot = ScreenGrounder.annotate_a11y_tree(screenshot, a11y_tree)
                        screenshot_b64 = NimClient.pil_to_base64(som_screenshot)
                        console.print(f"[dim]⚡ Astra a11y tree: {len(a11y_tree.elements)} interactive elements indexed[/dim]")
                    else:
                        console.print(f"[dim]🔍 OCR: {element_count} elements (Attempt {step_attempts})[/dim]")
                    if element_count > 0 and not a11y_tree.elements:
                        for i, el in enumerate(ocr_elements[:12], start=1):
                            text_display = el.text.strip()
                            if len(text_display) > 50:
                                text_display = text_display[:47] + "..."
                            console.print(f"[dim]  [{i}] \"{text_display}\" at ({el.center_x:.0f}, {el.center_y:.0f})[/dim]")
                        if element_count > 12:
                            console.print(f"[dim]  ... and {element_count - 12} more[/dim]")

                    # Pre-flight: detect if step is already done
                    if ocr_elements and step_attempts == 1:
                        step_lower = (current_step.title + " " + current_step.description).lower()
                        top_bar_texts = [el.text.strip().lower() for el in ocr_elements if el.top < 50]
                        frontmost_app = ocr_elements[0].text.strip().lower()

                        skip_app = self._should_auto_skip_app_open(
                            current_step=current_step,
                            active_app=active_app,
                            frontmost_app=frontmost_app,
                            top_bar_texts=top_bar_texts,
                        )
                        if skip_app:
                            console.print(f"[bold green]✓ '{skip_app}' is already visible/active on screen — skipping open step![/bold green]")
                            history.append(f"Step {current_step.step_index}: AUTO_SKIP - App active on screen: {skip_app}")
                            step_completed = True
                            break

                        # Check if this checklist step is a continuous background watcher step
                        if any(kw in step_lower for kw in ("watch and click", "watch for", "repeating until", "again and again", "until instructed to stop", "until i tell you to stop")):
                            target = self._clean_watcher_target(step_lower)
                            console.print(f"[bold green]👁️ Step is a continuous watcher: starting persistent background watcher for '{target}'...[/bold green]")
                            await self._start_watcher(instruction, {
                                "watch_for": target,
                                "action": "CLICK",
                                "stop_when": None,
                                "poll_interval": self.config.watcher_poll_interval,
                            })
                            history.append(f"Step {current_step.step_index}: Launched persistent background watcher for '{target}'")
                            step_completed = True
                            break

                        # Check if this checklist step is an internal workflow recording step
                        if any(kw in step_lower for kw in ("record custom workflow", "record workflow", "create workflow", "start workflow", "interactive voice workflow")):
                            console.print(f"[bold green]🎙️ Step is an internal workflow tool: launching interactive workflow recorder...[/bold green]")
                            return await self._interactive_create_workflow()

                    if step_completed:
                        break

                    # ── AUTO-INTERSTITIAL HANDLER ("Continue", "Get Started", "Agree") ──
                    interstitial_keywords = ("continue", "get started", "agree and continue", "agree", "let's go", "start using")
                    interstitial_clicked = False
                    for el in ocr_elements:
                        el_clean = el.text.strip().lower()
                        if el_clean in interstitial_keywords:
                            # Verify it's in a target app (e.g. WhatsApp, Chrome) or welcome dialog
                            if any(target in active_app.lower() for target in ("whatsapp", "chrome", "safari", "telegram", "slack")) or any("welcome" in o.text.lower() for o in ocr_elements[:10]):
                                console.print(f"[bold green]🔄 Detected interstitial button '{el.text}' in {active_app}: clicking to advance into app...[/bold green]")
                                await self.driver.click(el.center_x, el.center_y)
                                await asyncio.sleep(2.0)
                                interstitial_clicked = True
                                break
                    if interstitial_clicked:
                        continue

                    # Reasoning model
                    console.print(f"[dim]🧠 Reasoning...[/dim]")
                    action_data: GroundedAction = self.nim_client.ground_step_action(
                        screenshot_base64=screenshot_b64,
                        goal=instruction,
                        current_step=current_step,
                        total_steps=total_steps,
                        history=history,
                        ocr_elements=ocr_elements,
                        a11y_tree=a11y_tree,
                    )

                    console.print(f"[bold cyan]🧠 Thought:[/bold cyan] {action_data.thought}")
                    console.print(f"[bold yellow]⚡ Action:[/bold yellow] [bold]{action_data.action}[/bold]")
                    if action_data.text:
                        console.print(f"[dim]   Target: \"{action_data.text}\"[/dim]")

                    act_type = action_data.action.upper()

                    # ── ACTION DEDUPLICATION ──
                    # If the model suggests the exact same action+target as last time,
                    # it means the action already succeeded → auto-advance
                    current_action_key = f"{act_type}:{action_data.text or ''}"
                    if current_action_key == last_action_key and act_type in ("CLICK", "DOUBLE_CLICK", "TYPE", "PRESS_KEY"):
                        console.print(f"[bold green]✓ Action '{act_type}' already performed — advancing to next step![/bold green]")
                        step_completed = True
                        break
                    last_action_key = current_action_key

                    # ── POST-TYPE VERIFICATION ──
                    # If we typed text in a previous attempt, check if it's now on screen
                    typed_texts = [a[1] for a in actions_performed if a[0] == "TYPE"]
                    if typed_texts and act_type == "TYPE" and action_data.text:
                        # Check if previous typed text is now visible on screen
                        for prev_typed in typed_texts:
                            for el in ocr_elements:
                                if prev_typed.lower() in el.text.strip().lower():
                                    console.print(f"[bold green]✓ Text '{prev_typed}' already on screen — step done![/bold green]")
                                    step_completed = True
                                    break
                            if step_completed:
                                break
                        if step_completed:
                            break

                    if step_attempts >= 3 and "search" in current_step.title.lower():
                        query = action_data.text or "search query"
                        console.print(f"[bold green]🔍 Auto-search: '{query}'[/bold green]")
                        await self.driver.click(640.0, 130.0)
                        await asyncio.sleep(0.2)
                        await self.driver.hotkey("command", "a")
                        await self.driver.type_text(query)
                        await self.driver.press_key("return")
                        await asyncio.sleep(2.0)
                        step_completed = True
                        break

                    if act_type == "FINISH":
                        step_completed = True
                        plan.active_index = total_steps + 1
                        break

                    elif act_type == "STEP_DONE":
                        console.print(f"[bold green]✓ Step {current_step.step_index} done![/bold green]")
                        step_completed = True
                        break

                    elif act_type == "OPEN_APP":
                        app_name = action_data.text or "Google Chrome"
                        is_launch_step = any(kw in current_step.title.lower() for kw in ("open ", "launch ", "switch to ", "focus ")) and not any(
                            sub in current_step.title.lower() for sub in ("search", "chat", "message", "reply", "type", "send", "click")
                        )
                        already_open = False
                        try:
                            from magnum.driver.macos_ax import MacOSAccessibilityDriver
                            frontmost = MacOSAccessibilityDriver.get_frontmost_app()
                            if frontmost and frontmost.get("name"):
                                f_name = frontmost["name"].lower()
                                a_name = app_name.lower()
                                if a_name in f_name or f_name in a_name or any(w in f_name for w in a_name.split() if len(w) > 3):
                                    already_open = True
                        except Exception:
                            pass

                        if not already_open and ocr_elements:
                            frontmost_ocr = ocr_elements[0].text.strip().lower()
                            app_lower = app_name.lower()
                            if app_lower in frontmost_ocr or frontmost_ocr in app_lower or any(w in frontmost_ocr for w in app_lower.split() if len(w) > 3):
                                already_open = True

                        if not already_open:
                            console.print(f"[bold green]🚀 Opening: {app_name}[/bold green]")
                            await self.driver.navigate(app_name)
                            await asyncio.sleep(2.5)
                        else:
                            console.print(f"[bold green]✓ '{app_name}' already active[/bold green]")

                        if is_launch_step:
                            step_completed = True
                            break
                        else:
                            console.print(f"[bold cyan]📱 Brought '{app_name}' to foreground. Continuing step execution inside '{app_name}'...[/bold cyan]")
                            await asyncio.sleep(1.0)
                            continue

                    elif act_type == "NAVIGATE":
                        target = action_data.text or "https://google.com"
                        if not target.startswith("http") and (" " in target or "." not in target):
                            console.print(f"[bold green]🔍 Searching: '{target}'[/bold green]")
                            await self.driver.click(640.0, 130.0)
                            await self.driver.hotkey("command", "a")
                            await self.driver.type_text(target)
                            await self.driver.press_key("return")
                            await asyncio.sleep(2.0)
                        else:
                            console.print(f"[bold green]🌐 Navigating: {target}[/bold green]")
                            await self.driver.navigate(target)
                            await asyncio.sleep(2.0)

                    elif act_type == "SEARCH":
                        query = action_data.text or "search query"
                        console.print(f"[bold green]🔍 Searching: '{query}'[/bold green]")
                        self.overlay.set_status(f"SEARCHING: {query.upper()}")
                    
                        search_el_text = action_data.search_element
                        sx, sy = 640.0, 130.0
                        if search_el_text:
                            coords = ScreenGrounder.find_element_by_text(screenshot, search_el_text)
                            if coords:
                                sx, sy = coords
                        elif action_data.coordinates:
                            sx = action_data.coordinates.get("x", 640.0)
                            sy = action_data.coordinates.get("y", 130.0)
                    
                        await self.driver.click(sx, sy)
                        await asyncio.sleep(0.3)
                        await self.driver.hotkey("command", "a")
                        await asyncio.sleep(0.1)
                        await self.driver.type_text(query)
                        await asyncio.sleep(0.2)
                        await self.driver.press_key("return")
                        await asyncio.sleep(2.5)

                    elif act_type in ("REPORT", "READ"):
                        findings = action_data.text or action_data.thought
                        console.print(f"\n[bold green]📊 FINDINGS:[/bold green]\n{findings}\n")
                        self.overlay.set_status("FINDINGS READY")
                        await self.hitl_handler.confirm_comment_async(
                            draft_comment=findings,
                            author="Search Results",
                            post_summary=instruction,
                        )
                        step_completed = True
                        plan.active_index = total_steps + 1
                        break

                    elif act_type == "OBSTACLE_DETECTED":
                        obstacle_msg = action_data.text or "An obstacle on screen requires your attention. Please check your screen."
                        console.print(f"\n[bold yellow]════════════════════════════════════════════════════════════[/bold yellow]")
                        console.print(f"[bold yellow]🔐 OBSTACLE DETECTED BY MODEL:[/bold yellow] {action_data.thought}")
                        console.print(f"[bold cyan]🗣️  Speaking to user:[/bold cyan] '{obstacle_msg}'")
                        console.print(f"[bold yellow]════════════════════════════════════════════════════════════[/bold yellow]\n")

                        self.voice_engine.speak(obstacle_msg)
                        self.overlay.set_status(f"⚠️ {obstacle_msg[:35].upper()}...")

                        # Closed-loop visual monitoring: wait until user resolves obstacle on screen
                        resolved = await self._wait_for_obstacle_resolution(
                            baseline_ocr=ocr_elements,
                            prompt_msg=obstacle_msg,
                            max_wait_seconds=120,
                        )
                        if resolved:
                            confirm_msg = "Obstacle cleared. Continuing with your task."
                            self.voice_engine.speak(confirm_msg)
                            console.print("[bold green]✅ Visual state change confirmed! Resuming execution...[/bold green]")
                            self.overlay.set_status("RESUMING TASK")
                            await asyncio.sleep(1.0)
                            # Re-perceive the new screen state
                            screenshot = await self.driver.screenshot(f"step_{current_step.step_index}_post_obstacle.png")
                            ocr_elements = ScreenGrounder.extract_screen_text_elements(screenshot)
                            element_count = len(ocr_elements)
                            try:
                                from magnum.driver.macos_ax import MacOSAccessibilityDriver
                                frontmost = MacOSAccessibilityDriver.get_frontmost_app()
                                if frontmost and frontmost.get("name"):
                                    active_app = frontmost["name"]
                            except Exception:
                                pass
                            a11y_tree = A11yEngine.build_tree(
                                image=screenshot,
                                active_app=active_app,
                                browser_controller=browser_ctrl,
                                ocr_elements=ocr_elements,
                            )
                            log_perception(active_app, element_count, len(a11y_tree.elements))
                            continue
                        else:
                            console.print(f"[bold red]❌ Obstacle wait timed out after 120s[/bold red]")
                            self.voice_engine.speak("Wait timed out. Please try again.")
                            step_completed = False
                            break

                    elif act_type == "ASK_USER":
                        draft_text = action_data.text or action_data.thought
                        options = action_data.options
                        if options and any("approve" in o.lower() or "post" in o.lower() for o in options):
                            decision = await self.hitl_handler.confirm_comment_async(
                                draft_comment=draft_text, author=current_step.title, post_summary=instruction,
                            )
                            if decision.action == HitlActionType.APPROVE:
                                step_completed = True
                            else:
                                return False
                        else:
                            user_ans = await self.hitl_handler.ask_user_text_async(draft_text)
                            console.print(f"[bold green]✓ User: '{user_ans}'[/bold green]")
                            history.append(f"Agent Asked: {draft_text} | User: {user_ans}")

                            # Store in persistent memory & knowledge
                            from magnum.intelligence.memory import get_memory
                            mem = get_memory()
                            mem.learn_user_rule(f"When asked '{draft_text}', user answered: '{user_ans}'")
                            if "antigravity" in draft_text.lower() or "antigravity" in user_ans.lower():
                                mem.learn_app_mapping("antigravity", "Antigravity IDE")
                                mem.learn_user_rule("Antigravity is already open and focused in the current workspace.")
                            await asyncio.sleep(1.0)

                    elif act_type in ("CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"):
                        # ── SAFETY GUARD: APP LAUNCH REDIRECTION ──
                        # If current step is to open/launch an app, and the model attempts to click
                        # on editor code text matching the app name or step title inside Antigravity / editor,
                        # intercept and perform native system app launch instead!
                        step_title_lower = current_step.title.lower()
                        is_launch_step = any(kw in step_title_lower for kw in ("open ", "launch ", "start ", "focus ")) and not any(
                            sub in step_title_lower for sub in ("search", "chat", "message", "reply", "type", "send")
                        )
                        target_id = action_data.target_id
                        target_el = a11y_tree.get_element_by_id(target_id) if (target_id and a11y_tree) else None

                        if is_launch_step:
                            target_text = (action_data.text or "").strip().lower()
                            if target_el and target_el.label:
                                target_text = target_el.label.strip().lower()
                            
                            # Check if user/agent is currently inside an IDE / text editor
                            if any(ide in active_app.lower() for ide in ("antigravity", "code", "terminal", "sublime", "editor")):
                                known_apps = ["whatsapp", "chrome", "google chrome", "safari", "spotify", "slack", "discord", "notes", "finder", "terminal"]
                                target_app = next((app for app in known_apps if app in step_title_lower or app in target_text), None)
                                if target_app:
                                    console.print(f"[bold yellow]🛡️ Intercepted click on editor text '{target_text}'! Redirecting to native launch: '{target_app.title()}'[/bold yellow]")
                                    await self.driver.navigate(target_app.title())
                                    await asyncio.sleep(2.5)
                                    step_completed = True
                                    break

                        x, y = 0.0, 0.0

                        # ── MULTI-TIER EXECUTION CASCADE ──
                        # Tier 1: Browser DOM dispatch (Chrome/Safari)
                        handled = False
                        if target_id and target_el and target_el.source == "browser_dom":
                            handled = browser_ctrl.click_element_by_id(target_id)
                            if handled:
                                console.print(f"[bold green]⚡ Astra DOM Click: [{target_id}] '{target_el.label}'[/bold green]")

                        # Tier 2: Open Computer Use (OCU / Codex) Engine Bridge
                        if not handled and target_id:
                            try:
                                from magnum.driver.open_computer_use_bridge import OpenComputerUseBridge
                                if OpenComputerUseBridge.is_available():
                                    handled = OpenComputerUseBridge.click(app=active_app, element_index=str(target_id))
                                    if handled:
                                        console.print(f"[bold green]👾 Open-Computer-Use Click: [{target_id}] in {active_app}[/bold green]")
                            except Exception as e:
                                logger.debug(f"OpenComputerUseBridge click error: {e}")

                        # Tier 3: Native macOS AXUIElement dispatch (Finder, Settings, Antigravity, etc.)
                        if not handled and target_id and target_el and target_el.source == "macos_ax" and target_el.native_element:
                            try:
                                from magnum.driver.macos_ax import MacOSAccessibilityDriver
                                handled = MacOSAccessibilityDriver.press_ax_element(target_el.native_element)
                                if handled:
                                    console.print(f"[bold green]🍏 Astra Native AX Press: [{target_id}] '{target_el.label}'[/bold green]")
                            except Exception as e:
                                logger.debug(f"Native AX click error: {e}")

                        # Tier 4: Physical mouse click fallback
                        if not handled:
                            if target_el:
                                x, y = target_el.center_x, target_el.center_y
                                console.print(f"[bold green]🎯 Astra Target [{target_id}]: '{target_el.label}' → ({x:.0f}, {y:.0f})[/bold green]")
                            elif action_data.text:
                                exact = ScreenGrounder.find_element_by_text(screenshot, action_data.text)
                                if exact:
                                    x, y = exact
                                    console.print(f"[bold green]🎯 OCR: '{action_data.text}' → ({x:.0f}, {y:.0f})[/bold green]")
                                else:
                                    console.print(f"[bold yellow]⚠️ OCR can't find '{action_data.text}'[/bold yellow]")

                            if x == 0.0 and y == 0.0 and action_data.coordinates:
                                x = action_data.coordinates.get("x", 0.0)
                                y = action_data.coordinates.get("y", 0.0)

                            if x == 0.0 and y == 0.0:
                                console.print(f"[bold red]❌ Cannot resolve click target[/bold red]")
                            else:
                                if act_type == "CLICK":
                                    await self.driver.click(x, y)
                                elif act_type == "DOUBLE_CLICK":
                                    await self.driver.double_click(x, y)
                                elif act_type == "RIGHT_CLICK":
                                    await self.driver.right_click(x, y)

                    elif act_type == "TYPE" and action_data.text:
                        t = action_data.text
                        target_id = action_data.target_id
                        target_el = a11y_tree.get_element_by_id(target_id) if (target_id and a11y_tree) else None

                        # ── MULTI-TIER TYPE CASCADE ──
                        # Tier 1: Browser DOM type
                        typed = False
                        if target_id and target_el and target_el.source == "browser_dom":
                            typed = browser_ctrl.type_element_by_id(target_id, t)
                            if typed:
                                console.print(f"[bold green]⚡ Astra DOM Type into [{target_id}]: '{t[:60]}'[/bold green]")

                        # Tier 2: Open Computer Use (OCU / Codex) Engine Bridge
                        if not typed and target_id:
                            try:
                                from magnum.driver.open_computer_use_bridge import OpenComputerUseBridge
                                if OpenComputerUseBridge.is_available():
                                    typed = OpenComputerUseBridge.set_value(app=active_app, element_index=str(target_id), value=t)
                                    if typed:
                                        console.print(f"[bold green]👾 Open-Computer-Use Set Value: [{target_id}] in {active_app}[/bold green]")
                            except Exception as e:
                                logger.debug(f"OpenComputerUseBridge set_value error: {e}")

                        # Tier 3: Native macOS AXUIElement value set
                        if not typed and target_id and target_el and target_el.source == "macos_ax" and target_el.native_element:
                            try:
                                from magnum.driver.macos_ax import MacOSAccessibilityDriver
                                typed = MacOSAccessibilityDriver.set_ax_element_value(target_el.native_element, t)
                                if typed:
                                    console.print(f"[bold green]🍏 Astra Native AX Set Value [{target_id}]: '{t[:60]}'[/bold green]")
                            except Exception as e:
                                logger.debug(f"Native AX set value error: {e}")

                        # Tier 4: Physical click + keyboard typing
                        if not typed:
                            if target_el:
                                await self.driver.click(target_el.center_x, target_el.center_y)
                                await asyncio.sleep(0.15)
                            console.print(f"[bold green]⌨️ Typing: '{t[:60]}{'...' if len(t)>60 else ''}'[/bold green]")
                            await self.driver.type_text(t)

                    elif act_type == "PRESS_KEY" and action_data.key:
                        handled_key = False
                        try:
                            from magnum.driver.open_computer_use_bridge import OpenComputerUseBridge
                            if OpenComputerUseBridge.is_available():
                                handled_key = OpenComputerUseBridge.press_key(app=active_app, key=action_data.key)
                        except Exception:
                            pass
                        if not handled_key:
                            await self.driver.press_key(action_data.key)

                    elif act_type == "HOTKEY" and action_data.hotkeys:
                        step_text = (current_step.title + " " + current_step.description).lower()
                        if "navigate" in step_text or "open" in step_text:
                            url_map = {"youtube": "https://www.youtube.com", "gmail": "https://mail.google.com",
                                       "linkedin": "https://www.linkedin.com", "github": "https://github.com"}
                            url = next((v for k, v in url_map.items() if k in step_text), "https://google.com")
                            await self.driver.navigate(url)
                        else:
                            await self.driver.hotkey(*action_data.hotkeys)

                    elif act_type == "SCROLL":
                        handled_scroll = False
                        try:
                            from magnum.driver.open_computer_use_bridge import OpenComputerUseBridge
                            if OpenComputerUseBridge.is_available():
                                handled_scroll = OpenComputerUseBridge.scroll(
                                    app=active_app,
                                    direction=action_data.scroll_direction or "down",
                                )
                        except Exception:
                            pass
                        if not handled_scroll:
                            await self.driver.scroll(direction=action_data.scroll_direction or "down", amount=action_data.scroll_amount or 300)

                    elif act_type == "WAIT":
                        await asyncio.sleep(1.5)

                    # Track performed actions for deduplication
                    actions_performed.append((act_type, action_data.text or ""))
                    history.append(f"Step {current_step.step_index}: {action_data.action} - {action_data.thought}")

                    # ── ASTRA CLOSED-LOOP STATE VALIDATION ("Check Its Work") ──
                    await asyncio.sleep(self.config.step_delay)
                    try:
                        verify_screenshot = await self.driver.screenshot(f"verify_step_{current_step.step_index}.png")
                        verify_ocr = ScreenGrounder.extract_screen_text_elements(verify_screenshot)

                        if act_type == "TYPE" and action_data.text:
                            typed_lower = action_data.text.strip().lower()
                            if any(typed_lower in el.text.strip().lower() for el in verify_ocr):
                                console.print(f"[bold green]✓ Astra Verification: Text '{action_data.text[:30]}' confirmed on screen[/bold green]")
                                step_completed = True
                        elif act_type in ("CLICK", "DOUBLE_CLICK"):
                            # If this was an app launch step, verify that the frontmost app actually changed to target app
                            step_title_lower = current_step.title.lower()
                            is_launch_step = any(kw in step_title_lower for kw in ("open ", "launch ", "start ", "focus ")) and not any(
                                sub in step_title_lower for sub in ("search", "chat", "message", "reply", "type", "send")
                            )
                            if is_launch_step:
                                try:
                                    from magnum.driver.macos_ax import MacOSAccessibilityDriver
                                    front = MacOSAccessibilityDriver.get_frontmost_app()
                                    front_name = (front.get("name") or "").lower() if front else ""
                                    target_words = [w for w in step_title_lower.split() if w not in ("open", "launch", "the", "app", "application", "window", "desktop")]
                                    if target_words and any(w in front_name for w in target_words):
                                        console.print(f"[bold green]✓ Astra Verification: Target app '{front_name}' is now active foreground[/bold green]")
                                        step_completed = True
                                except Exception:
                                    pass
                    except Exception as e:
                        logger.debug(f"Astra verification check skipped: {e}")

                    # Record this exact attempt in the flight recorder
                    from magnum.logger import flight_recorder
                    flight_recorder.record_attempt(
                        step_index=current_step.step_index,
                        attempt_number=step_attempts,
                        active_app=active_app,
                        screenshot=f"step_{current_step.step_index}_attempt_{step_attempts}.png",
                        ocr_count=element_count,
                        a11y_targets_count=len(a11y_tree.elements) if (a11y_tree and a11y_tree.elements) else 0,
                        ai_thought=action_data.thought,
                        chosen_action=act_type,
                        execution_tier="Multi-Tier Astra Cascade",
                        target_id=action_data.target_id,
                        target_label=action_data.text,
                        target_coords=action_data.coordinates,
                        input_text=action_data.text,
                        key_pressed=action_data.key,
                        execution_success=step_completed or (act_type in ("WAIT", "ASK_USER", "OBSTACLE_DETECTED")),
                        verification_confirmed=step_completed,
                        verification_note="Closed-loop verification confirmed state change" if step_completed else "Awaiting visual confirmation",
                        failure_reason=None if step_completed else f"Attempt {step_attempts} not yet verified",
                    )

                # Mark complete on HUD
                if not step_completed:
                    all_steps_successful = False
                    log_failure(
                        context=f"Step {current_step.step_index} exceeded {max_step_attempts} attempts without verification",
                        error=f"Step '{current_step.title}' not verified as completed.",
                        step_info=f"[{current_step.step_index}/{total_steps}] {current_step.title}",
                        screenshot_path=f"step_{current_step.step_index}_attempt_{step_attempts}.png",
                    )
                    flight_recorder.complete_step(current_step.step_index, success=False)
                    console.print(f"[bold red]❌ Step {current_step.step_index} failed after {step_attempts} attempts![/bold red]")
                else:
                    flight_recorder.complete_step(current_step.step_index, success=True)
                    completed_indices.append(current_step.step_index)
                    console.print(f"[bold green]✓ Step {current_step.step_index} completed![/bold green]")

                plan.advance_to_next_step()
                self.overlay.update_plan(
                    steps=plan.get_titles_list(),
                    active_idx=plan.active_index,
                    completed=completed_indices,
                )

            # Done
            if all_steps_successful:
                self.overlay.set_status("GOAL ACCOMPLISHED")
                self.overlay.flash()
                self.hitl_handler.notify("Goal Completed!", f"Done: {instruction}")
                try:
                    from magnum.notifications import send_notification
                    send_notification("⚡ Magnum: Goal Completed", f"Done: {instruction}", sound="Glass")
                except Exception:
                    pass
                self.voice_engine.speak(f"Finished. {instruction}")
                console.print(f"\n[bold green]🎉 Done: {instruction}[/bold green]\n")
                log_task_complete(instruction, success=True, duration_seconds=time.time() - start_exec_time)
                flight_recorder.finish_task(success=True)

                # Show watcher status
                watchers = self.task_queue.get_active_watchers()
                if watchers:
                    console.print(f"[dim]👁️ {len(watchers)} background watcher(s) resumed[/dim]")

                return True
            else:
                self.overlay.set_status("TASK INCOMPLETE")
                self.voice_engine.speak(f"Could not complete all steps for {instruction}. Please check the screen.")
                console.print(f"\n[bold red]⚠️ Task incomplete: some steps could not be verified.[/bold red]\n")
                log_task_complete(instruction, success=False, duration_seconds=time.time() - start_exec_time)
                flight_recorder.finish_task(success=False)
                return False

        except Exception as e:
            log_failure(
                context=f"Fatal exception executing instruction: '{instruction}'",
                error=e,
            )
            log_task_complete(instruction, success=False, duration_seconds=time.time() - start_exec_time)
            flight_recorder.record_failure(
                context=f"Fatal exception executing instruction: '{instruction}'",
                error=e,
            )
            flight_recorder.finish_task(success=False)
            console.print(f"[bold red]❌ Error executing task:[/bold red] {e}")
            self.voice_engine.speak("An error occurred during task execution. Check magnum log.")
            return False
