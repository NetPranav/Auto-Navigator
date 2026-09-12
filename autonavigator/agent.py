"""
Auto-Navigator Master Agent.
Executes user instructions via dynamic AI planning, on-screen live checklist tracking,
Apple Vision OCR-first screen perception, and bottom-center interactive cards.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from autonavigator.config import config, AppConfig
from autonavigator.drivers.base import BaseDriver
from autonavigator.drivers.desktop_driver import DesktopDriver
from autonavigator.drivers.headless_driver import HeadlessDriver
from autonavigator.hitl.hitl_handler import HitlHandler, HitlActionType
from autonavigator.intelligence.grounding import ScreenGrounder, TextElement
from autonavigator.intelligence.nim_client import NimClient, GroundedAction
from autonavigator.intelligence.planner import PlanChecklist, PlanStep
from autonavigator.ui import get_overlay

logger = logging.getLogger(__name__)
console = Console()


class AutoNavigatorAgent:
    """Master AI agent with dynamic planning, live HUD checklist, and OCR-first vision grounding."""

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

        if self.mode == "desktop":
            self.driver: BaseDriver = DesktopDriver()
        else:
            self.driver = HeadlessDriver(
                headless=False,  # Visible browser for visual feedback
                user_data_dir=self.config.browser_profile_path,
                viewport_width=self.config.viewport_width,
                viewport_height=self.config.viewport_height,
            )

    async def __aenter__(self) -> AutoNavigatorAgent:
        await self.driver.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.driver.close()

    async def execute(self, instruction: str) -> bool:
        """Dynamically plan and execute the user's task step-by-step."""
        console.print(f"\n[bold cyan]🎯 Goal:[/bold cyan] {instruction}")
        console.print("[dim]🧠 Generating dynamic AI plan checklist with NVIDIA NIM...[/dim]")

        # 1. Generate dynamic plan checklist
        plan: PlanChecklist = self.nim_client.generate_plan(instruction)

        # Print plan table in terminal
        table = Table(title="📋 AI PLAN CHECKLIST", border_style="cyan")
        table.add_column("Step", style="bold yellow", width=6)
        table.add_column("Title", style="bold white")
        table.add_column("Description", style="dim")
        for s in plan.steps:
            table.add_row(str(s.step_index), s.title, s.description)
        console.print(table)

        # 2. Display checklist on top-left screen HUD
        completed_indices: List[int] = []
        self.overlay.update_plan(
            steps=plan.get_titles_list(),
            active_idx=plan.active_index,
            completed=completed_indices,
        )

        total_steps = len(plan.steps)
        history: List[str] = []

        # 3. Execute step-by-step
        while not plan.is_finished:
            current_step = plan.current_step
            if not current_step:
                break

            console.print(
                f"\n[bold blue]─── Executing Step {current_step.step_index}/{total_steps}: {current_step.title} ───[/bold blue]"
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

                # Capture screen (triggers flash & audio)
                screenshot = await self.driver.screenshot(
                    f"step_{current_step.step_index}_attempt_{step_attempts}.png"
                )
                screenshot_b64 = NimClient.pil_to_base64(screenshot)

                # OCR-First: Extract all text elements from screen
                ocr_elements = ScreenGrounder.extract_screen_text_elements(screenshot)
                element_count = len(ocr_elements)

                # Print OCR manifest in terminal for transparency
                console.print(f"[dim]🔍 Apple Vision OCR: {element_count} elements detected (Attempt {step_attempts})[/dim]")
                if element_count > 0:
                    # Print top 15 most relevant elements for readability
                    display_elements = ocr_elements[:15]
                    for i, el in enumerate(display_elements, start=1):
                        text_display = el.text.strip()
                        if len(text_display) > 50:
                            text_display = text_display[:47] + "..."
                        console.print(f"[dim]  [{i}] \"{text_display}\" at ({el.center_x:.0f}, {el.center_y:.0f})[/dim]")
                    if element_count > 15:
                        console.print(f"[dim]  ... and {element_count - 15} more elements[/dim]")

                # ── Pre-flight intelligence: detect if step is already done ──
                # The first OCR element is always the macOS menu bar app name
                if ocr_elements and step_attempts == 1:
                    frontmost_app = ocr_elements[0].text.strip().lower()
                    step_lower = (current_step.title + " " + current_step.description).lower()
                    
                    # Known desktop apps (not websites)
                    desktop_apps = [
                        "antigravity", "vs code", "visual studio code", "xcode",
                        "terminal", "finder", "spotify", "discord", "slack",
                        "notes", "textedit", "preview", "system settings",
                        "safari", "google chrome", "chrome", "firefox", "arc",
                        "iterm", "warp", "sublime", "atom", "intellij",
                    ]
                    
                    # Check if this step is trying to "open" or "navigate to" something
                    # that's actually already the active desktop app
                    if any(kw in step_lower for kw in ["open", "launch", "navigate to", "switch to"]):
                        for app in desktop_apps:
                            if app in step_lower and app in frontmost_app:
                                console.print(f"[bold green]✓ Pre-flight: '{ocr_elements[0].text.strip()}' is already the active app — skipping step![/bold green]")
                                history.append(f"Step {current_step.step_index}: AUTO_SKIP - App already active: {ocr_elements[0].text.strip()}")
                                step_completed = True
                                break
                        # Also check if the frontmost app name appears in the step text
                        if not step_completed:
                            for word in frontmost_app.split():
                                if len(word) > 3 and word in step_lower:
                                    console.print(f"[bold green]✓ Pre-flight: '{ocr_elements[0].text.strip()}' matches step target — skipping![/bold green]")
                                    history.append(f"Step {current_step.step_index}: AUTO_SKIP - App already active: {ocr_elements[0].text.strip()}")
                                    step_completed = True
                                    break
                
                if step_completed:
                    break

                # Query reasoning model with OCR manifest
                console.print(f"[dim]🧠 Sending OCR manifest to reasoning model...[/dim]")
                action_data: GroundedAction = self.nim_client.ground_step_action(
                    screenshot_base64=screenshot_b64,
                    goal=instruction,
                    current_step=current_step,
                    total_steps=total_steps,
                    history=history,
                    ocr_elements=ocr_elements,
                )

                console.print(f"[bold cyan]🧠 Reasoning Thought:[/bold cyan] {action_data.thought}")
                console.print(f"[bold yellow]⚡ Next Action:[/bold yellow] [bold]{action_data.action}[/bold]")
                if action_data.text:
                    console.print(f"[dim]   Target: \"{action_data.text}\"[/dim]")

                # Handle action types
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

                # Loop breaker: If model has tried multiple times on a search step, auto-execute search
                if step_attempts >= 3 and "search" in current_step.title.lower():
                    query = action_data.text or "search query"
                    console.print(f"[bold green]🔍 Submitting Search for: '{query}'[/bold green]")
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
                    console.print(f"[bold green]✓ Step {current_step.step_index} visually verified complete on screen![/bold green]")
                    step_completed = True
                    break

                elif act_type == "OPEN_APP":
                    app_name = action_data.text or "Google Chrome"
                    
                    # Smart check: is this app already the active foreground app?
                    # The first OCR element is always the macOS menu bar app name
                    already_open = False
                    if ocr_elements:
                        frontmost_app = ocr_elements[0].text.strip().lower()
                        app_name_lower = app_name.lower()
                        # Fuzzy match app names (e.g. "Antigravity IDE" matches "antigravity")
                        if (app_name_lower in frontmost_app 
                            or frontmost_app in app_name_lower
                            or any(word in frontmost_app for word in app_name_lower.split())):
                            already_open = True
                            console.print(f"[bold green]✓ '{app_name}' is already the active app (detected: '{ocr_elements[0].text.strip()}')[/bold green]")
                            step_completed = True
                            break
                    
                    if not already_open:
                        console.print(f"[bold green]🚀 Opening App: {app_name}[/bold green]")
                        await self.driver.navigate(app_name)
                        await asyncio.sleep(2.0)

                elif act_type == "NAVIGATE":
                    target = action_data.text or "https://google.com"
                    if not target.startswith("http://") and not target.startswith("https://") and (" " in target or "." not in target):
                        # Misclassified search query
                        console.print(f"[bold green]🔍 Executing Search: '{target}'[/bold green]")
                        await self.driver.click(640.0, 130.0)
                        await self.driver.hotkey("command", "a")
                        await self.driver.type_text(target)
                        await self.driver.press_key("return")
                        await asyncio.sleep(2.0)
                    else:
                        console.print(f"[bold green]🌐 Navigating to URL: {target}[/bold green]")
                        await self.driver.navigate(target)
                        await asyncio.sleep(2.0)

                elif act_type == "SEARCH":
                    query = action_data.text or "search query"
                    console.print(f"[bold green]🔍 Executing Search: '{query}'[/bold green]")
                    self.overlay.set_status(f"SEARCHING: {query.upper()}")
                    
                    # Try to find search bar by OCR element text
                    search_el_text = action_data.search_element
                    sx, sy = 640.0, 130.0  # Default fallback
                    
                    if search_el_text:
                        coords = ScreenGrounder.find_element_by_text(screenshot, search_el_text)
                        if coords:
                            sx, sy = coords
                            console.print(f"[bold green]🎯 Found search bar '{search_el_text}' at ({sx:.0f}, {sy:.0f})[/bold green]")
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
                    console.print(f"\n[bold green]📊 SUMMARY OF FINDINGS:[/bold green]\n{findings}\n")
                    self.overlay.set_status("FINDINGS SUMMARY READY")
                    
                    await self.hitl_handler.confirm_comment_async(
                        draft_comment=findings,
                        author="Search & Inspection Results",
                        post_summary=instruction,
                    )
                    step_completed = True
                    plan.active_index = total_steps + 1
                    break

                elif act_type == "CLICK_AND_TYPE":
                    # First click the target element
                    target_text = action_data.text
                    x, y = 0.0, 0.0
                    if target_text:
                        coords = ScreenGrounder.find_element_by_text(screenshot, target_text)
                        if coords:
                            x, y = coords
                    if x == 0.0 and y == 0.0 and action_data.coordinates:
                        x = action_data.coordinates.get("x", 0)
                        y = action_data.coordinates.get("y", 0)
                    if x > 0 and y > 0:
                        await self.driver.click(x, y)
                        await asyncio.sleep(0.2)
                    if action_data.text:
                        await self.driver.type_text(action_data.text)
                    if action_data.key:
                        await self.driver.press_key(action_data.key)
                    await asyncio.sleep(1.0)

                elif act_type == "OBSTACLE_DETECTED":
                    console.print(f"[bold yellow]🔐 Obstacle / Authentication Required:[/bold yellow] {action_data.text}")
                    self.overlay.set_status("🔐 LOGIN / AUTHENTICATION REQUIRED")
                    
                    decision = await self.hitl_handler.confirm_comment_async(
                        draft_comment=action_data.text or "Please sign in to your account on screen. Click Continue when ready.",
                        author="Authentication Required",
                        post_summary="Login Obstacle",
                    )
                    if decision.action == HitlActionType.APPROVE:
                        console.print("[bold green]✓ Login confirmed by user. Resuming plan...[/bold green]")
                        await asyncio.sleep(2.0)
                    else:
                        console.print("[bold red]❌ Task cancelled by user.[/bold red]")
                        self.hitl_handler.notify("Task Cancelled", "Cancelled at login step.")
                        return False

                elif act_type == "ASK_USER":
                    draft_text = action_data.text or action_data.thought
                    options = action_data.options
                    
                    if options and any("approve" in opt.lower() or "post" in opt.lower() for opt in options):
                        decision = await self.hitl_handler.confirm_comment_async(
                            draft_comment=draft_text,
                            author=current_step.title,
                            post_summary=instruction,
                        )
                        if decision.action == HitlActionType.APPROVE:
                            console.print("[bold green]✓ Approved by user via on-screen card![/bold green]")
                            step_completed = True
                        else:
                            console.print("[bold red]❌ Action cancelled by user.[/bold red]")
                            self.hitl_handler.notify("Action Cancelled", f"Cancelled by user at step: {current_step.title}")
                            return False
                    else:
                        # User clarifying question (e.g. asking where file/folder is located)
                        user_ans = await self.hitl_handler.ask_user_text_async(draft_text)
                        console.print(f"[bold green]✓ Received User Response:[/bold green] '{user_ans}'")
                        history.append(f"Agent Asked: {draft_text} | User Answered: {user_ans}")
                        await asyncio.sleep(1.0)

                elif act_type in ("CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"):
                    x, y = 0.0, 0.0
                    target_text = action_data.text
                    
                    # 1. High-Precision OCR text grounding (primary)
                    if target_text:
                        exact_coords = ScreenGrounder.find_element_by_text(screenshot, target_text)
                        if exact_coords:
                            x, y = exact_coords
                            console.print(f"[bold green]🎯 Apple Vision Grounded '{target_text}' → ({x:.0f}, {y:.0f})[/bold green]")
                        else:
                            console.print(f"[bold yellow]⚠️  OCR could not find '{target_text}' on screen[/bold yellow]")
                    
                    # 2. Coordinate fallback if not grounded by text
                    if x == 0.0 and y == 0.0 and action_data.coordinates:
                        x = action_data.coordinates.get("x", 0.0)
                        y = action_data.coordinates.get("y", 0.0)
                        console.print(f"[dim]   Using coordinate fallback: ({x:.0f}, {y:.0f})[/dim]")
                    
                    # If still 0,0, skip this click (don't click randomly)
                    if x == 0.0 and y == 0.0:
                        console.print(f"[bold red]❌ Cannot resolve click target. Skipping.[/bold red]")
                    else:
                        if act_type == "CLICK":
                            await self.driver.click(x, y)
                        elif act_type == "DOUBLE_CLICK":
                            await self.driver.double_click(x, y)
                        elif act_type == "RIGHT_CLICK":
                            await self.driver.right_click(x, y)

                elif act_type == "TYPE" and action_data.text:
                    console.print(f"[bold green]⌨️  Typing: '{action_data.text[:60]}...'[/bold green]" if len(action_data.text) > 60 else f"[bold green]⌨️  Typing: '{action_data.text}'[/bold green]")
                    await self.driver.type_text(action_data.text)

                elif act_type == "PRESS_KEY" and action_data.key:
                    await self.driver.press_key(action_data.key)

                elif act_type == "HOTKEY" and action_data.hotkeys:
                    step_text = (current_step.title + " " + current_step.description).lower()
                    if "navigate" in step_text or "open" in step_text:
                        if "youtube" in step_text:
                            url = "https://www.youtube.com"
                        elif "gmail" in step_text:
                            url = "https://mail.google.com"
                        elif "linkedin" in step_text:
                            url = "https://www.linkedin.com"
                        elif "github" in step_text:
                            url = "https://github.com"
                        else:
                            url = "https://google.com"
                        console.print(f"[bold green]🌐 Fast Navigating to URL: {url}[/bold green]")
                        await self.driver.navigate(url)
                    else:
                        await self.driver.hotkey(*action_data.hotkeys)

                elif act_type == "SCROLL":
                    direction = action_data.scroll_direction or "down"
                    amount = action_data.scroll_amount or 300
                    await self.driver.scroll(direction=direction, amount=amount)

                elif act_type == "WAIT":
                    await asyncio.sleep(1.5)

                # Track performed actions for deduplication
                actions_performed.append((act_type, action_data.text or ""))
                history.append(f"Step {current_step.step_index}: {action_data.action} - {action_data.thought}")
                await asyncio.sleep(self.config.step_delay)

            # Mark step completed on HUD
            completed_indices.append(current_step.step_index)
            plan.advance_to_next_step()
            self.overlay.update_plan(
                steps=plan.get_titles_list(),
                active_idx=plan.active_index,
                completed=completed_indices,
            )
            console.print(f"[bold green]✓ Step {current_step.step_index} completed![/bold green]")

        # 4. Final completion
        self.overlay.set_status("GOAL ACCOMPLISHED")
        self.overlay.flash()
        self.hitl_handler.notify("Goal Completed!", f"Successfully accomplished: {instruction}")
        console.print(f"\n[bold green]🎉 Task Accomplished successfully: {instruction}[/bold green]\n")
        return True
