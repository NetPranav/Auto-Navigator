"""
Generic vision-driven autonomous navigation workflow.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional
from rich.console import Console

from autonavigator.config import config
from autonavigator.drivers.base import BaseDriver
from autonavigator.hitl.hitl_handler import HitlHandler
from autonavigator.intelligence.grounding import ScreenGrounder
from autonavigator.intelligence.nim_client import NimClient, GroundedAction

logger = logging.getLogger(__name__)
console = Console()


class GenericWorkflow:
    """Perceive -> Reason -> Act loop driven by NVIDIA NIM vision grounding."""

    def __init__(
        self,
        driver: BaseDriver,
        nim_client: Optional[NimClient] = None,
        hitl_handler: Optional[HitlHandler] = None,
    ) -> None:
        self.driver = driver
        self.nim_client = nim_client or NimClient()
        self.hitl_handler = hitl_handler or HitlHandler()

    async def run(self, goal: str, max_steps: Optional[int] = None) -> bool:
        """Run the autonomous vision navigation loop."""
        steps_limit = max_steps or config.max_steps
        history: List[str] = []

        console.print(f"[bold cyan]🎯 Goal:[/bold cyan] {goal}")
        console.print(f"[dim]Starting autonomous vision loop (max {steps_limit} steps)...[/dim]\n")

        for step in range(1, steps_limit + 1):
            console.print(f"[bold blue]─── Step {step}/{steps_limit} ───[/bold blue]")

            # 1. Capture screen
            screenshot = await self.driver.screenshot(f"loop_step_{step:03d}.png")
            screenshot_b64 = NimClient.pil_to_base64(screenshot)

            # 2. Get next grounded action from NIM
            console.print("[dim]Analyzing screen with NVIDIA NIM vision...[/dim]")
            action_data: GroundedAction = self.nim_client.ground_action(
                screenshot_base64=screenshot_b64,
                goal=goal,
                history=history,
            )

            console.print(f"[bold cyan]Thought:[/bold cyan] {action_data.thought}")
            console.print(f"[bold yellow]Action:[/bold yellow] [bold]{action_data.action}[/bold]")

            # Check if task is finished
            if action_data.action == "FINISH" or action_data.is_terminal:
                console.print(f"\n[bold green]🎉 Goal Accomplished in {step} steps![/bold green]")
                self.hitl_handler.notify("Goal Completed", f"Successfully completed: {goal}")
                return True

            # 3. Execute action
            await self._execute_action(action_data, screenshot.width, screenshot.height)
            history.append(f"Step {step}: {action_data.action} - {action_data.thought}")

            # Brief delay between steps
            await asyncio.sleep(config.step_delay)

        console.print(f"[bold red]⚠️ Max steps reached ({steps_limit}). Task ended.[/bold red]")
        return False

    async def _execute_action(
        self, action: GroundedAction, image_width: int, image_height: int
    ) -> None:
        """Translate grounded action to driver calls."""
        act_type = action.action.upper()

        if act_type == "CLICK" and action.coordinates:
            x, y = action.coordinates.get("x", 0), action.coordinates.get("y", 0)
            await self.driver.click(x, y)

        elif act_type == "DOUBLE_CLICK" and action.coordinates:
            x, y = action.coordinates.get("x", 0), action.coordinates.get("y", 0)
            await self.driver.double_click(x, y)

        elif act_type == "RIGHT_CLICK" and action.coordinates:
            x, y = action.coordinates.get("x", 0), action.coordinates.get("y", 0)
            await self.driver.right_click(x, y)

        elif act_type == "TYPE" and action.text:
            await self.driver.type_text(action.text)

        elif act_type == "PRESS_KEY" and action.key:
            await self.driver.press_key(action.key)

        elif act_type == "SCROLL":
            direction = action.scroll_direction or "down"
            amount = action.scroll_amount or 300
            await self.driver.scroll(direction=direction, amount=amount)

        elif act_type == "WAIT":
            await asyncio.sleep(2.0)

        elif act_type == "ASK_USER":
            approved = self.hitl_handler.confirm_general_action(
                action_title="Action Approval",
                details=action.text or action.thought,
            )
            if not approved:
                logger.info("User declined action.")
