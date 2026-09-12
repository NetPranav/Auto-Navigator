"""
Human-in-the-Loop (HITL) Handler for Auto-Navigator.
Provides 1-click approvals via on-screen bottom-center cards, native macOS popups, and rich CLI.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import subprocess
from typing import List, Optional
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from magnum.config import config
from magnum.ui import get_overlay

logger = logging.getLogger(__name__)
console = Console()


class HitlActionType(str, enum.Enum):
    APPROVE = "APPROVE"
    EDIT = "EDIT"
    REGENERATE = "REGENERATE"
    REJECT = "REJECT"


class HitlDecision(BaseModel):
    """Result of user interaction."""

    action: HitlActionType
    final_text: Optional[str] = None
    user_notes: Optional[str] = None


class HitlHandler:
    """Manages approvals, interactive prompts, and desktop notifications."""

    def __init__(self, interface: Optional[str] = None) -> None:
        self.interface = interface or config.hitl_interface
        self.overlay = get_overlay()

    def notify(self, title: str, message: str, sound: bool = True) -> None:
        """Send a native macOS desktop notification."""
        sound_cmd = 'sound name "Glass"' if sound else ""
        escaped_title = title.replace('"', '\\"')
        escaped_msg = message.replace('"', '\\"')
        script = f'display notification "{escaped_msg}" with title "Auto-Navigator" subtitle "{escaped_title}" {sound_cmd}'
        try:
            subprocess.run(["osascript", "-e", script], check=False)
        except Exception as e:
            logger.debug(f"Notification error: {e}")
        console.print(f"[bold cyan]🔔 [Auto-Navigator][/bold cyan] [bold]{title}:[/bold] {message}")

    async def confirm_comment_async(
        self,
        draft_comment: str,
        author: Optional[str] = None,
        post_summary: Optional[str] = None,
    ) -> HitlDecision:
        """Show the on-screen bottom-center card for 1-click approval."""
        prompt_text = f"Proposed comment for {author or 'User'}:\n\n\"{draft_comment}\""
        
        # Display rich CLI panel simultaneously
        header = f"[bold green]Proposed Comment for:[/bold green] [bold yellow]{author or 'User'}[/bold yellow]"
        body = f"[dim]Topic:[/dim] {post_summary or 'Update'}\n\n[bold white]\"{draft_comment}\"[/bold white]"
        console.print()
        console.print(Panel(body, title=header, border_style="green", expand=False))
        console.print("[dim]Click the on-screen card button or respond in terminal...[/dim]")

        # Trigger on-screen interactive question card
        choice = await self.overlay.ask_question_card(
            prompt=prompt_text,
            options=["Approve & Post", "Cancel"],
        )

        if "approve" in choice.lower() or "post" in choice.lower():
            return HitlDecision(action=HitlActionType.APPROVE, final_text=draft_comment)
        else:
            return HitlDecision(action=HitlActionType.REJECT)

    def confirm_comment(
        self,
        draft_comment: str,
        author: Optional[str] = None,
        post_summary: Optional[str] = None,
    ) -> HitlDecision:
        """Sync fallback CLI prompt."""
        header = f"[bold green]Proposed Comment for:[/bold green] [bold yellow]{author or 'User'}[/bold yellow]"
        body = f"[dim]Topic:[/dim] {post_summary or 'Update'}\n\n[bold white]\"{draft_comment}\"[/bold white]"
        
        console.print()
        console.print(Panel(body, title=header, border_style="green", expand=False))
        console.print("[cyan][y] Approve & Post (1-click)[/cyan] | [yellow][e] Edit text[/yellow] | [red][n] Cancel[/red]")
        
        choice = Prompt.ask("Your choice", choices=["y", "e", "n"], default="y")

        if choice == "y":
            return HitlDecision(action=HitlActionType.APPROVE, final_text=draft_comment)
        elif choice == "e":
            edited = Prompt.ask("Enter custom comment", default=draft_comment)
            return HitlDecision(action=HitlActionType.APPROVE, final_text=edited)
        else:
            return HitlDecision(action=HitlActionType.REJECT)

    async def ask_user_choice_async(
        self, prompt_text: str, options: Optional[List[str]] = None
    ) -> str:
        """Ask user a choice via the on-screen bottom card."""
        console.print(f"[bold yellow]❓ Question:[/bold yellow] {prompt_text}")
        return await self.overlay.ask_question_card(prompt=prompt_text, options=options)

    async def ask_user_text_async(self, question: str, default: str = "") -> str:
        """Ask user for text input via on-screen card and interactive terminal."""
        console.print()
        console.print(Panel(f"[bold white]{question}[/bold white]", title="[bold yellow]❓ AUTO-NAVIGATOR QUESTION[/bold yellow]", border_style="yellow"))
        
        # Show on-screen card with terminal instruction
        try:
            await self.overlay.ask_question_card(
                prompt=question,
                options=["Answer in Terminal", "Cancel"],
                timeout=5.0,  # Short timeout since user answers in terminal
            )
        except Exception:
            pass  # Card display is best-effort

        try:
            loop = asyncio.get_running_loop()
            user_input = await loop.run_in_executor(
                None, lambda: Prompt.ask("[bold cyan]👉 Your Answer[/bold cyan]", default=default)
            )
            self.overlay.hide_question_card()
            return user_input.strip()
        except Exception:
            self.overlay.hide_question_card()
            return default

