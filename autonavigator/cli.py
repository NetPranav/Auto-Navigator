"""
Command Line Interface for Auto-Navigator.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from autonavigator.agent import AutoNavigatorAgent
from autonavigator.config import config

console = Console()

BANNER = """
[bold cyan]    ___         __          _   __            _             __            [/bold cyan]
[bold cyan]   /   | __  __/ /_____    / | / /___ __   __(_)___ _____ _/ /_____  _____[/bold cyan]
[bold cyan]  / /| |/ / / / __/ __ \\  /  |/ / __ `/ | / / / __ `/ __ `/ __/ __ \\/ ___/[/bold cyan]
[bold cyan] / ___ / /_/ / /_/ /_/ / / /|  / /_/ /| |/ / / /_/ / /_/ / /_/ /_/ / /    [/bold cyan]
[bold cyan]/_/  |_\\__,_/\\__/\\____/ /_/ |_/\\__,_/ |___/_/\\__, /\\__,_/\\__/\\____/_/     [/bold cyan]
[bold cyan]                                            /____/                        [/bold cyan]
[bold white]  Autonomous Computer-Use & Web Agent • Powered by NVIDIA NIM Vision [/bold white]
"""

HELP_HINTS = """[bold yellow]💡 Try commands like:[/bold yellow]
  • [white]open gmail and check unread emails[/white]
  • [white]post comment to Yash Rai on LinkedIn[/white]
  • [white]copy op_celestia from github from overxpowered[/white]
  • [white]open spotlight and launch spotify[/white]
[dim](Type 'exit' or 'quit' to close)[/dim]
"""


async def run_single_task(instruction: str, mode: str) -> None:
    """Run a single navigation instruction."""
    async with AutoNavigatorAgent(mode=mode) as agent:
        await agent.execute(instruction)


async def run_interactive_repl(mode: str) -> None:
    """Run interactive REPL loop allowing arbitrary user instructions."""
    console.print(Panel(BANNER, border_style="cyan"))
    console.print(Panel(HELP_HINTS, title="[bold green]Ready for Instructions[/bold green]", border_style="green"))
    console.print(f"[dim]Mode: [bold]{mode.upper()}[/bold] (Live Desktop with HUD Border & Shutter Flash)[/dim]\n")

    async with AutoNavigatorAgent(mode=mode) as agent:
        while True:
            try:
                task = Prompt.ask("\n[bold cyan]⚡ Auto-Navigator >[/bold cyan] What would you like me to do?")
                if not task.strip():
                    continue
                if task.lower() in ("exit", "quit", "q"):
                    console.print("[dim]Goodbye![/dim]")
                    break

                await agent.execute(task)
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Session ended.[/dim]")
                break
            except Exception as e:
                console.print(f"[bold red]Error during execution:[/bold red] {e}")


def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(
        prog="autonavigator",
        description="Auto-Navigator: AI Agent for Desktop & Web Navigation with NVIDIA NIM",
    )
    parser.add_argument(
        "instruction",
        nargs="?",
        help="Navigation task instruction (e.g. 'open gmail', 'post comment on LinkedIn')",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["desktop", "headless"],
        default=config.default_execution_mode,
        help="Execution mode (default: desktop live screen control)",
    )
    parser.add_argument(
        "--hitl",
        choices=["cli", "native"],
        default=config.hitl_interface,
        help="HITL approval interface: 'cli' or 'native' macOS dialog",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run interactive REPL session",
    )

    args = parser.parse_args()

    config.default_execution_mode = args.mode
    config.hitl_interface = args.hitl

    if args.interactive or not args.instruction:
        asyncio.run(run_interactive_repl(mode=args.mode))
    else:
        console.print(BANNER)
        asyncio.run(run_single_task(instruction=args.instruction, mode=args.mode))


if __name__ == "__main__":
    main()
