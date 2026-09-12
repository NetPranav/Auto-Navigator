"""
Magnum CLI — Command Line Interface for Pranav's AI Desktop Assistant.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from magnum.agent import MagnumAgent
from magnum.config import config

console = Console()

BANNER = """
[bold cyan]    __  ___                                  [/bold cyan]
[bold cyan]   /  |/  /___ _____ _____  __  ______ ___  [/bold cyan]
[bold cyan]  / /|_/ / __ `/ __ `/ __ \\/ / / / __ `__ \\ [/bold cyan]
[bold cyan] / /  / / /_/ / /_/ / / / / /_/ / / / / / / [/bold cyan]
[bold cyan]/_/  /_/\\__,_/\\__, /_/ /_/\\__,_/_/ /_/ /_/  [/bold cyan]
[bold cyan]             /____/                          [/bold cyan]
[bold white]     Pranav's AI Desktop Assistant             [/bold white]
[bold white]     Powered by Apple Vision + NVIDIA NIM      [/bold white]
"""

HELP_HINTS = """[bold yellow]💡 Try commands like:[/bold yellow]
  • [white]open youtube in chrome[/white]
  • [white]write 'hello world' in the text box[/white]
  • [white]open the coa_unit_3 pdf from downloads[/white]
  • [white]click Submit whenever it appears until AI is done[/white]
  • [white]voice[/white] (switch to hands-free "Hey Magnum" voice mode)
[bold green]🤖 Autopilot Commands:[/bold green]
  • [white]automate antigravity[/white] (start full-auto Antigravity pilot)
  • [white]check antigravity[/white] (one-shot state scan)
  • [white]autopilot status[/white] / [white]stop autopilot[/white]
[bold cyan]👁️ Sentinel (Universal Monitor):[/bold cyan]
  • [white]watch email for messages from Master[/white]
  • [white]watch WhatsApp for important messages[/white]
  • [white]stop watching email[/white] / [white]sentinel status[/white]
[bold magenta]💡 Suggestions & Activity:[/bold magenta]
  • [white]suggest[/white] / [white]what should I do next[/white]
  • [white]start suggestions[/white] (auto-observe & suggest)
  • [white]what was I doing[/white] / [white]activity log[/white]
[dim](Type 'exit' to close, 'status' for tasks, 'watchers' for background jobs)[/dim]
"""


async def run_voice_standalone(mode: str) -> None:
    """Run dedicated hands-free voice activation session that exits cleanly on Ctrl+C."""
    console.print(Panel(BANNER, border_style="cyan"))
    console.print(Panel(
        "[bold green]🎙️ Hands-Free Voice Mode ACTIVE[/bold green]\n\n"
        "[bold yellow]Say: 'Hey', 'Magnum', or 'Jarvis'[/bold yellow]\n"
        "[white]• Say: 'Hey' -> wait for chime -> speak your task[/white]\n"
        "[white]• Or all-in-one: 'Hey, open YouTube in Chrome'[/white]\n\n"
        "[dim](Press Ctrl+C to exit)[/dim]",
        title="[bold cyan]⚡ MAGNUM VOICE ASSISTANT[/bold cyan]",
        border_style="green",
    ))
    async with MagnumAgent(mode=mode) as agent:
        agent.overlay.set_status("🎙️ VOICE ACTIVE: 'HEY'")
        agent.voice_engine.speak("Magnum voice active. Listening.")

        try:
            await agent.voice_engine.listen_loop(
                on_command=agent.process_instruction,
                on_wake=lambda: agent.overlay.set_status("🎤 LISTENING..."),
            )
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            agent.voice_engine.stop()
            agent.task_queue.cancel_all_watchers()
            console.print("\n[dim]Magnum voice mode closed.[/dim]")


async def run_interactive_repl(mode: str) -> None:
    """Run interactive REPL with concurrent task and voice support."""
    console.print(Panel(BANNER, border_style="cyan"))
    console.print(Panel(HELP_HINTS, title="[bold green]Ready for Instructions[/bold green]", border_style="green"))
    console.print(f"[dim]Mode: [bold]{mode.upper()}[/bold] | Assistant: [bold]Magnum[/bold][/dim]\n")

    async with MagnumAgent(mode=mode) as agent:

        while True:
            try:
                # Show active watchers in prompt
                watchers = agent.task_queue.get_active_watchers()
                watcher_badge = f" [dim](👁️ {len(watchers)} watchers)[/dim]" if watchers else ""

                prompt_str = f"\n[bold cyan]⚡ Magnum >{watcher_badge}[/bold cyan] What should I do?"
                task = await asyncio.to_thread(Prompt.ask, prompt_str)
                if not task.strip():
                    continue

                lower = task.lower().strip()

                # Meta commands
                if lower in ("exit", "quit", "q"):
                    cancelled = agent.task_queue.cancel_all_watchers()
                    if cancelled:
                        console.print(f"[dim]Cancelled {cancelled} background watcher(s)[/dim]")
                    console.print("[dim]Goodbye, Pranav![/dim]")
                    break

                elif lower in ("voice", "listen", "mic"):
                    await run_voice_standalone(agent.mode)
                    continue

                elif lower in ("calibrate", "calib"):
                    from magnum.voice.calibration import run_calibration
                    run_calibration()
                    continue

                elif lower == "status":
                    summary = agent.task_queue.get_status_summary()
                    console.print(Panel(summary, title="[bold cyan]📊 Magnum Status[/bold cyan]", border_style="cyan"))
                    continue

                elif lower == "watchers":
                    active = agent.task_queue.get_active_watchers()
                    if not active:
                        console.print("[dim]No active background watchers.[/dim]")
                    else:
                        table = Table(title="👁️ Active Watchers", border_style="yellow")
                        table.add_column("ID", style="bold")
                        table.add_column("Watching For", style="yellow")
                        table.add_column("Action", style="cyan")
                        table.add_column("Triggers", style="green")
                        table.add_column("Stop When", style="red")
                        for w in active:
                            c = w.watcher_condition
                            table.add_row(
                                w.id, c.watch_for, c.action,
                                str(w.watcher_trigger_count),
                                c.stop_when or "manual",
                            )
                        console.print(table)
                    continue

                elif lower.startswith("cancel "):
                    task_id = lower.replace("cancel ", "").strip()
                    if agent.task_queue.cancel_watcher(task_id):
                        console.print(f"[bold green]✓ Cancelled watcher [{task_id}][/bold green]")
                    else:
                        console.print(f"[bold red]No active watcher with ID '{task_id}'[/bold red]")
                    continue

                elif lower == "cancel all":
                    count = agent.task_queue.cancel_all_watchers()
                    console.print(f"[bold green]✓ Cancelled {count} watcher(s)[/bold green]")
                    continue

                # Process instruction (auto-detects watcher vs foreground)
                await agent.process_instruction(task)

            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Session ended.[/dim]")
                agent.task_queue.cancel_all_watchers()
                agent.task_queue.cancel_autopilot()
                if agent._autopilot and agent._autopilot.is_active:
                    await agent._autopilot.stop()
                break
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}")


async def run_single_task(instruction: str, mode: str) -> None:
    """Run a single task."""
    async with MagnumAgent(mode=mode) as agent:
        await agent.process_instruction(instruction)


def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(
        prog="magnum",
        description="Magnum — Pranav's AI Desktop Assistant",
    )
    parser.add_argument(
        "instruction",
        nargs="?",
        help="Task instruction (e.g. 'open youtube', 'click Submit when it appears')",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["desktop", "headless"],
        default=config.default_execution_mode,
        help="Execution mode",
    )
    parser.add_argument(
        "--voice", "-v",
        action="store_true",
        help="Launch directly in hands-free voice wake-word mode",
    )
    parser.add_argument(
        "--hitl",
        choices=["cli", "native"],
        default=config.hitl_interface,
        help="HITL interface",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as background daemon",
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Start the Magnum daemon",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop the Magnum daemon",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run voice calibration wizard to tune microphone for near/far voice",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show daemon status",
    )

    args = parser.parse_args()
    config.default_execution_mode = args.mode
    config.hitl_interface = args.hitl

    if args.calibrate:
        from magnum.voice.calibration import run_calibration
        run_calibration()
        return

    # Daemon management
    if args.daemon:
        from magnum.daemon import MagnumDaemon
        daemon = MagnumDaemon()
        asyncio.run(daemon.run())
        return

    if args.start:
        from magnum.daemon import MagnumDaemon
        daemon = MagnumDaemon()
        if daemon.is_running():
            console.print("[bold yellow]Magnum is already running.[/bold yellow]")
        else:
            daemon.install_launchd()
            console.print("[bold green]✓ Magnum daemon started[/bold green]")
        return

    if args.stop:
        from magnum.daemon import MagnumDaemon
        daemon = MagnumDaemon()
        if daemon.stop():
            console.print("[bold green]✓ Magnum daemon stopped[/bold green]")
        else:
            console.print("[dim]Magnum daemon is not running.[/dim]")
        return

    if args.status:
        from magnum.daemon import MagnumDaemon
        daemon = MagnumDaemon()
        if daemon.is_running():
            console.print(f"[bold green]✓ Magnum is running (PID {daemon.get_pid()})[/bold green]")
        else:
            console.print("[dim]Magnum daemon is not running.[/dim]")
        return

    # Interactive, voice, or single task
    try:
        if args.instruction:
            console.print(BANNER)
            asyncio.run(run_single_task(instruction=args.instruction, mode=args.mode))
        elif args.voice:
            asyncio.run(run_voice_standalone(mode=args.mode))
        else:
            asyncio.run(run_interactive_repl(mode=args.mode))
    except (KeyboardInterrupt, SystemExit):
        console.print("\n[dim]Session closed.[/dim]")


if __name__ == "__main__":
    main()
