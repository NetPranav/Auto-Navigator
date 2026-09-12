"""
Global Quick-Launcher Bar for Auto-Navigator / Magnum.
Listens for global hotkey Option + Space (<alt>+<space>).
Pops up an instant native macOS launcher prompt to execute tasks from any workspace or app.
"""

import asyncio
import logging
import subprocess
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class QuickLauncher:
    """Global hotkey listener for Option + Space that summons an instant command bar."""

    def __init__(self, on_command: Optional[Callable[[str], None]] = None) -> None:
        self.on_command = on_command
        self._listener = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._active = False

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Start the global hotkey listener for Option + Space."""
        if self._active:
            return

        try:
            self._loop = loop or asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()

        self._active = True

        def _on_hotkey_triggered() -> None:
            logger.info("⚡ Option+Space triggered: Opening Quick Launcher")
            try:
                subprocess.Popen(["afplay", "/System/Library/Sounds/Ping.aiff"], stderr=subprocess.DEVNULL)
            except Exception:
                pass

            script = '''
            tell application "System Events"
                activate
                try
                    set res to text returned of (display dialog "What should Magnum do?" default answer "" with title "⚡ Magnum Quick Launcher" buttons {"Cancel", "Execute"} default button "Execute")
                    return res
                on error
                    return ""
                end try
            end tell
            '''
            try:
                res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=False)
                cmd = res.stdout.strip()
                if cmd and self.on_command:
                    logger.info(f"Quick Launcher Command: '{cmd}'")
                    if self._loop and self._loop.is_running():
                        asyncio.run_coroutine_threadsafe(self.on_command(cmd), self._loop)
            except Exception as e:
                logger.debug(f"Quick Launcher prompt error: {e}")

        def _run_listener():
            try:
                from pynput import keyboard
                self._listener = keyboard.GlobalHotKeys({
                    "<alt>+<space>": _on_hotkey_triggered
                })
                self._listener.run()
            except Exception as e:
                logger.debug(f"Could not bind global hotkey <alt>+<space>: {e}")

        t = threading.Thread(target=_run_listener, daemon=True)
        t.start()
        logger.info("⚡ Quick Launcher active: Press Option + Space anytime!")

    def stop(self) -> None:
        """Stop the global hotkey listener."""
        if self._listener:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None
        self._active = False


_global_quick_launcher: Optional[QuickLauncher] = None


def get_quick_launcher(on_command: Optional[Callable] = None) -> QuickLauncher:
    global _global_quick_launcher
    if _global_quick_launcher is None:
        _global_quick_launcher = QuickLauncher(on_command=on_command)
    elif on_command:
        _global_quick_launcher.on_command = on_command
    return _global_quick_launcher
