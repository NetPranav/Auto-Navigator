"""
Magnum Always-On Daemon.
Manages the lifecycle of the Magnum AI assistant as a background service.
Handles task queue, voice engine integration, and auto-restart.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Optional

from magnum.config import config

logger = logging.getLogger(__name__)

LAUNCHD_PLIST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.pranav.magnum</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>-m</string>
        <string>magnum</string>
        <string>--daemon</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{working_dir}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_dir}/magnum.stdout.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/magnum.stderr.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>
    </dict>
</dict>
</plist>
"""


class MagnumDaemon:
    """Always-on background service manager for Magnum."""

    def __init__(self) -> None:
        self.pid_file = config.daemon_pid_file
        self.running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def plist_path(self) -> Path:
        return Path.home() / "Library" / "LaunchAgents" / "com.pranav.magnum.plist"

    @property
    def log_dir(self) -> Path:
        log_path = Path.home() / "Library" / "Logs" / "Magnum"
        log_path.mkdir(parents=True, exist_ok=True)
        return log_path

    def is_running(self) -> bool:
        """Check if daemon is currently running."""
        if self.pid_file.exists():
            try:
                pid = int(self.pid_file.read_text().strip())
                os.kill(pid, 0)  # Check if process exists
                return True
            except (ProcessLookupError, ValueError):
                self.pid_file.unlink(missing_ok=True)
        return False

    def get_pid(self) -> Optional[int]:
        """Get the PID of the running daemon."""
        if self.pid_file.exists():
            try:
                return int(self.pid_file.read_text().strip())
            except ValueError:
                pass
        return None

    def write_pid(self) -> None:
        """Write current PID to file."""
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(str(os.getpid()))

    def remove_pid(self) -> None:
        """Remove PID file."""
        self.pid_file.unlink(missing_ok=True)

    def install_launchd(self) -> bool:
        """Install launchd plist for auto-start at login."""
        plist_content = LAUNCHD_PLIST_TEMPLATE.format(
            python_path=sys.executable,
            working_dir=str(Path.cwd()),
            log_dir=str(self.log_dir),
        )
        try:
            self.plist_path.parent.mkdir(parents=True, exist_ok=True)
            self.plist_path.write_text(plist_content)
            os.system(f"launchctl load {self.plist_path}")
            logger.info(f"Installed launchd service at {self.plist_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to install launchd: {e}")
            return False

    def uninstall_launchd(self) -> None:
        """Remove launchd plist."""
        if self.plist_path.exists():
            os.system(f"launchctl unload {self.plist_path}")
            self.plist_path.unlink(missing_ok=True)
            logger.info("Uninstalled launchd service")

    def stop(self) -> bool:
        """Stop the running daemon."""
        pid = self.get_pid()
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
                self.remove_pid()
                logger.info(f"Stopped daemon (PID {pid})")
                return True
            except ProcessLookupError:
                self.remove_pid()
        return False

    async def run(self) -> None:
        """Main daemon loop — runs the task queue and voice listener."""
        self.running = True
        self.write_pid()

        # Handle shutdown signals
        def _shutdown(signum, frame):
            logger.info(f"Received signal {signum}, shutting down daemon...")
            self.running = False
            self.remove_pid()

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)

        logger.info(f"Magnum daemon started (PID {os.getpid()})")

        try:
            from magnum.agent import MagnumAgent
            from magnum.task_queue import TaskQueue

            task_queue = TaskQueue()

            async with MagnumAgent(mode="desktop") as agent:
                agent.task_queue = task_queue
                logger.info("Magnum daemon ready and listening")

                while self.running:
                    # The daemon stays alive, processing tasks from the queue
                    # Voice engine and external commands feed tasks into the queue
                    await asyncio.sleep(1.0)

        except Exception as e:
            logger.error(f"Daemon error: {e}")
        finally:
            self.remove_pid()
            logger.info("Magnum daemon stopped")
