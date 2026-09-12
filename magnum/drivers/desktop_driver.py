"""
Desktop OS Driver for macOS system-wide screen capture and control.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Optional
from PIL import Image, ImageGrab
import pyautogui
import pyperclip

from magnum.config import config
from magnum.drivers.base import BaseDriver
from magnum.intelligence.grounding import ScreenGrounder
from magnum.ui import get_overlay

logger = logging.getLogger(__name__)

# Configure PyAutoGUI safety settings
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.15


class DesktopDriver(BaseDriver):
    """OS-level desktop driver using macOS screen capture and PyAutoGUI."""

    def __init__(self) -> None:
        self.retina_scale: float = 1.0
        self.screen_width: int = 1280
        self.screen_height: int = 800
        self._step_counter = 0
        self.overlay = get_overlay()
        self._caffeinate_proc: Optional[subprocess.Popen] = None

    async def start(self) -> None:
        """Initialize desktop screen metrics, Retina scale, and visual HUD border."""
        config.ensure_directories()
        self.retina_scale = ScreenGrounder.get_mac_retina_scale()
        size = pyautogui.size()
        self.screen_width = size.width
        self.screen_height = size.height
        
        # Prevent macOS screen and system from sleeping while Auto-Navigator is running
        try:
            import os
            self._caffeinate_proc = subprocess.Popen(
                ["caffeinate", "-dims", "-w", str(os.getpid())],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.debug(f"Could not start caffeinate: {e}")

        # Start the glowing screen border overlay
        self.overlay.start("AUTO-NAVIGATOR ACTIVE")
        logger.info(
            f"DesktopDriver initialized. Logical Screen: {self.screen_width}x{self.screen_height}, "
            f"Retina Scale: {self.retina_scale}x"
        )

    async def close(self) -> None:
        """Clean up desktop driver and stop screen overlay."""
        if self._caffeinate_proc and self._caffeinate_proc.poll() is None:
            try:
                self._caffeinate_proc.terminate()
            except Exception:
                pass
        self.overlay.stop()
        logger.info("DesktopDriver closed.")

    async def navigate(self, target: str) -> None:
        """Launch or switch to target application and bring to frontmost focus."""
        logger.info(f"Switching to application or target: {target}")
        self.overlay.set_status(f"OPENING: {target.upper()}")
        
        if target.startswith("http://") or target.startswith("https://"):
            # Set Chrome active tab URL directly via AppleScript for instant, clean navigation
            script = f'''
            tell application "Google Chrome"
                activate
                if (count of windows) is 0 then
                    make new window
                end if
                set URL of active tab of front window to "{target}"
            end tell
            '''
            try:
                res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=False)
                if res.returncode != 0:
                    subprocess.run(["open", "-a", "Google Chrome", target], check=False)
            except Exception:
                subprocess.run(["open", target], check=False)
        else:
            from magnum.intelligence.memory import get_memory
            real_name = get_memory().resolve_app_name(target)
            target_lower = target.lower()

            # Special handling for Antigravity IDE (runs as Electron process on macOS)
            if "antigravity" in target_lower or "electron" in target_lower:
                script = '''
                tell application "System Events"
                    repeat with p in (every application process whose visible is true)
                        if name of p is "Electron" or name of p contains "Antigravity" then
                            set frontmost of p to true
                            return "OK"
                        end if
                    end repeat
                end tell
                '''
                res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=False)
                if "OK" not in res.stdout:
                    subprocess.run(["open", "-a", real_name], capture_output=True, check=False)
            else:
                # Open App and bring to front
                res = subprocess.run(["open", "-a", real_name], capture_output=True, check=False)
                if res.returncode != 0 and real_name != target:
                    subprocess.run(["open", "-a", target], capture_output=True, check=False)
                subprocess.run(["osascript", "-e", f'tell application "{real_name}" to activate'], capture_output=True, check=False)
        
        await asyncio.sleep(1.5)

    async def switch_workspace(self, space: int) -> None:
        """Switch macOS Space using Ctrl + number."""
        logger.info(f"Switching to macOS Space {space}")
        await self.hotkey("control", str(space))

    async def trigger_mission_control(self) -> None:
        """Open Mission Control."""
        logger.info("Triggering macOS Mission Control")
        await self.hotkey("control", "up")

    async def screenshot(self, filename: Optional[str] = None) -> Image.Image:
        """Capture screenshot of the macOS desktop with shutter flash animation."""
        self._step_counter += 1
        
        # Trigger visual screen flash & shutter audio animation
        self.overlay.set_status("CAPTURING SCREENSHOT")
        self.overlay.flash()
        await asyncio.sleep(0.1)

        # ImageGrab on macOS captures at physical display resolution
        image = ImageGrab.grab()

        if config.save_debug_screenshots:
            fname = filename or f"desktop_step_{self._step_counter:03d}.png"
            out_path = config.screenshots_dir / fname
            image.save(out_path)
            logger.debug(f"Saved desktop screenshot to {out_path}")

        return image

    def capture_window(self, app_name_or_title: str) -> Optional[Image.Image]:
        """Capture a specific application window directly using Quartz, even if on another Space or in background."""
        try:
            import Quartz
            from AppKit import NSBitmapImageRep, NSPNGFileType
            import io

            window_list = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionAll,
                Quartz.kCGNullWindowID,
            )
            target_lower = app_name_or_title.lower()
            target_win = None

            for w in window_list:
                owner = str(w.get("kCGWindowOwnerName", "")).lower()
                name = str(w.get("kCGWindowName", "")).lower()
                bounds = w.get("kCGWindowBounds", {})
                if bounds.get("Width", 0) < 300 or bounds.get("Height", 0) < 200:
                    continue
                if target_lower in owner or target_lower in name:
                    target_win = w
                    break

            if not target_win:
                return None

            wid = target_win["kCGWindowNumber"]
            cg_img = Quartz.CGWindowListCreateImage(
                Quartz.CGRectNull,
                Quartz.kCGWindowListOptionIncludingWindow,
                wid,
                Quartz.kCGWindowImageBoundsIgnoreFraming,
            )
            if not cg_img:
                return None

            rep = NSBitmapImageRep.alloc().initWithCGImage_(cg_img)
            png_data = rep.representationUsingType_properties_(NSPNGFileType, None)
            return Image.open(io.BytesIO(png_data.bytes()))
        except Exception as e:
            logger.debug(f"Window capture failed: {e}")
            return None

    def take_screenshot(self, target_app: Optional[str] = None, filename: Optional[str] = "watcher_poll.png") -> Image.Image:
        """Synchronous screenshot grab with optional background window targeting and disk saving."""
        image = ImageGrab.grab()
        if filename:
            try:
                config.ensure_directories()
                out_path = config.screenshots_dir / filename
                image.save(out_path)
            except Exception as e:
                logger.debug(f"Failed to save screenshot {filename}: {e}")
        return image

    def _to_logical_coords(self, x: float, y: float, image_width: Optional[float] = None) -> tuple[float, float]:
        """Convert pixel coordinate to logical display point coordinate safely."""
        # Check normalized coordinates (0.0 to 1.0)
        if 0.0 < x <= 1.0 and 0.0 < y <= 1.0:
            return (x * self.screen_width, y * self.screen_height)

        # Check if coordinates were omitted or (0,0) corner
        if x <= 10.0 and y <= 10.0:
            # Default to center of screen to prevent corner fail-safe triggers
            return (self.screen_width / 2.0, self.screen_height / 2.0)

        # Scale physical pixels to logical Retina points
        if image_width and image_width > self.screen_width:
            scale = image_width / self.screen_width
            return (x / scale, y / scale)
        if self.retina_scale > 1.0:
            # If coordinates are larger than logical screen size, scale down
            if x > self.screen_width or y > self.screen_height:
                return (x / self.retina_scale, y / self.retina_scale)
        
        return (x, y)

    async def click(self, x: float, y: float) -> None:
        """Click at logical coordinates safely."""
        lx, ly = self._to_logical_coords(x, y)
        logger.info(f"Desktop click at ({lx:.1f}, {ly:.1f}) [raw: ({x}, {y})]")
        try:
            pyautogui.click(lx, ly)
        except Exception as e:
            logger.warning(f"PyAutoGUI click notice: {e}")
        await asyncio.sleep(0.3)

    async def double_click(self, x: float, y: float) -> None:
        """Double click at logical coordinates."""
        lx, ly = self._to_logical_coords(x, y)
        logger.info(f"Desktop double-click at ({lx}, {ly})")
        pyautogui.doubleClick(lx, ly)
        await asyncio.sleep(0.3)

    async def right_click(self, x: float, y: float) -> None:
        """Right click at logical coordinates."""
        lx, ly = self._to_logical_coords(x, y)
        logger.info(f"Desktop right-click at ({lx}, {ly})")
        pyautogui.rightClick(lx, ly)
        await asyncio.sleep(0.3)

    async def type_text(self, text: str, delay_ms: int = 40) -> None:
        """Type text using PyAutoGUI or clipboard paste for fast reliable entry."""
        logger.info(f"Desktop typing text: '{text}'")
        # For longer text or special characters, clipboard paste is much faster and handles emojis/multiline
        if len(text) > 15 or "\n" in text:
            old_clip = pyperclip.paste()
            pyperclip.copy(text)
            pyautogui.hotkey("command", "v")
            await asyncio.sleep(0.2)
            # Restore clip
            pyperclip.copy(old_clip)
        else:
            interval = delay_ms / 1000.0
            pyautogui.write(text, interval=interval)
        await asyncio.sleep(0.3)

    async def press_key(self, key: str) -> None:
        """Press keyboard key (e.g. 'enter', 'tab', 'esc')."""
        key_name = key.lower()
        if key_name == "enter" or key_name == "return":
            key_name = "return"
        logger.info(f"Desktop pressing key: {key_name}")
        pyautogui.press(key_name)
        await asyncio.sleep(0.3)

    async def hotkey(self, *keys: str) -> None:
        """Execute key combination."""
        mapped = [k.replace("cmd", "command").replace("ctrl", "control") for k in keys]
        logger.info(f"Desktop hotkey: {mapped}")
        pyautogui.hotkey(*mapped)
        await asyncio.sleep(0.4)

    async def scroll(self, direction: str = "down", amount: int = 300) -> None:
        """Scroll macOS screen."""
        # On macOS pyautogui, negative scrolls down, positive scrolls up
        clicks = -(amount // 100) if direction == "down" else (amount // 100)
        logger.info(f"Desktop scrolling {direction} clicks={clicks}")
        pyautogui.scroll(clicks)
        await asyncio.sleep(0.5)

    async def get_clipboard(self) -> str:
        """Read system clipboard."""
        return pyperclip.paste()

    async def set_clipboard(self, text: str) -> None:
        """Write to system clipboard."""
        pyperclip.copy(text)
        logger.info("Copied text to macOS system clipboard.")

    async def extract_text_content(self) -> str:
        """Desktop text extraction fallback."""
        return ""
