"""
Native macOS notification dispatcher for Auto-Navigator / Magnum.
Displays native macOS banner notifications and plays system alerts.
"""

import subprocess
import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def send_notification(
    title: str,
    message: str,
    subtitle: Optional[str] = None,
    sound: Optional[str] = "Glass",
) -> None:
    """Display a native macOS banner notification asynchronously."""
    def _run() -> None:
        try:
            # Escape double quotes and backslashes
            clean_title = title.replace("\\", "\\\\").replace('"', '\\"')
            clean_msg = message.replace("\\", "\\\\").replace('"', '\\"')
            
            script_parts = [f'display notification "{clean_msg}" with title "{clean_title}"']
            if subtitle:
                clean_sub = subtitle.replace("\\", "\\\\").replace('"', '\\"')
                script_parts.append(f'subtitle "{clean_sub}"')
            if sound:
                clean_sound = sound.replace("\\", "\\\\").replace('"', '\\"')
                script_parts.append(f'sound name "{clean_sound}"')
            
            cmd = f"osascript -e '{' '.join(script_parts)}'"
            subprocess.run(cmd, shell=True, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            logger.debug(f"Failed to post macOS notification: {e}")

    # Run in background daemon thread to never block async event loop
    t = threading.Thread(target=_run, daemon=True)
    t.start()


def notify_and_speak(
    title: str,
    message: str,
    voice_engine=None,
    speech_text: Optional[str] = None,
    sound: Optional[str] = "Glass",
) -> None:
    """Post native banner notification and speak announcement."""
    send_notification(title=title, message=message, sound=sound)
    if voice_engine:
        text_to_speak = speech_text or message
        try:
            voice_engine.speak(text_to_speak, blocking=False)
        except Exception as e:
            logger.debug(f"Failed to speak notification: {e}")
