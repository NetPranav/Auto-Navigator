"""Tests for native macOS notifications module."""

from unittest.mock import patch
from magnum.notifications import send_notification, notify_and_speak


def test_send_notification_runs_thread():
    with patch("subprocess.run") as mock_run:
        send_notification("Test Title", "Test Message", sound="Glass")
        # Allow thread to execute
        import time
        time.sleep(0.1)
        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert "display notification" in cmd
        assert "Test Title" in cmd
        assert "Test Message" in cmd


def test_notify_and_speak():
    class DummyVoice:
        def __init__(self):
            self.spoken = []

        def speak(self, text, blocking=False):
            self.spoken.append(text)

    voice = DummyVoice()
    with patch("subprocess.run"):
        notify_and_speak("Alert", "Task completed", voice_engine=voice)
        assert "Task completed" in voice.spoken
