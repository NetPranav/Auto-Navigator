"""Tests for window-targeted capture in DesktopDriver."""

from unittest.mock import patch, MagicMock
from magnum.drivers.desktop_driver import DesktopDriver


def test_take_screenshot_fallback():
    driver = DesktopDriver()
    img = driver.take_screenshot()
    assert img is not None
    assert img.size[0] > 0 and img.size[1] > 0


def test_capture_window_not_found():
    driver = DesktopDriver()
    # Looking for a non-existent app returns None cleanly
    win = driver.capture_window("NonExistentAppXYZ12345")
    assert win is None
