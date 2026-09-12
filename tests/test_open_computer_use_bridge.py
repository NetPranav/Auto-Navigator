"""
Unit tests for Open Computer Use Bridge (deep integration with open-codex-computer-use).
"""

from unittest.mock import patch, MagicMock
import pytest
from magnum.driver.open_computer_use_bridge import OpenComputerUseBridge


def test_open_computer_use_bridge_availability():
    # Local node_modules/.bin/ocu is installed
    assert OpenComputerUseBridge.is_available() is True
    bin_path = OpenComputerUseBridge.get_binary_path()
    assert bin_path is not None
    assert "ocu" in bin_path or "open-computer-use" in bin_path


def test_open_computer_use_list_apps_live():
    # Test real list_apps call
    apps = OpenComputerUseBridge.list_apps()
    assert isinstance(apps, list)
    assert len(apps) > 0
    # Antigravity or Safari should be in running apps list
    app_names = [a["name"] for a in apps]
    assert any("antigravity" in name.lower() or "safari" in name.lower() or "finder" in name.lower() for name in app_names)


def test_open_computer_use_call_tool_mock():
    mock_res = MagicMock()
    mock_res.stdout = '{"content":[{"type":"text","text":"OK"}],"isError":false}'
    mock_res.stderr = ""
    mock_res.returncode = 0

    with patch("subprocess.run", return_value=mock_res) as mock_sub:
        res = OpenComputerUseBridge.call_tool("click", {"app": "Finder", "element_index": "0"})
        assert res.get("isError") is False
        assert mock_sub.called
        cmd = mock_sub.call_args[0][0]
        assert "call" in cmd
        assert "click" in cmd
