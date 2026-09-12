import pytest
from unittest.mock import patch, MagicMock
from autonavigator.ui import OverlayController


def test_overlay_controller_lifecycle():
    controller = OverlayController()
    
    with patch("subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.stdin = MagicMock()
        mock_popen.return_value = mock_process
        
        # Start overlay
        controller.start("TESTING STATUS")
        assert controller._process is not None
        mock_process.stdin.write.assert_called_with("STATUS TESTING STATUS\n")
        
        # Set status
        controller.set_status("NEW STATUS")
        mock_process.stdin.write.assert_called_with("STATUS NEW STATUS\n")
        
        # Flash
        controller.flash()
        mock_process.stdin.write.assert_called_with("FLASH\n")
        
        # Stop
        controller.stop()
        mock_process.stdin.write.assert_called_with("QUIT\n")
        mock_process.terminate.assert_called_once()
