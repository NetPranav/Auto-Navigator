"""
End-to-end test verifying that agent.process_instruction automatically translates
Hindi and Hinglish instructions before execution planning.
"""

import pytest
from unittest.mock import AsyncMock, patch
from magnum.agent import MagnumAgent


@pytest.mark.asyncio
async def test_agent_process_instruction_translates_hindi():
    agent = MagnumAgent(mode="desktop")
    
    # Mock task_queue.run_foreground and log_instruction so we verify the translation hook
    with patch.object(agent.task_queue, "run_foreground", new_callable=AsyncMock) as mock_run, \
         patch("magnum.logger.log_instruction") as mock_log:
        mock_run.return_value = True
        
        # Test Hinglish: "chrome kholo"
        res = await agent.process_instruction("chrome kholo")
        assert res is True
        
        # Check that log_instruction received English translation
        mock_log.assert_called()
        args, kwargs = mock_log.call_args
        assert "open chrome" in args[0].lower()
        assert kwargs.get("raw_input") == "chrome kholo"
        assert kwargs.get("language") == "hinglish"
