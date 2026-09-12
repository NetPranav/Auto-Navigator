"""Tests for Jarvis Visual Screen Q&A Engine."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image

from magnum.intelligence.screen_qa import ScreenQARunner, ScreenQAResult
from magnum.intelligence.grounding import TextElement


def test_image_to_base64_conversion():
    img = Image.new("RGB", (2000, 1000), color="white")
    b64 = ScreenQARunner._image_to_base64(img, max_dim=800)
    assert isinstance(b64, str)
    assert len(b64) > 100


@pytest.mark.asyncio
async def test_screen_qa_runner_analysis():
    dummy_img = Image.new("RGB", (800, 600), color="black")
    driver = MagicMock()
    driver.take_screenshot.return_value = dummy_img

    nim_client = MagicMock()
    nim_client.is_simulation = False
    nim_client.call_vision.return_value = (
        "**Overview**: VS Code is open with a Python file.\n"
        "**Key Findings**: TypeError on line 42.\n"
        "**Suggested Action**: Change integer to string."
    )

    mock_elements = [
        TextElement(text="TypeError: unsupported operand type(s)", left=10, top=10, right=100, bottom=20, center_x=55, center_y=15),
        TextElement(text="Antigravity IDE", left=10, top=30, right=80, bottom=50, center_x=45, center_y=40),
    ]

    with patch("magnum.intelligence.grounding.ScreenGrounder.extract_screen_text_elements", return_value=mock_elements):
        result = await ScreenQARunner.capture_and_analyze(
            driver=driver,
            nim_client=nim_client,
            user_query="What error is showing on my screen?",
        )

        assert isinstance(result, ScreenQAResult)
        assert len(result.detected_errors) >= 1
        assert "TypeError" in result.detected_errors[0]
        assert "Antigravity" in result.active_apps
        assert "TypeError" in result.analysis
        assert "error" in result.speak_text.lower()


@pytest.mark.asyncio
async def test_screen_qa_runner_fallback_on_error():
    dummy_img = Image.new("RGB", (400, 300), color="blue")
    driver = MagicMock()
    driver.take_screenshot.return_value = dummy_img

    nim_client = MagicMock()
    nim_client.is_simulation = False
    nim_client.call_vision.side_effect = RuntimeError("API unreachable")
    nim_client.call_reasoning.side_effect = RuntimeError("API unreachable")

    mock_elements = [
        TextElement(text="SyntaxError: invalid syntax", left=10, top=10, right=100, bottom=20, center_x=55, center_y=15),
    ]

    with patch("magnum.intelligence.grounding.ScreenGrounder.extract_screen_text_elements", return_value=mock_elements):
        result = await ScreenQARunner.capture_and_analyze(
            driver=driver,
            nim_client=nim_client,
            user_query="Look at my screen",
        )

        assert isinstance(result, ScreenQAResult)
        assert "SyntaxError" in result.detected_errors[0]
        assert "SyntaxError" in result.analysis


@pytest.mark.asyncio
async def test_agent_visual_qa_routing():
    from magnum.agent import MagnumAgent
    agent = MagnumAgent.__new__(MagnumAgent)
    agent.driver = MagicMock()
    agent.nim_client = MagicMock()
    agent.voice_engine = MagicMock()
    agent.overlay = MagicMock()

    mock_res = ScreenQAResult(
        headline="Terminal shows clean build",
        analysis="Build completed successfully with 0 errors.",
        speak_text="Everything looks clean on your display.",
    )

    with patch("magnum.intelligence.screen_qa.ScreenQARunner.capture_and_analyze", new_callable=AsyncMock) as mock_qa:
        mock_qa.return_value = mock_res
        processed = await agent.process_instruction("look at my screen and tell me if there are errors")
        assert processed is True
        assert mock_qa.called
        assert agent.voice_engine.speak.called
