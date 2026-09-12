import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image

from magnum.agent import MagnumAgent
from magnum.config import AppConfig
from magnum.intelligence.grounding import TextElement
from magnum.intelligence.planner import PlanStep


def make_text_element(
    text: str, left: float = 100, top: float = 100, width: float = 100, height: float = 30, confidence: float = 1.0
) -> TextElement:
    right = left + width
    bottom = top + height
    return TextElement(
        text=text,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        center_x=(left + right) / 2.0,
        center_y=(top + bottom) / 2.0,
        confidence=confidence,
    )


def test_should_auto_skip_app_open():
    agent = MagnumAgent(cfg=AppConfig(default_execution_mode="desktop"))

    # 1. App Launch step should auto-skip if target app is active
    step_open = PlanStep(step_index=1, title="Open WhatsApp", description="Launch the WhatsApp desktop application.")
    assert agent._should_auto_skip_app_open(
        current_step=step_open,
        active_app="WhatsApp",
        frontmost_app="whatsapp",
        top_bar_texts=["whatsapp", "file", "edit"],
    ) is not None

    # 2. Search step should NEVER be auto-skipped, even if WhatsApp is mentioned in title/description
    step_search = PlanStep(step_index=2, title="Search for Rishika", description="Use WhatsApp's search bar to find the contact named Rishika.")
    assert agent._should_auto_skip_app_open(
        current_step=step_search,
        active_app="WhatsApp",
        frontmost_app="whatsapp",
        top_bar_texts=["whatsapp", "file", "edit"],
    ) is None

    # 3. Subaction in title should NEVER be auto-skipped
    step_search_title = PlanStep(step_index=2, title="Search for Rishika in WhatsApp", description="Search contacts.")
    assert agent._should_auto_skip_app_open(
        current_step=step_search_title,
        active_app="WhatsApp",
        frontmost_app="whatsapp",
        top_bar_texts=["whatsapp"],
    ) is None

    # 4. Open chat / focus message box should NEVER be auto-skipped
    step_open_chat = PlanStep(step_index=3, title="Open chat and focus message box", description="Open Rishika's conversation.")
    assert agent._should_auto_skip_app_open(
        current_step=step_open_chat,
        active_app="WhatsApp",
        frontmost_app="whatsapp",
        top_bar_texts=["whatsapp"],
    ) is None

    # 5. Launch Chrome when VS Code is frontmost should NOT auto-skip
    step_chrome = PlanStep(step_index=1, title="Launch Google Chrome", description="Open Chrome.")
    assert agent._should_auto_skip_app_open(
        current_step=step_chrome,
        active_app="Code",
        frontmost_app="code",
        top_bar_texts=["code", "file", "edit"],
    ) is None

    # 6. Launch Chrome when Chrome is frontmost SHOULD auto-skip
    assert agent._should_auto_skip_app_open(
        current_step=step_chrome,
        active_app="Google Chrome",
        frontmost_app="google chrome",
        top_bar_texts=["google chrome", "file", "edit"],
    ) is not None


@pytest.mark.asyncio
async def test_handle_qr_gate_welcome_screen_and_qr_scan():
    agent = MagnumAgent(cfg=AppConfig(default_execution_mode="desktop"))
    agent.driver.click = AsyncMock()
    agent.driver.screenshot = AsyncMock(return_value=Image.new("RGB", (100, 100)))
    agent.voice_engine.speak = MagicMock()
    agent.overlay.set_status = MagicMock()

    # Screen 1: WhatsApp Welcome screen with Continue button (including unicode LTR mark \u200e)
    welcome_ocr = [
        make_text_element("WhatsApp", 100, 100, 200, 50),
        make_text_element("Simple. Reliable. Private.", 100, 160, 300, 30),
        make_text_element("\u200eContinue", 700, 650, 120, 40),
    ]

    # Screen 2 (after continue clicked): QR Code screen
    qr_ocr = [
        make_text_element("To use WhatsApp on your computer:", 100, 100, 400, 30),
        make_text_element("1. Open WhatsApp on your phone", 100, 140, 350, 25),
        make_text_element("2. Tap Menu or Settings and select Linked Devices", 100, 170, 400, 25),
        make_text_element("3. Point your phone to this screen to capture the code", 100, 200, 450, 25),
    ]

    # Screen 3 (after user scans with phone): Logged-in WhatsApp screen
    logged_in_ocr = [
        make_text_element("WhatsApp", 50, 50, 100, 30),
        make_text_element("Chats", 50, 100, 80, 30),
        make_text_element("Search or start new chat", 50, 140, 200, 30),
        make_text_element("Rishika", 50, 200, 100, 30),
        make_text_element("Status", 50, 250, 80, 30),
        make_text_element("Calls", 50, 300, 80, 30),
    ]

    # Sequence of OCR responses during polling:
    # 1. post_welcome_continue -> qr_ocr
    # 2. qr_poll attempt 1 -> qr_ocr (still waiting)
    # 3. qr_poll attempt 2 -> logged_in_ocr (user scanned!)
    with patch("magnum.intelligence.grounding.ScreenGrounder.extract_screen_text_elements", side_effect=[
        qr_ocr,
        qr_ocr,
        logged_in_ocr,
    ]):
        result = await agent._handle_login_or_qr_gate(
            ocr_elements=welcome_ocr,
            a11y_tree=None,
            active_app="WhatsApp",
            screenshot=Image.new("RGB", (100, 100)),
            max_wait_seconds=10,
        )

        assert result is True
        # Verify Continue button was clicked
        agent.driver.click.assert_called_once()
        # Verify voice instructions were spoken to the user
        spoken_calls = [call.args[0] for call in agent.voice_engine.speak.call_args_list]
        assert any("scan the qr code" in s.lower() for s in spoken_calls)
        assert any("login confirmed" in s.lower() for s in spoken_calls)


@pytest.mark.asyncio
async def test_handle_qr_gate_already_logged_in():
    agent = MagnumAgent(cfg=AppConfig(default_execution_mode="desktop"))
    agent.driver.click = AsyncMock()
    agent.voice_engine.speak = MagicMock()

    # WhatsApp is already logged in and showing chat list
    logged_in_ocr = [
        make_text_element("WhatsApp", 50, 50, 100, 30),
        make_text_element("Chats", 50, 100, 80, 30),
        make_text_element("Search", 50, 140, 200, 30),
        make_text_element("Rishika", 50, 200, 100, 30),
    ]

    result = await agent._handle_login_or_qr_gate(
        ocr_elements=logged_in_ocr,
        a11y_tree=None,
        active_app="WhatsApp",
        screenshot=Image.new("RGB", (100, 100)),
        max_wait_seconds=10,
    )

    # Should return False immediately without doing anything
    assert result is False
    agent.driver.click.assert_not_called()
    agent.voice_engine.speak.assert_not_called()


@pytest.mark.asyncio
async def test_handle_qr_gate_unrelated_app():
    agent = MagnumAgent(cfg=AppConfig(default_execution_mode="desktop"))
    agent.voice_engine.speak = MagicMock()

    chrome_ocr = [
        make_text_element("Google", 100, 100, 100, 50),
        make_text_element("Search Google or type a URL", 100, 200, 300, 30),
    ]

    result = await agent._handle_login_or_qr_gate(
        ocr_elements=chrome_ocr,
        a11y_tree=None,
        active_app="Google Chrome",
        screenshot=Image.new("RGB", (100, 100)),
        max_wait_seconds=10,
    )

    assert result is False
    agent.voice_engine.speak.assert_not_called()


@pytest.mark.asyncio
async def test_handle_qr_gate_timeout():
    agent = MagnumAgent(cfg=AppConfig(default_execution_mode="desktop"))
    agent.driver.screenshot = AsyncMock(return_value=Image.new("RGB", (100, 100)))
    agent.voice_engine.speak = MagicMock()
    agent.overlay.set_status = MagicMock()

    qr_ocr = [
        make_text_element("Scan the QR code to use WhatsApp", 100, 100, 400, 30),
        make_text_element("Open WhatsApp on your phone", 100, 140, 350, 25),
    ]

    with patch("magnum.intelligence.grounding.ScreenGrounder.extract_screen_text_elements", return_value=qr_ocr):
        result = await agent._handle_login_or_qr_gate(
            ocr_elements=qr_ocr,
            a11y_tree=None,
            active_app="WhatsApp",
            screenshot=Image.new("RGB", (100, 100)),
            max_wait_seconds=0.1,  # Short timeout for test
        )

        assert result is False
        spoken_calls = [call.args[0] for call in agent.voice_engine.speak.call_args_list]
        assert any("scan the qr code" in s.lower() for s in spoken_calls)
        assert any("timed out" in s.lower() for s in spoken_calls)
