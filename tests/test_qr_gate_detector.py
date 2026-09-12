import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image

from magnum.agent import MagnumAgent
from magnum.config import AppConfig
from magnum.intelligence.grounding import TextElement
from magnum.intelligence.planner import PlanStep, PlanChecklist
from magnum.intelligence.nim_client import NimClient, GroundedAction


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


def test_nim_client_obstacle_detection_from_text():
    nim = NimClient()
    # When model outputs text containing QR code instruction
    action = nim.parse_json_response("The screen displays a WhatsApp QR code. Please scan the QR code with your phone.")
    assert action["action"] == "OBSTACLE_DETECTED"
    assert "qr code" in action["text"].lower()

    # When model outputs text containing CAPTCHA
    action_captcha = nim.parse_json_response("Cloudflare CAPTCHA verification required.")
    assert action_captcha["action"] == "OBSTACLE_DETECTED"

    # When model outputs text containing 2FA
    action_2fa = nim.parse_json_response("Enter 2FA verification code sent to phone.")
    assert action_2fa["action"] == "OBSTACLE_DETECTED"


@pytest.mark.asyncio
async def test_wait_for_obstacle_resolution_success():
    agent = MagnumAgent(cfg=AppConfig(default_execution_mode="desktop"))
    agent.driver.screenshot = AsyncMock(return_value=Image.new("RGB", (100, 100)))

    # Initial obstacle screen (e.g. WhatsApp QR, Steam QR, Cloudflare Captcha)
    obstacle_ocr = [
        make_text_element("Scan this QR code with your phone", 100, 100, 400, 30),
        make_text_element("Open mobile app and link device", 100, 140, 350, 25),
        make_text_element("Point camera to screen", 100, 170, 300, 25),
    ]

    # Destination UI after user scans (e.g. Chat UI loaded, dashboard loaded)
    cleared_ocr = [
        make_text_element("Chats", 50, 100, 80, 30),
        make_text_element("Search or start new chat", 50, 140, 200, 30),
        make_text_element("Rishika", 50, 200, 100, 30),
        make_text_element("Status", 50, 250, 80, 30),
        make_text_element("Calls", 50, 300, 80, 30),
        make_text_element("Settings", 50, 350, 80, 30),
        make_text_element("Archived", 50, 400, 80, 30),
    ]

    # First poll returns obstacle still present, second poll returns cleared UI
    with patch("magnum.intelligence.grounding.ScreenGrounder.extract_screen_text_elements", side_effect=[
        obstacle_ocr,
        cleared_ocr,
    ]):
        resolved = await agent._wait_for_obstacle_resolution(
            baseline_ocr=obstacle_ocr,
            prompt_msg="Please scan QR code",
            max_wait_seconds=10,
            poll_interval=0.01,
        )
        assert resolved is True


@pytest.mark.asyncio
async def test_wait_for_obstacle_resolution_timeout():
    agent = MagnumAgent(cfg=AppConfig(default_execution_mode="desktop"))
    agent.driver.screenshot = AsyncMock(return_value=Image.new("RGB", (100, 100)))

    obstacle_ocr = [
        make_text_element("Scan this QR code with your phone", 100, 100, 400, 30),
    ]

    with patch("magnum.intelligence.grounding.ScreenGrounder.extract_screen_text_elements", return_value=obstacle_ocr):
        resolved = await agent._wait_for_obstacle_resolution(
            baseline_ocr=obstacle_ocr,
            prompt_msg="Please scan QR code",
            max_wait_seconds=0.05,
            poll_interval=0.01,
        )
        assert resolved is False


@pytest.mark.asyncio
async def test_agent_handles_model_obstacle_detected():
    agent = MagnumAgent(cfg=AppConfig(default_execution_mode="desktop"))
    agent.driver.screenshot = AsyncMock(return_value=Image.new("RGB", (100, 100)))
    agent.driver.click = AsyncMock()
    agent.voice_engine.speak = MagicMock()
    agent.overlay.set_status = MagicMock()

    # Step: "Open chat and focus message box"
    step = PlanStep(step_index=1, title="Open chat and focus message box", description="Open Rishika's conversation.")
    mock_plan = PlanChecklist(
        goal_summary="Send message to Rishika",
        steps=[step],
        active_index=1,
    )
    agent.nim_client.generate_plan = MagicMock(return_value=mock_plan)

    # Attempt 1: Model evaluates screen, detects QR obstacle, and outputs OBSTACLE_DETECTED
    obstacle_action = GroundedAction(
        thought="WhatsApp is not logged in and displays a QR code for mobile device pairing.",
        action="OBSTACLE_DETECTED",
        text="WhatsApp is not logged in. Please scan the QR code on your screen with your phone. I am waiting for you to scan it.",
    )

    # Attempt 2 (after obstacle cleared): Model sees Rishika's chat and clicks it
    click_action = GroundedAction(
        thought="The chat UI is now loaded. Clicking on Rishika's chat.",
        action="CLICK",
        target_id=5,
        text="Rishika",
    )

    # Attempt 3: Step done
    done_action = GroundedAction(
        thought="Message box focused.",
        action="STEP_DONE",
    )

    agent.nim_client.ground_step_action = MagicMock(side_effect=[
        obstacle_action,
        click_action,
        done_action,
    ])

    # Screen transitions:
    # 1. First perception -> obstacle screen
    # 2. _wait_for_obstacle_resolution poll -> destination screen (user scanned!)
    # 3. Post-obstacle re-perception -> destination screen
    # 4. Attempt 2 perception -> destination screen
    # 5. Attempt 3 perception -> destination screen
    obstacle_elements = [make_text_element("Scan this QR code", 100, 100, 300, 30)]
    chat_elements = [
        make_text_element("Chats", 50, 50, 80, 30),
        make_text_element("Search", 50, 90, 80, 30),
        make_text_element("Rishika", 50, 140, 100, 30),
        make_text_element("Status", 50, 180, 80, 30),
        make_text_element("Calls", 50, 220, 80, 30),
        make_text_element("Archived", 50, 260, 80, 30),
        make_text_element("Settings", 50, 300, 80, 30),
    ]

    call_count = 0
    def mock_extract(screenshot):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return obstacle_elements
        return chat_elements

    with patch("magnum.intelligence.grounding.ScreenGrounder.extract_screen_text_elements", side_effect=mock_extract):
        with patch.object(agent.overlay, "update_plan"), patch.object(agent.overlay, "set_status"), patch.object(agent.overlay, "flash"):
            success = await agent.execute("jakar WhatsApp mein Rishika text do")

            assert success is True
            # Verify the model's voice message was spoken to the user
            spoken_messages = [call.args[0] for call in agent.voice_engine.speak.call_args_list]
            assert any("scan the qr code on your screen with your phone" in m.lower() for m in spoken_messages)
            assert any("obstacle cleared" in m.lower() for m in spoken_messages)
