import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image

from magnum.agent import MagnumAgent
from magnum.config import AppConfig
from magnum.intelligence.grounding import TextElement
from magnum.intelligence.planner import PlanStep, PlanChecklist
from magnum.intelligence.nim_client import GroundedAction
from magnum.intelligence.a11y_tree import A11yTree, A11yElement


def make_text_el(text: str, left: float = 100, top: float = 100, width: float = 100, height: float = 30) -> TextElement:
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
        confidence=1.0,
    )


@pytest.mark.asyncio
async def test_open_app_during_interaction_step_does_not_prematurely_complete():
    """Verify that if OPEN_APP is executed during a non-launch step (e.g. Select chat and reply),
    it brings the app to the foreground and DOES NOT mark the step completed."""
    agent = MagnumAgent(cfg=AppConfig(default_execution_mode="desktop"))
    agent.driver.navigate = AsyncMock()
    agent.driver.screenshot = AsyncMock(return_value=Image.new("RGB", (100, 100), color="white"))

    # Mock step 1: interaction step
    step1 = PlanStep(step_index=1, title="Select chat and send reply", description="Click chat and send message")
    plan = PlanChecklist(goal_summary="Reply on WhatsApp", steps=[step1], active_index=1)
    agent.nim_client.generate_plan = MagicMock(return_value=plan)

    # In attempt 1, model decides OPEN_APP WhatsApp because WhatsApp wasn't open
    # In attempt 2, model completes the step with STEP_DONE
    action_open = GroundedAction(thought="WhatsApp is not active, opening it", action="OPEN_APP", text="WhatsApp")
    action_done = GroundedAction(thought="Chat selected and reply sent", action="STEP_DONE")
    agent.nim_client.ground_step_action = MagicMock(side_effect=[action_open, action_done])

    with patch("magnum.intelligence.grounding.ScreenGrounder.extract_screen_text_elements", return_value=[make_text_el("Antigravity IDE")]), \
         patch("magnum.driver.macos_ax.MacOSAccessibilityDriver.get_frontmost_app", return_value={"name": "Antigravity IDE"}):
        with patch.object(agent.overlay, "update_plan"), patch.object(agent.overlay, "set_status"), patch.object(agent.overlay, "flash"):
            success = await agent.execute("Reply on WhatsApp")
        assert success is True
        # Verify that navigate was called to open WhatsApp
        agent.driver.navigate.assert_called_with("WhatsApp")
        # Verify that ground_step_action was called twice (not terminated after attempt 1)
        assert agent.nim_client.ground_step_action.call_count == 2


@pytest.mark.asyncio
async def test_intercept_click_on_editor_code_text():
    """Verify that clicking on editor code text matching the step title inside Antigravity IDE
    is intercepted and safely redirected to native app launch."""
    agent = MagnumAgent(cfg=AppConfig(default_execution_mode="desktop"))
    agent.driver.navigate = AsyncMock()
    agent.driver.click = AsyncMock()
    agent.driver.screenshot = AsyncMock(return_value=Image.new("RGB", (100, 100), color="white"))

    step1 = PlanStep(step_index=1, title="Open WhatsApp", description="Launch WhatsApp desktop")
    plan = PlanChecklist(goal_summary="Open WhatsApp", steps=[step1], active_index=1)
    agent.nim_client.generate_plan = MagicMock(return_value=plan)

    # Model attempts to click element 17 labeled "Open WhatsApp" inside Antigravity IDE
    action_click = GroundedAction(
        thought="Element 17 says Open WhatsApp, clicking it",
        action="CLICK",
        target_id=17,
        text="Open WhatsApp",
    )
    agent.nim_client.ground_step_action = MagicMock(return_value=action_click)

    a11y_el = A11yElement(
        id=17,
        role="AXStaticText",
        label="Open WhatsApp",
        bounds=(100, 100, 150, 120),
        center_x=125,
        center_y=110,
        source="macos_ax",
    )
    tree = A11yTree(elements=[a11y_el])

    with patch("magnum.intelligence.grounding.ScreenGrounder.extract_screen_text_elements", return_value=[make_text_el("Antigravity IDE"), make_text_el("Open WhatsApp")]), \
         patch("magnum.intelligence.a11y_tree.A11yEngine.build_tree", return_value=tree), \
         patch("magnum.driver.macos_ax.MacOSAccessibilityDriver.get_frontmost_app", return_value={"name": "Antigravity IDE"}):
        with patch.object(agent.overlay, "update_plan"), patch.object(agent.overlay, "set_status"), patch.object(agent.overlay, "flash"):
            success = await agent.execute("Open WhatsApp")
        assert success is True
        # Must have redirected to native launch: WhatsApp
        agent.driver.navigate.assert_called_with("Whatsapp")
        # Must NOT have physically clicked the IDE editor text
        agent.driver.click.assert_not_called()


@pytest.mark.asyncio
async def test_interstitial_continue_button_handling():
    """Verify that when an interstitial button like 'Continue' or 'Agree' is visible in WhatsApp,
    the agent automatically clicks it to proceed."""
    agent = MagnumAgent(cfg=AppConfig(default_execution_mode="desktop"))
    agent.driver.navigate = AsyncMock()
    agent.driver.click = AsyncMock()
    agent.driver.screenshot = AsyncMock(return_value=Image.new("RGB", (100, 100), color="white"))

    step = PlanStep(step_index=1, title="Select chat and reply", description="Chat with contact")
    plan = PlanChecklist(goal_summary="Chat", steps=[step], active_index=1)
    agent.nim_client.generate_plan = MagicMock(return_value=plan)

    # Attempt 1: Screen contains "Welcome to WhatsApp" and "Continue"
    # Attempt 2: After interstitial click, model finishes with STEP_DONE
    continue_el = make_text_el("Continue", left=200, top=300, width=80, height=30)
    welcome_el = make_text_el("Welcome to WhatsApp", left=150, top=100, width=200, height=30)

    action_done = GroundedAction(thought="Done", action="STEP_DONE")
    agent.nim_client.ground_step_action = MagicMock(return_value=action_done)

    extract_side_effects = [
        [welcome_el, continue_el],  # Attempt 1 (triggers interstitial click)
        [make_text_el("WhatsApp Chats")],  # Post-interstitial re-perceive
    ]

    with patch("magnum.intelligence.grounding.ScreenGrounder.extract_screen_text_elements", side_effect=extract_side_effects), \
         patch("magnum.driver.macos_ax.MacOSAccessibilityDriver.get_frontmost_app", return_value={"name": "WhatsApp"}):
        with patch.object(agent.overlay, "update_plan"), patch.object(agent.overlay, "set_status"), patch.object(agent.overlay, "flash"):
            success = await agent.execute("Chat on WhatsApp")
        assert success is True
        # Verify that Continue was clicked at its center coordinates (240, 315)
        agent.driver.click.assert_called_with(240.0, 315.0)
