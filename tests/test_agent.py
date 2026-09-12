import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from PIL import Image
from autonavigator.agent import AutoNavigatorAgent
from autonavigator.config import AppConfig
from autonavigator.intelligence.planner import PlanChecklist, PlanStep
from autonavigator.intelligence.nim_client import GroundedAction


@pytest.mark.asyncio
async def test_agent_dynamic_plan_and_execution():
    cfg = AppConfig(default_execution_mode="desktop")
    agent = AutoNavigatorAgent(cfg=cfg)

    # Mock driver screenshot
    dummy_img = Image.new("RGB", (100, 100), color="blue")
    agent.driver.screenshot = AsyncMock(return_value=dummy_img)
    agent.driver.click = AsyncMock()
    agent.driver.type_text = AsyncMock()

    # Mock planning to return 2 simple steps
    mock_plan = PlanChecklist(
        goal_summary="Test dynamic goal",
        steps=[
            PlanStep(step_index=1, title="Step 1", description="Do first thing", status="active"),
            PlanStep(step_index=2, title="Step 2", description="Do second thing"),
        ],
        active_index=1,
    )
    agent.nim_client.generate_plan = MagicMock(return_value=mock_plan)

    # Mock step grounding action: STEP_DONE for both steps
    agent.nim_client.ground_step_action = MagicMock(
        return_value=GroundedAction(
            thought="Step completed",
            action="STEP_DONE",
            is_terminal=False,
        )
    )

    with patch.object(agent.overlay, "update_plan"), patch.object(agent.overlay, "set_status"), patch.object(agent.overlay, "flash"):
        result = await agent.execute("post comment to Yash Rai on LinkedIn")
        assert result is True
        assert agent.driver.screenshot.call_count >= 2
        assert mock_plan.is_finished is True
