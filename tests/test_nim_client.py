import json
import pytest
from PIL import Image
from autonavigator.intelligence.nim_client import (
    NimClient,
    GroundedAction,
    LinkedInCommentDraft,
)
from autonavigator.intelligence.planner import PlanChecklist, PlanStep


def test_pil_to_base64():
    img = Image.new("RGB", (50, 50), color="blue")
    b64 = NimClient.pil_to_base64(img)
    assert isinstance(b64, str)
    assert len(b64) > 0


def test_parse_json_response_with_markdown_fences():
    client = NimClient(api_key="test_key")
    raw = '```json\n{"thought": "click search", "action": "CLICK", "coordinates": {"x": 100, "y": 200}}\n```'
    parsed = client.parse_json_response(raw)
    assert parsed["action"] == "CLICK"
    assert parsed["coordinates"]["x"] == 100


def test_generate_plan_simulation_mode():
    client = NimClient(api_key="")
    plan = client.generate_plan("post comment on LinkedIn")
    assert isinstance(plan, PlanChecklist)
    assert len(plan.steps) >= 2
    assert plan.current_step is not None


def test_ground_step_action_simulation_mode():
    client = NimClient(api_key="")
    img = Image.new("RGB", (100, 100), color="red")
    b64 = NimClient.pil_to_base64(img)
    step = PlanStep(step_index=1, title="Open App", description="Open Chrome")
    action = client.ground_step_action(b64, "Search LinkedIn", step, 3)
    assert isinstance(action, GroundedAction)
    assert action.action in ("STEP_DONE", "FINISH", "WAIT")


def test_synthesize_comment_fallback():
    client = NimClient(api_key="")
    draft = client.synthesize_comment("Great updates about AI Agents!", author="Yash Rai")
    assert isinstance(draft, LinkedInCommentDraft)
    assert len(draft.draft_comment) > 5
