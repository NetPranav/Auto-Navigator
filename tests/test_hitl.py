import pytest
from unittest.mock import patch
from autonavigator.hitl.hitl_handler import (
    HitlHandler,
    HitlDecision,
    HitlActionType,
)


def test_hitl_confirm_comment_approved():
    handler = HitlHandler(interface="cli")
    with patch("rich.prompt.Prompt.ask", return_value="y"):
        decision = handler.confirm_comment(
            draft_comment="Great perspective on multi-modal AI!",
            author="Yash Rai",
            post_summary="Multi-modal AI agents",
        )
        assert decision.action == HitlActionType.APPROVE
        assert decision.final_text == "Great perspective on multi-modal AI!"


def test_hitl_confirm_comment_rejected():
    handler = HitlHandler(interface="cli")
    with patch("rich.prompt.Prompt.ask", return_value="n"):
        decision = handler.confirm_comment(
            draft_comment="Great perspective on multi-modal AI!",
            author="Yash Rai",
        )
        assert decision.action == HitlActionType.REJECT


def test_hitl_confirm_comment_edited():
    handler = HitlHandler(interface="cli")
    # First prompt asks choice ('e'), second asks for custom text
    with patch("rich.prompt.Prompt.ask", side_effect=["e", "Custom edited comment"]):
        decision = handler.confirm_comment(
            draft_comment="Original draft",
            author="Yash Rai",
        )
        assert decision.action == HitlActionType.APPROVE
        assert decision.final_text == "Custom edited comment"
