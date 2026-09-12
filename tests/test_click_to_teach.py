"""Tests for Click-to-Teach Visual Demonstration Mode and Visual Anchors."""

from magnum.intelligence.memory import MagnumMemory, VisualAnchor


def test_visual_anchor_save_and_retrieve():
    mem = MagnumMemory()
    mem.learn_visual_anchor("Submit", 1200.0, 750.0, app_name="Antigravity IDE")

    anchor = mem.get_visual_anchor("submit")
    assert anchor is not None
    assert anchor.label == "Submit"
    assert anchor.x == 1200.0
    assert anchor.y == 750.0
    assert anchor.app_name == "Antigravity IDE"


def test_visual_anchor_case_insensitivity():
    mem = MagnumMemory()
    mem.learn_visual_anchor("Proceed With Plan", 640.0, 480.0)

    assert mem.get_visual_anchor("proceed with plan") is not None
    assert mem.get_visual_anchor("PROCEED WITH PLAN") is not None
    assert mem.get_visual_anchor("nonexistent") is None
