"""Tests for Magnum Memory and Learned Knowledge Store."""

from magnum.intelligence.memory import MagnumMemory


def test_memory_default_rules():
    mem = MagnumMemory()
    assert "antigravity" in mem.app_mappings
    assert mem.resolve_app_name("antigravity") == "Antigravity IDE"
    assert mem.resolve_app_name("chrome") == "Google Chrome"


def test_memory_learn_rule():
    mem = MagnumMemory()
    mem.learn_user_rule("Always use dark mode")
    assert "Always use dark mode" in mem.user_rules

    mem.learn_app_mapping("editor", "Antigravity IDE")
    assert mem.resolve_app_name("editor") == "Antigravity IDE"


def test_memory_prompt_context():
    mem = MagnumMemory()
    ctx = mem.get_prompt_context()
    assert "[LEARNED SYSTEM KNOWLEDGE & CORRECTIONS]" in ctx
    assert "Antigravity IDE" in ctx
