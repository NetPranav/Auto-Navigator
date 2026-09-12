"""Tests for Magnum Voice Engine."""

from magnum.voice import VoiceEngine


def test_voice_engine_wake_word_extraction():
    engine = VoiceEngine(wake_words=["hey", "magnum", "hey magnum", "jarvis"])

    # Exact wake word 'hey'
    detected, cmd = engine.extract_command("hey")
    assert detected is True
    assert cmd is None

    # 'Hey' uppercase with punctuation
    detected, cmd = engine.extract_command("Hey!")
    assert detected is True
    assert cmd is None

    # 'Hey' + command
    detected, cmd = engine.extract_command("hey open chrome")
    assert detected is True
    assert cmd == "open chrome"

    # Exact wake word 'magnum'
    detected, cmd = engine.extract_command("magnum")
    assert detected is True
    assert cmd is None

    # Wake word + command
    detected, cmd = engine.extract_command("magnum open youtube")
    assert detected is True
    assert cmd == "open youtube"

    # Hey Magnum + punctuation
    detected, cmd = engine.extract_command("hey magnum, click the submit button")
    assert detected is True
    assert cmd == "click the submit button"

    # Non-wake word
    detected, cmd = engine.extract_command("what is the weather today")
    assert detected is False
    assert cmd is None


def test_voice_engine_clean_speech():
    engine = VoiceEngine()
    # Ensure speak doesn't crash on empty or simple text
    engine.speak("")
    assert engine.is_listening is False
