"""
Tests for Magnum Local On-The-Fly Command & Task Intent Detector.
Validates:
- Detection of actionable computer tasks without wake words (e.g. 'WhatsApp message ko reply')
- English and Hindi/Hinglish task classification
- Rejection of non-task casual background chatter
- Integration with VoiceEngine.extract_command
"""

import pytest
from magnum.voice.intent_classifier import CommandIntentClassifier, get_command_classifier
from magnum.voice import VoiceEngine


def test_user_exact_phrase_accepted():
    clf = get_command_classifier()
    is_cmd, conf, reason = clf.is_actionable_command("WhatsApp message ko reply")
    assert is_cmd is True
    assert conf >= 0.80
    assert "action" in reason or "intent" in reason


def test_various_actionable_commands():
    clf = get_command_classifier()
    
    commands = [
        "Safari kholo",
        "Chrome band karo",
        "reply to Vishesh",
        "send message to Vishesh",
        "open YouTube and play music",
        "search for AI news on Google",
        "check my downloads folder",
        "click the submit button",
        "can you look at my screen",
        "screen dekho kya chal raha hai",
        "stop the watcher",
    ]
    for cmd in commands:
        is_cmd, conf, reason = clf.is_actionable_command(cmd)
        assert is_cmd is True, f"Failed to detect command: '{cmd}' (conf={conf}, reason={reason})"


def test_casual_chatter_rejected():
    clf = get_command_classifier()

    chatter = [
        "I was talking to him yesterday about the party",
        "the weather in Mumbai is really hot today",
        "haha that was so funny",
        "yeah I think so too",
        "my dog is sleeping on the couch",
        "she told me she might come tomorrow",
    ]
    for non_cmd in chatter:
        is_cmd, conf, reason = clf.is_actionable_command(non_cmd)
        assert is_cmd is False, f"Falsely classified chatter as command: '{non_cmd}' (conf={conf}, reason={reason})"


def test_voice_engine_extract_command_without_wake_word():
    engine = VoiceEngine()

    # Utterance with NO wake word
    detected, cmd = engine.extract_command("WhatsApp message ko reply")
    assert detected is True
    assert cmd == "WhatsApp message ko reply"

    # Utterance WITH traditional wake word
    detected, cmd = engine.extract_command("Hey Magnum open Safari")
    assert detected is True
    assert "open safari" in cmd.lower()

    # Casual conversation without wake word should NOT trigger
    detected, cmd = engine.extract_command("the weather in Mumbai is really hot today")
    assert detected is False
    assert cmd is None
