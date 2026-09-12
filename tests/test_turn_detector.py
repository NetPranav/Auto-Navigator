"""
Tests for Magnum Smart Turn Detector & Interruption Manager.
Validates:
- Dynamic silence timeout extensions during thinking / hesitation pauses
- Fake interruption rejection (coughs, mic thumps, short duration <350ms, backchannels)
- True interruption qualification for explicit stop keywords and substantive speech
"""

import pytest
from magnum.voice.turn_detector import TurnDetector, TurnDetectorConfig, TurnState


def test_turn_detector_hesitation_tokens():
    detector = TurnDetector()

    # English fillers
    assert detector.is_filler_or_hesitation("uh") is True
    assert detector.is_filler_or_hesitation("um") is True
    assert detector.is_filler_or_hesitation("let me see") is True
    assert detector.is_filler_or_hesitation("I want to open Safari and uh") is True

    # Hindi / Hinglish fillers
    assert detector.is_filler_or_hesitation("arre") is True
    assert detector.is_filler_or_hesitation("ek second") is True
    assert detector.is_filler_or_hesitation("ruko") is True
    assert detector.is_filler_or_hesitation("matlab") is True
    assert detector.is_filler_or_hesitation("socho") is True
    assert detector.is_filler_or_hesitation("chrome kholo aur ek second") is True


def test_turn_detector_trailing_connectors():
    detector = TurnDetector()

    # Clauses ending in conjunctions or prepositions (incomplete thoughts)
    assert detector.is_incomplete_clause("open Chrome and") is True
    assert detector.is_incomplete_clause("go to the website that") is True
    assert detector.is_incomplete_clause("Safari kholo aur") is True

    # Complete commands
    assert detector.is_incomplete_clause("open Chrome") is False
    assert detector.is_incomplete_clause("close the tab") is False


def test_adaptive_silence_timeout():
    detector = TurnDetector()

    # Complete command has standard short timeout
    std_timeout = detector.get_adaptive_silence_timeout("open Safari")
    assert std_timeout == detector.config.min_delay  # 0.8s

    # Hesitant / thinking phrase gets extended silence window
    hesitant_timeout = detector.get_adaptive_silence_timeout("I want to uhm...")
    assert hesitant_timeout == detector.config.hesitation_delay  # 3.2s
    assert hesitant_timeout > 3.0

    # Hindi thinking phrase gets extended silence window
    hindi_hesitant_timeout = detector.get_adaptive_silence_timeout("ek second ruko...")
    assert hindi_hesitant_timeout == detector.config.hesitation_delay


def test_fake_interruption_short_burst_rejected():
    detector = TurnDetector()

    # A short noise burst (cough / desk thump, e.g. 0.15 seconds) while agent is speaking
    qualified, reason = detector.qualify_barge_in(
        audio_duration_sec=0.15,
        transcript=None,
        is_agent_speaking=True,
    )
    assert qualified is False
    assert "noise_gated" in reason
    assert detector.state == TurnState.NOISE_GATED


def test_fake_interruption_backchannel_rejected():
    detector = TurnDetector()

    # User says "uh-huh" or "hmm" while agent speaks
    qualified, reason = detector.qualify_barge_in(
        audio_duration_sec=0.45,
        transcript="uh-huh",
        is_agent_speaking=True,
    )
    assert qualified is False
    assert "backchannel" in reason


def test_true_interruption_stop_keyword():
    detector = TurnDetector()

    # Explicit stop words always interrupt instantly
    for stop_word in ["stop", "ruko", "cancel", "wait"]:
        qualified, reason = detector.qualify_barge_in(
            audio_duration_sec=0.4,
            transcript=stop_word,
            is_agent_speaking=True,
        )
        assert qualified is True, f"Failed on {stop_word}"
        assert detector.state == TurnState.INTERRUPTED


def test_true_interruption_substantive_speech():
    detector = TurnDetector()

    # Substantive speech interrupts agent
    qualified, reason = detector.qualify_barge_in(
        audio_duration_sec=1.2,
        transcript="actually do something else instead",
        is_agent_speaking=True,
    )
    assert qualified is True
    assert detector.state == TurnState.INTERRUPTED
