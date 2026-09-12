"""Tests for audio chimes and voice engine sound integration."""

import os
from magnum.voice import (
    CHIME_WAKE,
    CHIME_DONE,
    CHIME_CANCEL,
    CHIME_TICK,
    CHIME_SUCCESS,
    VoiceEngine,
)


def test_system_chime_paths_exist():
    # macOS built-in system sound paths
    for chime in (CHIME_WAKE, CHIME_DONE, CHIME_CANCEL, CHIME_TICK, CHIME_SUCCESS):
        assert os.path.exists(chime), f"System sound not found at {chime}"


def test_voice_engine_barge_in():
    ve = VoiceEngine()
    ve.stop_speaking()
    assert ve._current_tts_proc is None
