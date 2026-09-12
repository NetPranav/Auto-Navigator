"""
Tests for Magnum Hindi and Hinglish to English Translator.
Validates:
- Devanagari script detection and translation
- Romanized Hinglish detection and fast rule translation
- English passthrough (zero modification and low latency)
"""

import pytest
from magnum.voice.translator import HindiEnglishTranslator


def test_language_detection():
    translator = HindiEnglishTranslator(use_llm_fallback=False)

    # Devanagari Hindi
    assert translator.detect_language("सफारी खोलो") == "hindi"
    assert translator.detect_language("यूट्यूब पर गाना चलाओ") == "hindi"

    # Hinglish
    assert translator.detect_language("Safari kholo aur google pe search karo") == "hinglish"
    assert translator.detect_language("chrome band karo") == "hinglish"
    assert translator.detect_language("ek second ruko") == "hinglish"

    # Pure English
    assert translator.detect_language("open Safari and search for news") == "english"
    assert translator.detect_language("close the browser window") == "english"


def test_devanagari_translation():
    translator = HindiEnglishTranslator(use_llm_fallback=False)

    res = translator.translate("सफारी खोलो")
    assert res.was_translated is True
    assert res.detected_language == "hindi"
    assert "open Safari" in res.translated_text


def test_hinglish_translation():
    translator = HindiEnglishTranslator(use_llm_fallback=False)

    res1 = translator.translate("chrome kholo")
    assert res1.was_translated is True
    assert res1.detected_language == "hinglish"
    assert "open chrome" in res1.translated_text.lower()

    res2 = translator.translate("safari band karo")
    assert res2.was_translated is True
    assert "close safari" in res2.translated_text.lower()

    res3 = translator.translate("ek second ruko")
    assert res3.was_translated is True
    assert "wait" in res3.translated_text.lower()


def test_english_passthrough():
    translator = HindiEnglishTranslator(use_llm_fallback=False)

    text = "open Safari and search for Apple stock price"
    res = translator.translate(text)
    assert res.was_translated is False
    assert res.detected_language == "english"
    assert res.translated_text == text
