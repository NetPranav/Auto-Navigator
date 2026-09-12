"""
Magnum Hindi & Hinglish to English Speech & Instruction Translator.

Handles:
- Native Devanagari script (Hindi: \u0900-\u097F)
- Romanized Hindi / Code-mixed Hinglish (e.g., "Safari kholo aur google pe AI news search karo")
- Fast rule-based translation for common computer and browser commands (<1ms)
- LLM semantic translation fallback (NVIDIA NIM / OpenAI) for complex multi-clause instructions
"""

from __future__ import annotations

import logging
import re
import string
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    """Outcome of language detection and translation."""
    original_text: str
    translated_text: str
    detected_language: str  # "english", "hindi", "hinglish"
    was_translated: bool
    confidence: float = 1.0


# Common Hinglish verbs, nouns, and conjunctions for OS navigation
HINGLISH_KEYWORDS = {
    # Actions / Verbs
    "kholo": "open", "khol": "open", "open karo": "open",
    "band karo": "close", "band kar do": "close", "hatao": "remove",
    "chalao": "play", "bajao": "play", "chala": "play", "play karo": "play",
    "dhundo": "search", "khojo": "search", "search karo": "search",
    "dekho": "look", "dekh": "look", "dikhaye": "show", "dikhao": "show",
    "likho": "type", "type karo": "type", "likh": "write",
    "banao": "create", "bana": "create",
    "delete karo": "delete", "hata do": "delete", "mita do": "delete",
    "refresh karo": "refresh", "reload karo": "reload",
    "batao": "tell me", "samjhao": "explain",
    "ruko": "wait", "ruk": "wait", "thoda ruko": "wait a moment",
    
    # Common words
    "aur": "and", "phir": "then", "ke baad": "after that",
    "pe": "on", "par": "on", "me": "in", "se": "from", "ko": "to",
    "mera": "my", "meri": "my", "mere": "my",
    "ek second": "one second", "zara": "please",
    "kaise": "how", "kya": "what", "kyun": "why",
}


class HindiEnglishTranslator:
    """
    Fast, reliable Hindi and Hinglish to English instruction translator.
    Ensures the downstream LLM / MagnumAgent receives clean, unambiguous English actions.
    """

    def __init__(self, use_llm_fallback: bool = True) -> None:
        self.use_llm_fallback = use_llm_fallback
        self._cache: Dict[str, str] = {}

    def detect_language(self, text: str) -> str:
        """
        Detect if text is 'hindi' (Devanagari script), 'hinglish' (Romanized Hindi),
        or 'english'.
        """
        if not text:
            return "english"

        # 1. Check for Devanagari script Unicode range (\u0900 - \u097F)
        if re.search(r"[\u0900-\u097F]", text):
            return "hindi"

        # 2. Check for Romanized Hinglish vocabulary
        clean = text.lower().translate(str.maketrans("", "", string.punctuation))
        words = set(clean.split())
        hinglish_matches = words.intersection(set(HINGLISH_KEYWORDS.keys()))

        # If 1 or more typical Hindi/Hinglish action markers exist, classify as Hinglish
        if hinglish_matches:
            return "hinglish"

        # Common multi-word Hinglish patterns
        multi_patterns = [
            r"\bek second\b", r"\bband karo\b", r"\bsearch karo\b",
            r"\bopen karo\b", r"\bpe jao\b", r"\bdekhna hai\b",
            r"\bchal raha\b", r"\bkarna hai\b", r"\bkaro aur\b"
        ]
        if any(re.search(pat, clean) for pat in multi_patterns):
            return "hinglish"

        return "english"

    def _fast_rule_translate(self, text: str) -> Optional[str]:
        """
        Ultra-fast (<1ms) pattern matcher for standard desktop commands in Hindi/Hinglish.
        Handles high-frequency commands without network roundtrip.
        """
        clean = text.strip()
        lower = clean.lower()

        # Direct Devanagari common shortcuts
        devanagari_map = {
            "सफारी खोलो": "open Safari",
            "क्रोम खोलो": "open Chrome",
            "टर्मिनल खोलो": "open Terminal",
            "यूट्यूब खोलो": "open YouTube",
            "यूट्यूब पर गाना चलाओ": "open YouTube and play a song",
            "गाना चलाओ": "play a song",
            "रुक जाओ": "stop",
            "रुको": "stop",
            "बंद करो": "close",
            "स्क्रीन देखो": "look at the screen",
            "क्या चल रहा है": "what is happening on screen",
            "डाउनलोड फोल्डर खोलो": "open downloads folder",
        }
        for dev_phrase, eng_phrase in devanagari_map.items():
            if dev_phrase in clean:
                # Replace matching phrase
                return clean.replace(dev_phrase, eng_phrase).strip()

        # Regex replacements for common Hinglish phrasing
        patterns = [
            # "chrome kholo aur youtube pe lofi chalao" -> "open chrome and play lofi on youtube"
            (r"(.*?)\s+(?:ko\s+)?kholo\s+aur\s+(.*?)\s+(?:pe|par)\s+(.*?)\s+chalao", r"open \1 and play \3 on \2"),
            # "(app) kholo" -> "open (app)"
            (r"^([a-zA-Z0-9\s]+?)\s+(?:ko\s+)?(?:kholo|open karo|khol do)$", r"open \1"),
            # "(app) band karo" -> "close (app)"
            (r"^([a-zA-Z0-9\s]+?)\s+(?:ko\s+)?(?:band karo|close karo|hata do)$", r"close \1"),
            # "google pe (query) search karo" -> "search for (query) on google"
            (r"(?:google|youtube|bing|safari)\s+(?:pe|par)\s+(.*?)\s+(?:search karo|dhundo|dekho)", r"search for \1 on Google"),
            # "(query) search karo" -> "search for (query)"
            (r"^(.*?)\s+(?:ko\s+)?(?:search karo|dhundo|khojo)$", r"search for \1"),
            # "(app/tab) refresh karo" -> "refresh (app/tab)"
            (r"^(.*?)\s+(?:ko\s+)?refresh karo$", r"refresh \1"),
            # "mera (folder) dikhao / open karo"
            (r"^mera\s+(.*?)\s+(?:folder\s+)?(?:dikhao|kholo)$", r"open my \1 folder"),
            # "screen dekho / kya chal raha hai"
            (r"^(?:screen|desktop)\s+(?:dekho|check karo)$", r"look at the screen and inspect active window"),
            # "wait / ruko"
            (r"^(?:ek second\s+)?(?:ruko|ruk ja|ruk jao)$", r"wait"),
        ]

        for pat, repl in patterns:
            m = re.search(pat, lower)
            if m:
                translated = re.sub(pat, repl, lower).strip()
                return translated

        return None

    def _llm_translate_sync(self, text: str, source_lang: str) -> Optional[str]:
        """Translate Hindi / Hinglish to imperative English using LLM."""
        try:
            from magnum.config import get_config
            import openai

            cfg = get_config()
            client = openai.OpenAI(
                base_url=cfg.nim_base_url,
                api_key=cfg.nim_api_key,
            )

            prompt = (
                "You are an expert real-time voice command translator for an autonomous AI desktop agent.\n"
                "The user gave an instruction in Hindi or Hinglish (code-mixed Hindi + English).\n"
                "Translate this into a concise, imperative, actionable English command for the OS/computer.\n"
                "Rules:\n"
                "- Output ONLY the English translation.\n"
                "- Do NOT include quotes, explanations, prefixes, or commentary.\n"
                "- Preserve all application names, URLs, file paths, and proper nouns exactly.\n\n"
                f"User Utterance: {text}\n"
                "English Command:"
            )

            response = client.chat.completions.create(
                model=cfg.default_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=60,
                timeout=4.0,
            )
            ans = response.choices[0].message.content
            if ans:
                clean_ans = ans.strip().strip('"').strip("'")
                return clean_ans
        except Exception as e:
            logger.debug(f"LLM translation notice: {e}")

        return None

    def translate(self, text: str) -> TranslationResult:
        """
        Translate input text to English if it is Hindi or Hinglish.
        Returns a TranslationResult.
        """
        if not text or not text.strip():
            return TranslationResult(
                original_text="",
                translated_text="",
                detected_language="english",
                was_translated=False,
            )

        trimmed = text.strip()
        lang = self.detect_language(trimmed)

        if lang == "english":
            return TranslationResult(
                original_text=trimmed,
                translated_text=trimmed,
                detected_language="english",
                was_translated=False,
            )

        # Check in-memory cache
        if trimmed in self._cache:
            return TranslationResult(
                original_text=trimmed,
                translated_text=self._cache[trimmed],
                detected_language=lang,
                was_translated=True,
            )

        # 1. Try fast local pattern translator (<1ms)
        fast_result = self._fast_rule_translate(trimmed)
        if fast_result:
            self._cache[trimmed] = fast_result
            return TranslationResult(
                original_text=trimmed,
                translated_text=fast_result,
                detected_language=lang,
                was_translated=True,
            )

        # 2. Try LLM semantic translation if enabled
        if self.use_llm_fallback:
            llm_result = self._llm_translate_sync(trimmed, lang)
            if llm_result:
                self._cache[trimmed] = llm_result
                return TranslationResult(
                    original_text=trimmed,
                    translated_text=llm_result,
                    detected_language=lang,
                    was_translated=True,
                )

        # Fallback: simple token replacement if all else fails
        fallback_words = []
        for w in trimmed.split():
            clean_w = w.lower().translate(str.maketrans("", "", string.punctuation))
            if clean_w in HINGLISH_KEYWORDS:
                fallback_words.append(HINGLISH_KEYWORDS[clean_w])
            else:
                fallback_words.append(w)
        fallback_str = " ".join(fallback_words)
        self._cache[trimmed] = fallback_str

        return TranslationResult(
            original_text=trimmed,
            translated_text=fallback_str,
            detected_language=lang,
            was_translated=True,
        )


# Singleton instance
_translator_instance: Optional[HindiEnglishTranslator] = None


def get_hindi_translator() -> HindiEnglishTranslator:
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = HindiEnglishTranslator()
    return _translator_instance
