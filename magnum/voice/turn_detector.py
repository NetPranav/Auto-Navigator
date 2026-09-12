"""
Magnum Smart Turn Detector & Interruption Manager.
Inspired by LiveKit Agents (livekit/agents voice/turn.py) and BargeKit (bargekit).

Provides:
- Dynamic endpointing with hesitation and thinking-pause resilience
- Fake interruption rejection (filters out coughs, mic thumps, short noises <350ms, and backchannels)
- Multi-lingual filler and disfluency recognition (English + Hindi / Hinglish)
- Barge-in qualification and state machine
"""

from __future__ import annotations

import enum
import logging
import re
import string
import time
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class TurnState(enum.Enum):
    """BargeKit-compatible turn-taking state machine."""
    IDLE = "idle"
    LISTENING = "listening"
    HESITATION_HELD = "hesitation_held"   # User is thinking/hesitating — extend silence
    AGENT_SPEAKING = "agent_speaking"     # Agent TTS active
    BARGE_PENDING = "barge_pending"       # User sound during agent speech being evaluated
    INTERRUPTED = "interrupted"           # Qualified true interruption
    NOISE_GATED = "noise_gated"           # Short transient / cough ignored (fake interruption rejected)
    COOLDOWN = "cooldown"


@dataclass
class TurnDetectorConfig:
    """Turn detection and interruption qualification parameters."""
    # Silence timeouts (seconds)
    min_delay: float = 0.8             # Standard pause to confirm turn completion
    hesitation_delay: float = 3.2      # Silence allowed when user is thinking/hesitating
    max_delay: float = 4.5             # Absolute maximum silence allowed before closing turn

    # Barge-in qualification thresholds
    min_interruption_duration: float = 0.35  # Audio bursts shorter than this are noise-gated (<350ms)
    false_interruption_timeout: float = 2.0  # Silence window before declaring a false interruption
    resume_false_interruption: bool = True   # Resume agent output if interruption was false

    # Explicit high-priority stop/cancel keywords that ALWAYS trigger instant interruption
    stop_keywords: Set[str] = field(default_factory=lambda: {
        "stop", "cancel", "halt", "wait", "shh", "shut up", "ruko", "ruk",
        "bas", "rok", "roko", "band karo", "chup", "pause", "terminate"
    })

    # Non-interruptive backchannels (sounds users make when agreeing or listening)
    backchannels: Set[str] = field(default_factory=lambda: {
        "uh-huh", "uh huh", "mm-hmm", "mm hmm", "hmm", "yeah", "yes", "okay",
        "ok", "haan", "accha", "achha", "right", "sure"
    })

    # English disfluency & thinking filler words
    english_fillers: Set[str] = field(default_factory=lambda: {
        "uh", "um", "uhm", "umm", "uhh", "err", "erm", "ah", "hmm", "well", "wait",
        "hold on", "let me think", "let me see", "like", "so", "actually", "basically",
        "you know", "i mean"
    })

    # Hindi / Hinglish disfluency & thinking filler words
    hindi_fillers: Set[str] = field(default_factory=lambda: {
        "arre", "are", "ek second", "ek sec", "ruko", "ruk", "matlab", "socho",
        "dekh", "dekho", "toh", "to", "aur", "ki", "acha", "achha", "bhai",
        "suno", "wait karo", "phir", "lekin", "magar", "yaani", "ruk jao", "thoda"
    })

    # Trailing connectors indicating incomplete sentence (don't cut off!)
    trailing_connectors: Set[str] = field(default_factory=lambda: {
        "and", "or", "to", "that", "with", "for", "in", "on", "into", "because",
        "then", "so", "but", "if", "after", "before", "about",
        "aur", "ki", "par", "me", "se", "ko", "karke", "ke baad", "wale", "wala", "ya"
    })


class TurnDetector:
    """
    Intelligent turn-taking detector with fake-interruption prevention
    and adaptive thinking silence buffers.
    """

    def __init__(self, config: Optional[TurnDetectorConfig] = None) -> None:
        self.config = config or TurnDetectorConfig()
        self.state = TurnState.IDLE
        self._last_speech_time: float = 0.0
        self._interruption_start_time: Optional[float] = None
        self._hesitation_detected: bool = False

    def is_filler_or_hesitation(self, text: str) -> bool:
        """Check if an utterance consists of or ends in hesitation/thinking markers."""
        if not text:
            return False

        clean = text.lower().strip()
        words = clean.translate(str.maketrans("", "", string.punctuation)).split()
        if not words:
            return False

        # Entire utterance is a filler
        if clean in self.config.english_fillers or clean in self.config.hindi_fillers:
            return True

        # Last word or two matches a filler
        last_word = words[-1]
        last_two = " ".join(words[-2:]) if len(words) >= 2 else ""

        if last_word in self.config.english_fillers or last_word in self.config.hindi_fillers:
            return True
        if last_two in self.config.english_fillers or last_two in self.config.hindi_fillers:
            return True

        # Trailing connector (e.g., "I want you to...", "Safari open karo aur...")
        if last_word in self.config.trailing_connectors:
            return True

        # Incomplete starter phrases
        incomplete_starts = (
            "i want you to", "i want you", "i want to", "i want", "i need you to",
            "i need to", "can you", "could you", "please", "ek second", "ruko ek second",
            "mujhe chahiye ki", "kya tum", "zara", "soch raha hu", "let me think",
            "let me", "i am trying to"
        )
        if any(clean.startswith(prefix) for prefix in incomplete_starts) and len(words) <= 5:
            return True

        return False

    def is_incomplete_clause(self, text: str) -> bool:
        """Detect trailing incomplete grammar indicating the user hasn't finished speaking."""
        if not text:
            return False
        words = text.lower().strip().translate(str.maketrans("", "", string.punctuation)).split()
        if not words:
            return False
        return words[-1] in self.config.trailing_connectors

    def get_adaptive_silence_timeout(self, partial_transcript: Optional[str] = None) -> float:
        """
        Dynamically determine how long to wait before concluding user speech.
        If user is hesitating/thinking, returns hesitation_delay (3.2s) instead of min_delay (0.8s).
        """
        if not partial_transcript:
            return self.config.min_delay

        if self.is_filler_or_hesitation(partial_transcript) or self.is_incomplete_clause(partial_transcript):
            self._hesitation_detected = True
            logger.info(f"⏳ Hesitation detected in '{partial_transcript}'. Extending silence timeout to {self.config.hesitation_delay}s")
            return self.config.hesitation_delay

        return self.config.min_delay

    def qualify_barge_in(
        self,
        audio_duration_sec: float,
        transcript: Optional[str] = None,
        is_agent_speaking: bool = True,
    ) -> Tuple[bool, str]:
        """
        Determine if an audio event is a genuine interruption or a fake interruption.
        
        Fake interruptions rejected:
        - Burst duration < 0.35s (throat clear, cough, keyclick, mic bump)
        - Non-substantive backchannels ("mm-hmm", "uh-huh", "yeah")
        
        Genuine interruptions accepted:
        - Explicit stop keywords ("stop", "ruko", "cancel", "wait")
        - Sustained utterance with actionable words (>0.35s and >=1 meaningful word)
        """
        if not is_agent_speaking:
            return True, "agent_not_speaking"

        # 1. Reject transient short noises (Fake Interruption rejection)
        if audio_duration_sec < self.config.min_interruption_duration:
            self.state = TurnState.NOISE_GATED
            return False, f"noise_gated: duration {audio_duration_sec:.2f}s < threshold {self.config.min_interruption_duration:.2f}s"

        if not transcript:
            # If duration is substantial (>0.7s) even without decoded words yet, consider barge-in
            if audio_duration_sec >= 0.7:
                self.state = TurnState.INTERRUPTED
                return True, "sustained_audio_energy"
            self.state = TurnState.NOISE_GATED
            return False, "unrecognized_short_audio"

        clean = transcript.lower().strip()
        words = clean.translate(str.maketrans("", "", string.punctuation)).split()

        # 2. Check explicit stop/barge-in keywords (Instant interrupt)
        for w in words:
            if w in self.config.stop_keywords:
                self.state = TurnState.INTERRUPTED
                return True, f"stop_keyword: '{w}'"

        # 3. Check for backchannels (Fake Interruption rejection)
        if clean in self.config.backchannels or (len(words) == 1 and words[0] in self.config.backchannels):
            self.state = TurnState.NOISE_GATED
            return False, f"suppressed_backchannel: '{clean}'"

        # 4. Check word count (must have at least 1 meaningful word)
        meaningful_words = [w for w in words if w not in self.config.english_fillers and w not in self.config.hindi_fillers]
        if not meaningful_words:
            # Utterance was purely hesitation fillers ("uh... um...")
            self.state = TurnState.HESITATION_HELD
            return False, f"suppressed_filler_during_speech: '{clean}'"

        # Qualified genuine interruption!
        self.state = TurnState.INTERRUPTED
        return True, f"qualified_speech: '{clean}'"

    def mark_speech_start(self) -> None:
        """Record the onset of user speech."""
        self.state = TurnState.LISTENING
        self._interruption_start_time = time.time()
        self._hesitation_detected = False

    def mark_speech_end(self) -> None:
        """Record the conclusion of user speech."""
        self.state = TurnState.COOLDOWN
        self._last_speech_time = time.time()
        self._interruption_start_time = None
