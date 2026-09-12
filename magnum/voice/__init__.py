"""
Magnum Voice Engine — Wake Word, Real-Time Speech-to-Text, and Text-to-Speech.
Enables hands-free voice activation ("Hey", "Magnum", "Hey Magnum", "Jarvis") on macOS
with Sensitive VAD, Live Terminal Transcription Feedback, and Custom Acoustic Calibration.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import os
import string
import subprocess
import time
from typing import Callable, Deque, List, Optional, Tuple
import numpy as np
from rich.console import Console
import sounddevice as sd
import speech_recognition as sr

from magnum.voice.calibration import load_calibration

logger = logging.getLogger(__name__)
console = Console()

# Standard macOS system chime sounds
CHIME_WAKE = "/System/Library/Sounds/Ping.aiff"
CHIME_DONE = "/System/Library/Sounds/Glass.aiff"
CHIME_CANCEL = "/System/Library/Sounds/Basso.aiff"
CHIME_TICK = "/System/Library/Sounds/Tink.aiff"
CHIME_SUCCESS = "/System/Library/Sounds/Hero.aiff"


class VoiceEngine:
    """
    Hands-free Voice Engine for Magnum.
    Features:
    - High-sensitivity VAD calibrated to ambient noise and near/far voice
    - Live terminal feedback showing every transcribed utterance
    - Punctuation-normalized command extraction
    - Non-blocking responsive loop that exits cleanly on Ctrl+C
    - Echo prevention (mutes microphone during speech synthesis)
    """

    def __init__(
        self,
        wake_words: Optional[List[str]] = None,
        sample_rate: int = 16000,
        voice_name: str = "Samantha",
    ) -> None:
        raw_ww = wake_words or [
            "hey", "magnum", "hey magnum", "jarvis", "hey jarvis",
            "hi", "hello", "magnam", "magnet", "ay"
        ]
        self.wake_words = [w.lower().strip() for w in raw_ww]
        self.sample_rate = sample_rate
        self.voice_name = voice_name
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 250
        self.recognizer.dynamic_energy_threshold = False
        self.is_listening = False
        self._stop_event = asyncio.Event()

        # Load custom calibration if available
        calib = load_calibration()
        if calib:
            self.speech_thresh: float = calib.get("speech_threshold", 220.0)
            self.silence_thresh: float = calib.get("silence_threshold", 150.0)
            self.ambient_floor: float = calib.get("ambient_mean", 100.0)
            logger.info(f"Loaded custom acoustic calibration: speech={self.speech_thresh}, silence={self.silence_thresh}")
        else:
            self.ambient_floor = 120.0
            self.speech_thresh = 200.0
            self.silence_thresh = 140.0

        self._current_tts_proc: Optional[subprocess.Popen] = None

    def play_chime(self, sound_path: str = CHIME_WAKE) -> None:
        """Play a subtle native macOS system audio chime."""
        if os.path.exists(sound_path):
            try:
                subprocess.Popen(
                    ["afplay", sound_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

    def stop_speaking(self) -> None:
        """Interrupt and stop any ongoing speech synthesis immediately."""
        if self._current_tts_proc:
            try:
                self._current_tts_proc.terminate()
            except Exception:
                pass
            self._current_tts_proc = None

    def speak(self, text: str, blocking: bool = True) -> None:
        """
        Speak response aloud using macOS native speech synthesizer.
        Blocks until finished + short cooldown so the mic doesn't hear the computer's own voice.
        Supports instant barge-in interruption via stop_speaking().
        """
        if not text or not text.strip():
            return
        self.stop_speaking()
        clean_text = text.replace("*", "").replace("#", "").replace("`", "").replace("✓", "").replace("⚡", "").strip()
        logger.info(f"🔊 Magnum Speaking: {clean_text}")
        try:
            cmd = ["say", "-v", self.voice_name, clean_text]
            self._current_tts_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if blocking:
                self._current_tts_proc.wait()
                time.sleep(0.15)
                self._current_tts_proc = None
        except Exception as e:
            logger.warning(f"Voice speech synthesis notice: {e}")

    def capture_phrase_sync(
        self,
        max_duration: float = 30.0,
        no_speech_timeout: float = 3.0,
        silence_timeout: float = 3.0,
    ) -> Optional[sr.AudioData]:
        """
        Record audio with sensitive calibrated VAD and 3.0-second silence timeout.
        Allows the user to pause, think, and speak naturally without premature cutoff.
        """
        if self._stop_event.is_set() or not self.is_listening:
            return None

        chunk_size = int(self.sample_rate * 0.1)  # 100ms chunks
        silence_chunks_needed = max(5, int(silence_timeout / 0.1))  # 30 chunks = 3.0s of continuous silence
        pre_roll: Deque[np.ndarray] = collections.deque(maxlen=3)
        recorded_frames: List[np.ndarray] = []

        is_speaking = False
        consecutive_silent = 0
        start_time = time.time()

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=chunk_size,
            ) as stream:
                while time.time() - start_time < max_duration:
                    if self._stop_event.is_set() or not self.is_listening:
                        return None

                    chunk, _ = stream.read(chunk_size)
                    audio_chunk = chunk.flatten()
                    rms = float(np.sqrt(np.mean(audio_chunk.astype(float) ** 2)))

                    if not is_speaking:
                        # Update ambient floor only on quiet samples
                        if rms < self.ambient_floor * 1.25:
                            self.ambient_floor = 0.96 * self.ambient_floor + 0.04 * rms

                        # Dynamically adapt thresholds if not hard-calibrated
                        speech_trigger = max(self.speech_thresh, self.ambient_floor * 1.35)
                        pre_roll.append(audio_chunk)

                        if rms > speech_trigger:
                            is_speaking = True
                            recorded_frames.extend(list(pre_roll))
                            pre_roll.clear()
                            consecutive_silent = 0

                        elif time.time() - start_time >= no_speech_timeout:
                            return None
                    else:
                        recorded_frames.append(audio_chunk)
                        silence_trigger = max(self.silence_thresh, self.ambient_floor * 1.15)

                        if rms < silence_trigger:
                            consecutive_silent += 1
                            if consecutive_silent >= silence_chunks_needed:
                                break
                        else:
                            consecutive_silent = 0

        except Exception as e:
            logger.debug(f"Audio stream read error: {e}")
            return None

        if not is_speaking or not recorded_frames:
            return None

        raw_bytes = np.concatenate(recorded_frames).tobytes()
        return sr.AudioData(raw_bytes, self.sample_rate, 2)

    def transcribe_audio(self, audio: sr.AudioData) -> Optional[str]:
        """Convert recorded AudioData to text via speech recognition."""
        if self._stop_event.is_set() or not self.is_listening:
            return None
        try:
            text = self.recognizer.recognize_google(audio)
            return text.strip()
        except sr.UnknownValueError:
            return None
        except Exception as e:
            logger.debug(f"Speech transcription notice: {e}")
            return None

    def extract_command(self, transcript: str) -> Tuple[bool, Optional[str]]:
        """
        Robust wake-word and command extraction.
        Normalizes punctuation and matches longest wake words first.
        """
        text = transcript.lower().strip()
        clean = text.translate(str.maketrans("", "", string.punctuation)).strip()

        # Sort wake words by length descending so "hey magnum" matches before "hey"
        sorted_ww = sorted(self.wake_words, key=len, reverse=True)
        for ww in sorted_ww:
            clean_ww = ww.strip().lower()
            if clean == clean_ww:
                return True, None
            if clean.startswith(clean_ww + " "):
                cmd = clean[len(clean_ww):].strip()
                return True, cmd if cmd else None

        return False, None

    async def listen_for_instruction(self, silence_timeout: float = 2.2) -> Optional[str]:
        """Actively listen for a spoken instruction after wake word."""
        loop = asyncio.get_running_loop()
        self.play_chime(CHIME_WAKE)
        audio = await loop.run_in_executor(
            None,
            lambda: self.capture_phrase_sync(max_duration=30.0, no_speech_timeout=5.0, silence_timeout=silence_timeout),
        )
        if not audio or self._stop_event.is_set():
            return None
        return await loop.run_in_executor(None, lambda: self.transcribe_audio(audio))

    async def listen_loop(
        self,
        on_command: Callable[[str], asyncio.Future],
        on_wake: Optional[Callable[[], None]] = None,
        silence_timeout: Optional[float] = None,
    ) -> None:
        """
        Continuously listen in background for wake word and dispatch commands.
        Uses adaptive silence timeout (3.0s default, 2.2s after prompts) so you can speak naturally.
        """
        self.is_listening = True
        self._stop_event.clear()
        loop = asyncio.get_running_loop()
        vad_silence = silence_timeout or 3.0

        console.print(f"[dim]🎤 Voice listener active (Wake words: 'Hey', 'Magnum', 'Jarvis' | {vad_silence:.1f}s silence timeout)[/dim]")

        incomplete_fillers = [
            "i want you to", "i want you", "i need you to", "can you",
            "could you", "please", "do this", "do this that", "i want", "i am",
        ]

        while self.is_listening and not self._stop_event.is_set():
            try:
                audio = await loop.run_in_executor(
                    None,
                    lambda: self.capture_phrase_sync(max_duration=30.0, no_speech_timeout=2.5, silence_timeout=vad_silence),
                )
                if not audio or self._stop_event.is_set():
                    await asyncio.sleep(0.05)
                    continue

                transcript = await loop.run_in_executor(None, lambda: self.transcribe_audio(audio))
                if not transcript or self._stop_event.is_set():
                    await asyncio.sleep(0.05)
                    continue

                # VISIBLE IN TERMINAL: Show user what the microphone captured!
                console.print(f"[dim]🎙️  Heard: '{transcript}'[/dim]")

                detected, command = self.extract_command(transcript)
                if not detected:
                    # Allow direct high-priority system commands even if wake word was skipped or missed
                    lower_clean = transcript.lower().strip()
                    direct_system_keywords = (
                        "start a workflow", "create a workflow", "record a workflow", "new workflow",
                        "start the workflow", "create the workflow", "start workflow", "create workflow",
                        "start a voucher", "create a workshop", "start a work flow",
                        "stop watching", "cancel watchers", "stop all watchers", "stop watcher",
                        "what are you watching", "list watchers", "watcher status",
                        "list workflows", "show workflows", "what workflows do i have",
                    )
                    if any(kw in lower_clean for kw in direct_system_keywords):
                        detected = True
                        command = transcript
                    else:
                        continue

                # Wake word detected!
                self.stop_speaking()
                console.print(f"[bold green]⚡ Wake Word Detected: '{transcript}'[/bold green]")
                self.play_chime(CHIME_WAKE)
                if on_wake:
                    on_wake()

                if not command:
                    # Heard just wake word ("hey" or "magnum")
                    console.print("[bold yellow]🎤 Magnum:[/bold yellow] Yes?")
                    self.speak("Yes?", blocking=True)
                    console.print("[dim]🎤 Listening for your command...[/dim]")
                    command = await self.listen_for_instruction(silence_timeout=2.2)

                # Guard against incomplete filler phrases like "i want you to"
                if command and any(command.strip().lower() == filler for filler in incomplete_fillers):
                    console.print(f"[bold yellow]🎤 Incomplete phrase ('{command}'). Waiting for complete instruction...[/bold yellow]")
                    self.speak("I'm listening, go ahead.", blocking=True)
                    console.print("[dim]🎤 Continue speaking your task...[/dim]")
                    more_command = await self.listen_for_instruction(silence_timeout=2.5)
                    if more_command:
                        command = f"{command} {more_command}".strip()
                    else:
                        command = None

                if command and not self._stop_event.is_set():
                    console.print(f"[bold cyan]⚡ Executing Command:[/bold cyan] '{command}'")
                    self.speak("On it.", blocking=False)
                    await on_command(command)
                    self.play_chime(CHIME_DONE)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Voice listening loop notice: {e}")
                self.play_chime(CHIME_CANCEL)
                await asyncio.sleep(0.2)

        self.is_listening = False

    def stop(self) -> None:
        """Immediately stop listening and break streams."""
        self.is_listening = False
        self._stop_event.set()


# Singleton instance
_voice_engine: Optional[VoiceEngine] = None


def get_voice_engine() -> VoiceEngine:
    global _voice_engine
    if _voice_engine is None:
        _voice_engine = VoiceEngine()
    return _voice_engine
