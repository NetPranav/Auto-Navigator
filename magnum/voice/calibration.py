"""
Magnum Voice Calibration Wizard.
Measures ambient acoustics, near/normal voice, and far/soft voice to calculate
optimal VAD thresholds specifically tailored to the user's room and voice.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
import sounddevice as sd

console = Console()
CALIBRATION_FILE = Path.home() / ".magnum_voice_calib.json"


def record_samples(duration_sec: float, sample_rate: int = 16000) -> List[float]:
    """Record audio in 100ms chunks and return list of RMS values."""
    chunk_size = int(sample_rate * 0.1)  # 100ms
    num_chunks = int(duration_sec / 0.1)
    rms_values = []

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        blocksize=chunk_size,
    ) as stream:
        for _ in range(num_chunks):
            chunk, _ = stream.read(chunk_size)
            rms = float(np.sqrt(np.mean(chunk.flatten().astype(float) ** 2)))
            rms_values.append(rms)

    return rms_values


def run_calibration() -> Dict[str, float]:
    """Run interactive 3-step calibration wizard."""
    console.print(Panel(
        "[bold cyan]🎙️ MAGNUM VOICE CALIBRATION WIZARD[/bold cyan]\n"
        "[white]We will calibrate your microphone in 3 quick steps:[/white]\n"
        "  1. Ambient room silence (2 seconds)\n"
        "  2. Normal voice: say 'Hey' or 'Magnum' close/normal\n"
        "  3. Distant voice: say 'Hey' from a little further away",
        title="[bold yellow]Audio Calibration[/bold yellow]",
        border_style="cyan",
    ))

    # Step 1: Ambient room noise
    Prompt.ask("\n[bold yellow]Step 1/3:[/bold yellow] Stay quiet for 2 seconds to measure room noise. Press [bold]Enter[/bold] when ready")
    console.print("[dim]Measuring room acoustics...[/dim]")
    ambient_rms = record_samples(2.0)
    ambient_mean = float(np.mean(ambient_rms))
    ambient_max = float(np.max(ambient_rms))
    console.print(f"[green]✓ Ambient Noise: Mean = {ambient_mean:.1f}, Max = {ambient_max:.1f}[/green]")

    # Step 2: Normal voice
    Prompt.ask("\n[bold yellow]Step 2/3:[/bold yellow] Press [bold]Enter[/bold], then clearly say [bold cyan]'Hey'[/bold cyan] in your normal voice")
    console.print("[bold cyan]🎙️ Listening now... Say 'Hey' or 'Magnum'![/bold cyan]")
    normal_rms = record_samples(2.5)
    normal_peak = float(np.max(normal_rms))
    normal_speech_samples = [x for x in normal_rms if x > ambient_mean * 1.5]
    normal_avg = float(np.mean(normal_speech_samples)) if normal_speech_samples else normal_peak
    console.print(f"[green]✓ Normal Voice: Peak = {normal_peak:.1f}, Speech Avg = {normal_avg:.1f}[/green]")

    # Step 3: Far / Soft voice
    Prompt.ask("\n[bold yellow]Step 3/3:[/bold yellow] Press [bold]Enter[/bold], then say [bold cyan]'Hey'[/bold cyan] from a little farther away or softer")
    console.print("[bold cyan]🎙️ Listening now... Say 'Hey' from farther back![/bold cyan]")
    far_rms = record_samples(2.5)
    far_peak = float(np.max(far_rms))
    far_speech_samples = [x for x in far_rms if x > ambient_mean * 1.3]
    far_avg = float(np.mean(far_speech_samples)) if far_speech_samples else far_peak
    console.print(f"[green]✓ Distant/Soft Voice: Peak = {far_peak:.1f}, Speech Avg = {far_avg:.1f}[/green]")

    # Compute optimal thresholds
    # Speech threshold should be lower than far voice peak, but above ambient max
    target_low = min(normal_avg, far_avg)
    speech_thresh = max(ambient_max * 1.15, min(target_low * 0.7, ambient_mean * 1.5))
    silence_thresh = max(ambient_mean * 1.2, speech_thresh * 0.65)

    calib_data = {
        "ambient_mean": round(ambient_mean, 1),
        "ambient_max": round(ambient_max, 1),
        "normal_peak": round(normal_peak, 1),
        "far_peak": round(far_peak, 1),
        "speech_threshold": round(speech_thresh, 1),
        "silence_threshold": round(silence_thresh, 1),
    }

    # Save to file
    try:
        CALIBRATION_FILE.write_text(json.dumps(calib_data, indent=2))
        console.print(f"\n[dim]Saved calibration to {CALIBRATION_FILE}[/dim]")
    except Exception as e:
        console.warning(f"Could not save calibration file: {e}")

    # Display results table
    table = Table(title="📊 Calibration Results", border_style="green")
    table.add_column("Metric", style="bold white")
    table.add_column("Energy (RMS)", style="bold yellow")
    table.add_column("Description", style="dim")

    table.add_row("Ambient Room Noise", f"{ambient_mean:.1f}", "Background floor")
    table.add_row("Normal Voice Peak", f"{normal_peak:.1f}", "Voice at normal distance")
    table.add_row("Distant Voice Peak", f"{far_peak:.1f}", "Voice when sitting back/farther")
    table.add_row("Speech Trigger Threshold", f"{speech_thresh:.1f}", "Sound level required to start listening")
    table.add_row("Silence Cutoff Threshold", f"{silence_thresh:.1f}", "Sound level recognized as silence")

    console.print(table)
    console.print("\n[bold green]🎉 Calibration Complete! Magnum is now tuned specifically to your room & voice.[/bold green]\n")

    return calib_data


def load_calibration() -> Optional[Dict[str, float]]:
    """Load existing calibration data if available."""
    if CALIBRATION_FILE.exists():
        try:
            return json.loads(CALIBRATION_FILE.read_text())
        except Exception:
            pass
    return None
