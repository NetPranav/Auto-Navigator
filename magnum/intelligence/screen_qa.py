"""
Visual Screen Q&A Engine for Auto-Navigator (Magnum / Jarvis).
Combines Apple Vision OCR ground-truth with NVIDIA NIM Multimodal Vision (Llama-3.2-11B-Vision)
and Nemotron-120B reasoning to visually comprehend and explain the user's active display.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from PIL import Image

from magnum.intelligence.grounding import ScreenGrounder, TextElement

logger = logging.getLogger(__name__)


@dataclass
class ScreenQAResult:
    headline: str
    analysis: str
    speak_text: str
    detected_errors: List[str] = field(default_factory=list)
    active_apps: List[str] = field(default_factory=list)


class ScreenQARunner:
    """Multimodal screen inspector and visual question-answering engine."""

    @staticmethod
    def _image_to_base64(image: Image.Image, max_dim: int = 1400) -> str:
        """Resize image preserving aspect ratio and convert to base64 JPEG."""
        img = image.copy()
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @classmethod
    async def capture_and_analyze(
        cls,
        driver: Any,
        nim_client: Any,
        user_query: Optional[str] = None,
        target_app: Optional[str] = None,
    ) -> ScreenQAResult:
        """
        Capture the screen/window, extract OCR text, and query NVIDIA NIM Vision / Reasoning.
        """
        # 1. Capture screen buffer
        try:
            if hasattr(driver, "take_screenshot"):
                screenshot = driver.take_screenshot(target_app=target_app, filename="screen_qa.png")
            else:
                screenshot = await driver.screenshot("screen_qa.png")
        except Exception as e:
            logger.error(f"Failed to capture screen for Q&A: {e}")
            from PIL import ImageGrab
            screenshot = ImageGrab.grab()

        # 2. Extract OCR ground truth via Apple Vision
        elements = ScreenGrounder.extract_screen_text_elements(screenshot)
        raw_texts = [el.text.strip() for el in elements if len(el.text.strip()) > 1]
        ocr_manifest = "\n".join(raw_texts[:120])

        # 3. Detect prominent errors or stack traces in OCR
        error_keywords = ("error", "traceback", "exception", "failed", "fatal", "syntaxerror", "typeerror", "cannot find", "errno")
        detected_errors = []
        for line in raw_texts:
            line_lower = line.lower()
            if any(k in line_lower for k in error_keywords) and len(line) < 120:
                if line not in detected_errors:
                    detected_errors.append(line)

        # 4. Detect open apps / window titles from OCR headers
        active_apps = []
        for line in raw_texts[:20]:
            for app in ("Chrome", "Safari", "Terminal", "Antigravity", "Visual Studio Code", "Slack", "Discord", "Finder"):
                if app.lower() in line.lower() and app not in active_apps:
                    active_apps.append(app)

        # 5. Build prompt
        query = (user_query or "").strip()
        if not query:
            query = "Explain what is on my screen, identify any errors or warnings, active applications, and what I should do next."

        system_prompt = (
            "You are Jarvis, a brilliant, hyper-competent AI desktop visual assistant.\n"
            "Analyze the user's active screen capture with precision.\n"
            "Structure your answer with:\n"
            "1. **Overview**: A crisp summary of the main active application and current visual context.\n"
            "2. **Key Findings / Errors**: Specifically highlight visible errors, stack traces, warnings, or interesting UI elements.\n"
            "3. **Suggested Next Action**: Concrete code fix, terminal command, or workflow step to take.\n"
            "Be direct, insightful, and concise."
        )

        user_content = (
            f"User Question: {query}\n\n"
            f"Ground-Truth Screen Text (Extracted via OCR):\n"
            f"```\n{ocr_manifest[:3500]}\n```\n"
        )

        image_b64 = cls._image_to_base64(screenshot)

        # 6. Query NVIDIA NIM Vision
        analysis_text = ""
        try:
            # Try multimodal vision completion
            if hasattr(nim_client, "call_vision") and not getattr(nim_client, "is_simulation", False):
                analysis_text = nim_client.call_vision(
                    prompt=user_content,
                    image_base64=image_b64,
                    system_prompt=system_prompt,
                    temperature=0.2,
                )

            # If empty or in simulation mode, fallback to reasoning model with OCR context
            if not analysis_text:
                analysis_text = nim_client.call_reasoning(
                    prompt=user_content,
                    system_prompt=system_prompt,
                    temperature=0.2,
                )
        except Exception as e:
            logger.warning(f"Screen Q&A NIM call failed: {e}")
            if detected_errors:
                analysis_text = (
                    f"**Detected Errors on Screen:**\n" + "\n".join(f"- `{e}`" for e in detected_errors[:5]) +
                    f"\n\n**Visible Text Summary:**\n{ocr_manifest[:400]}..."
                )
            else:
                analysis_text = f"**Screen Content Preview:**\n{ocr_manifest[:500]}..."

        # 7. Formulate 1-sentence spoken headline
        headline = "Screen analyzed."
        for line in analysis_text.split("\n"):
            clean_l = line.strip(" *#-`")
            if clean_l and len(clean_l) > 15:
                headline = clean_l
                break

        if detected_errors:
            speak_text = f"I examined your screen. I detected {len(detected_errors)} errors, including {detected_errors[0][:60]}."
        else:
            app_mention = f"in {active_apps[0]}" if active_apps else "on your desktop"
            speak_text = f"I've analyzed your screen. You are working {app_mention}. {headline[:90]}."

        return ScreenQAResult(
            headline=headline,
            analysis=analysis_text,
            speak_text=speak_text,
            detected_errors=detected_errors,
            active_apps=active_apps,
        )
