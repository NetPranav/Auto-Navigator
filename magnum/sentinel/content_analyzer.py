"""
Content Analyzer — AI-powered content evaluation engine.
Uses NVIDIA NIM reasoning models to evaluate screen content against monitoring rules.
This is the intelligence core of the Sentinel system.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from PIL import Image
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ContentMatch(BaseModel):
    """Result of evaluating content against a monitoring rule."""
    matched: bool = False
    confidence: float = 0.0
    summary: str = ""
    importance: int = 0          # 1-10 scale
    matched_items: List[str] = Field(default_factory=list)
    suggested_action: str = ""
    raw_response: str = ""


class ContentAnalyzer:
    """
    Uses NIM reasoning to evaluate screen content against monitoring rules.
    Supports both text-only (OCR) and vision (screenshot + OCR) analysis.
    """

    def __init__(self, nim_client: Any = None) -> None:
        self._nim_client = nim_client

    @property
    def nim_client(self) -> Any:
        if self._nim_client is None:
            from magnum.intelligence.nim_client import NimClient
            self._nim_client = NimClient()
        return self._nim_client

    def evaluate_content(
        self,
        ocr_text: str,
        rule_description: str,
        watch_for: str,
        app_name: str = "",
        screenshot: Optional[Image.Image] = None,
    ) -> ContentMatch:
        """
        Ask AI: does this screen content match the monitoring rule?

        Args:
            ocr_text: Extracted text from the screen via OCR.
            rule_description: Human description of what to watch for.
            watch_for: Specific thing to watch for (e.g. "emails from Master").
            app_name: Name of the app being monitored.
            screenshot: Optional screenshot for vision analysis.
        """
        prompt = self._build_eval_prompt(ocr_text, rule_description, watch_for, app_name)
        system_prompt = self._build_system_prompt()

        try:
            raw = self.nim_client.call_reasoning(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2,
            )

            parsed = self.nim_client.parse_json_response(raw)
            if parsed:
                return ContentMatch(
                    matched=bool(parsed.get("matched", False)),
                    confidence=float(parsed.get("confidence", 0.0)),
                    summary=str(parsed.get("summary", "")),
                    importance=int(parsed.get("importance", 0)),
                    matched_items=list(parsed.get("matched_items", [])),
                    suggested_action=str(parsed.get("suggested_action", "")),
                    raw_response=raw,
                )

            # Fallback: simple keyword match if AI response fails to parse
            return self._fallback_match(ocr_text, watch_for)

        except Exception as e:
            logger.warning(f"Content analysis failed: {e}")
            return self._fallback_match(ocr_text, watch_for)

    def analyze_activity(
        self,
        ocr_text: str,
        app_name: str = "",
    ) -> Dict[str, Any]:
        """
        Analyze current screen to determine what the user is doing.
        Returns activity summary, importance, and app context.
        """
        prompt = (
            f"Analyze this screen content and describe what the user is currently doing.\n"
            f"App: {app_name or 'Unknown'}\n\n"
            f"Screen Text (OCR):\n```\n{ocr_text[:3000]}\n```\n\n"
            f"Respond in JSON:\n"
            f'{{\n'
            f'  "activity_summary": "1-2 sentence description of what the user is doing",\n'
            f'  "importance": "critical|high|medium|low",\n'
            f'  "importance_score": 1-10,\n'
            f'  "app_name": "detected app name",\n'
            f'  "context": "what file/page/conversation is open",\n'
            f'  "suggested_next": "what the user should do next"\n'
            f'}}'
        )

        system_prompt = (
            "You are Magnum, an intelligent desktop assistant that understands screen context.\n"
            "Analyze the screen text to understand what the user is currently doing.\n"
            "Be concise, accurate, and assess the importance of the current activity.\n"
            "Respond ONLY in valid JSON."
        )

        try:
            raw = self.nim_client.call_reasoning(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2,
            )
            parsed = self.nim_client.parse_json_response(raw)
            if parsed:
                return parsed
        except Exception as e:
            logger.warning(f"Activity analysis failed: {e}")

        # Fallback
        return {
            "activity_summary": f"Working in {app_name or 'unknown app'}",
            "importance": "medium",
            "importance_score": 5,
            "app_name": app_name,
            "context": "",
            "suggested_next": "",
        }

    def generate_suggestion(
        self,
        activity_log: str,
        current_screen_text: str,
        current_app: str = "",
    ) -> Dict[str, Any]:
        """
        Generate a proactive suggestion based on activity history and current context.
        """
        prompt = (
            f"You are Magnum, a proactive AI assistant.\n\n"
            f"Current App: {current_app or 'Unknown'}\n"
            f"Current Screen (OCR):\n```\n{current_screen_text[:2000]}\n```\n\n"
            f"Recent Activity Log:\n```\n{activity_log[:2000]}\n```\n\n"
            f"Based on what the user is doing and has been doing, suggest the most helpful "
            f"next action they should take. Consider urgency, importance, and context.\n\n"
            f"Respond in JSON:\n"
            f'{{\n'
            f'  "suggestion": "Clear, actionable suggestion",\n'
            f'  "reason": "Why this is the best next step",\n'
            f'  "priority": "critical|high|medium|low",\n'
            f'  "priority_emoji": "🔴|🟠|🟡|🟢",\n'
            f'  "category": "coding|communication|review|break|organization"\n'
            f'}}'
        )

        system_prompt = (
            "You are a proactive AI productivity assistant.\n"
            "Analyze the user's current activity and recent history to suggest the most "
            "impactful next action. Be specific, actionable, and prioritize by urgency.\n"
            "Respond ONLY in valid JSON."
        )

        try:
            raw = self.nim_client.call_reasoning(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
            )
            parsed = self.nim_client.parse_json_response(raw)
            if parsed:
                return parsed
        except Exception as e:
            logger.warning(f"Suggestion generation failed: {e}")

        return {
            "suggestion": "Continue your current task",
            "reason": "No specific suggestion available",
            "priority": "low",
            "priority_emoji": "🟢",
            "category": "organization",
        }

    def recall_from_journal(
        self,
        journal_text: str,
        question: str,
    ) -> str:
        """
        Answer a question about past activity using the activity journal.
        """
        prompt = (
            f"Here is a chronological activity log of what the user has been doing:\n\n"
            f"```\n{journal_text[:4000]}\n```\n\n"
            f"The user asks: \"{question}\"\n\n"
            f"Analyze the activity log and provide a clear, specific answer. "
            f"Mention specific apps, files, tasks, and timestamps."
        )

        system_prompt = (
            "You are Magnum, an AI assistant with perfect memory of the user's activities.\n"
            "Answer questions about what the user has been doing based on the activity log.\n"
            "Be specific, cite timestamps, and highlight the most important activities."
        )

        try:
            raw = self.nim_client.call_reasoning(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2,
            )
            return raw.strip() if raw else "I couldn't analyze your activity log at the moment."
        except Exception as e:
            logger.warning(f"Journal recall failed: {e}")
            return "I couldn't analyze your activity log at the moment."

    # ── Private Helpers ──

    def _build_eval_prompt(
        self, ocr_text: str, rule_desc: str, watch_for: str, app_name: str
    ) -> str:
        return (
            f"You are monitoring the app '{app_name}' for the user.\n"
            f"Rule: {rule_desc}\n"
            f"Watch for: {watch_for}\n\n"
            f"Screen Text (OCR extracted):\n```\n{ocr_text[:3000]}\n```\n\n"
            f"Does this screen content match the monitoring rule? "
            f"Analyze carefully and respond in JSON:\n"
            f'{{\n'
            f'  "matched": true/false,\n'
            f'  "confidence": 0.0-1.0,\n'
            f'  "summary": "what was found (1-2 sentences)",\n'
            f'  "importance": 1-10,\n'
            f'  "matched_items": ["list of specific matches found"],\n'
            f'  "suggested_action": "what the user should do about it"\n'
            f'}}'
        )

    def _build_system_prompt(self) -> str:
        return (
            "You are Magnum Sentinel, an intelligent content analysis engine.\n"
            "Your job is to evaluate screen content against monitoring rules.\n"
            "Be precise: only report genuine matches, not false positives.\n"
            "Respond ONLY in valid JSON."
        )

    def _fallback_match(self, ocr_text: str, watch_for: str) -> ContentMatch:
        """Simple keyword-based fallback matching when AI is unavailable."""
        lower_text = ocr_text.lower()
        lower_watch = watch_for.lower()

        # Extract key terms from watch_for
        terms = [t.strip() for t in lower_watch.split() if len(t.strip()) > 2]
        matched_terms = [t for t in terms if t in lower_text]

        if len(matched_terms) >= max(1, len(terms) // 2):
            return ContentMatch(
                matched=True,
                confidence=len(matched_terms) / max(len(terms), 1),
                summary=f"Keyword match: found {', '.join(matched_terms)}",
                importance=5,
                matched_items=matched_terms,
            )

        return ContentMatch(matched=False, confidence=0.0)
