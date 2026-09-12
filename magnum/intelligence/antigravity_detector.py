"""
Antigravity IDE State Detector.
Uses Apple Vision OCR to analyze screenshots and determine what state the
Antigravity IDE is currently in. This is the perception layer of the Autopilot.

Detectable states:
  IDLE            — Chat input visible, AI not working
  THINKING        — AI is generating (spinner, dots, "Thinking…")
  IMPLEMENTATION_PLAN — Plan artifact visible with "Proceed" button
  EXECUTING       — Tool calls running, file edits being applied
  PLAN_WITH_ROADMAP — Multi-step task list / roadmap detected
  WALKTHROUGH     — Walkthrough / completion summary visible
  AWAITING_INPUT  — Input field focused and empty
  ERROR           — Error messages or failure indicators
  DONE            — "Goal Completed" or similar completion signal
  UNKNOWN         — Cannot determine state
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Dict, List, Optional, Tuple

from PIL import Image
from pydantic import BaseModel, Field

from magnum.intelligence.grounding import ScreenGrounder, TextElement

logger = logging.getLogger(__name__)


class AntigravityState(str, Enum):
    """Possible states of the Antigravity IDE."""
    IDLE = "idle"
    THINKING = "thinking"
    IMPLEMENTATION_PLAN = "implementation_plan"
    EXECUTING = "executing"
    PLAN_WITH_ROADMAP = "plan_with_roadmap"
    WALKTHROUGH = "walkthrough"
    AWAITING_INPUT = "awaiting_input"
    ERROR = "error"
    DONE = "done"
    UNKNOWN = "unknown"


class PlanContent(BaseModel):
    """Extracted content from an implementation plan."""
    title: str = ""
    sections: List[str] = Field(default_factory=list)
    has_proceed_button: bool = False
    proceed_coords: Optional[Tuple[float, float]] = None
    raw_text: str = ""


class RoadmapStep(BaseModel):
    """A single step extracted from a roadmap/task list."""
    index: int
    text: str
    is_completed: bool = False


class DetectionResult(BaseModel):
    """Result of a state detection scan."""
    state: AntigravityState = AntigravityState.UNKNOWN
    confidence: float = 0.0
    details: str = ""
    elements_count: int = 0
    # Useful coordinates found during detection
    proceed_coords: Optional[Tuple[float, float]] = None
    submit_coords: Optional[Tuple[float, float]] = None
    input_coords: Optional[Tuple[float, float]] = None
    plan_content: Optional[PlanContent] = None
    roadmap_steps: List[RoadmapStep] = Field(default_factory=list)


class AntigravityDetector:
    """
    Detects Antigravity IDE state by analyzing screenshots with OCR.
    Uses pattern matching on recognized text elements to classify the current state.
    """

    # ── Text Patterns for State Detection ──

    # Indicators that AI is thinking/generating
    THINKING_PATTERNS = [
        "thinking", "generating", "analyzing", "processing",
        "searching", "reading", "writing", "editing",
        "running command", "executing", "creating",
        "viewing file", "semantic search", "searching the web",
        "planning", "researching",
    ]

    # Indicators that an implementation plan is visible
    PLAN_PATTERNS = [
        "implementation plan", "proposed changes", "verification plan",
        "user review required", "open questions",
        "automated tests", "manual verification",
    ]

    # The magic buttons we look for
    PROCEED_BUTTON_TEXTS = ["proceed", "approve"]
    SUBMIT_BUTTON_TEXTS = ["submit"]

    # Indicators of a roadmap / multi-step task list
    ROADMAP_PATTERNS = [
        "task tracker", "task list", "roadmap", "phase 1",
        "step 1", "component 1",
    ]

    # Indicators of completion
    DONE_PATTERNS = [
        "goal completed", "goal accomplished", "task completed",
        "all tasks complete", "done:", "finished",
        "walkthrough", "changes made", "what was tested",
        "validation results",
    ]

    # Indicators of errors
    ERROR_PATTERNS = [
        "error:", "failed:", "exception", "traceback",
        "syntax error", "cannot", "unable to",
        "build failed", "test failed", "lint error",
    ]

    # Indicators that input field is waiting
    INPUT_PATTERNS = [
        "type a message", "ask anything", "send a message",
        "what should i do", "enter your prompt",
        "how can i help",
    ]

    # Indicators that execution is in progress (tool calls, etc.)
    EXECUTING_PATTERNS = [
        "editing file", "creating file", "running command",
        "file edit", "command execution", "web search",
        "directory analysis", "viewing file",
    ]

    def __init__(self) -> None:
        self._last_state: AntigravityState = AntigravityState.UNKNOWN
        self._state_history: List[AntigravityState] = []
        self._consecutive_same_state: int = 0

    @property
    def last_state(self) -> AntigravityState:
        return self._last_state

    def detect_state(self, screenshot: Image.Image) -> DetectionResult:
        """
        Analyze a screenshot to determine Antigravity IDE's current state.
        Returns a DetectionResult with state, confidence, and any useful coordinates.
        """
        elements = ScreenGrounder.extract_screen_text_elements(screenshot)
        if not elements:
            return DetectionResult(
                state=AntigravityState.UNKNOWN,
                confidence=0.1,
                details="No text elements detected on screen",
            )

        # Build full-text corpus from all elements for pattern matching
        all_text_lower = " ".join(el.text.strip().lower() for el in elements)
        result = DetectionResult(elements_count=len(elements))

        # ── Phase 1: Look for specific buttons ──
        proceed_coords = self._find_button(screenshot, self.PROCEED_BUTTON_TEXTS)
        submit_coords = self._find_button(screenshot, self.SUBMIT_BUTTON_TEXTS)
        result.proceed_coords = proceed_coords
        result.submit_coords = submit_coords

        # ── Phase 2: Score each state ──
        scores: Dict[AntigravityState, float] = {s: 0.0 for s in AntigravityState}

        # Check for implementation plan
        plan_score = self._count_pattern_matches(all_text_lower, self.PLAN_PATTERNS)
        if plan_score >= 2:
            scores[AntigravityState.IMPLEMENTATION_PLAN] += 0.7
        elif plan_score == 1:
            scores[AntigravityState.IMPLEMENTATION_PLAN] += 0.3

        if proceed_coords:
            scores[AntigravityState.IMPLEMENTATION_PLAN] += 0.5

        # Check for roadmap
        roadmap_score = self._count_pattern_matches(all_text_lower, self.ROADMAP_PATTERNS)
        if roadmap_score >= 2:
            scores[AntigravityState.PLAN_WITH_ROADMAP] += 0.6
        # Numbered list detection (e.g. "1." "2." "3." in sequence)
        numbered_items = self._count_numbered_items(elements)
        if numbered_items >= 3:
            scores[AntigravityState.PLAN_WITH_ROADMAP] += 0.3

        # Check for thinking/generating
        thinking_score = self._count_pattern_matches(all_text_lower, self.THINKING_PATTERNS)
        if thinking_score >= 2:
            scores[AntigravityState.THINKING] += 0.5
        elif thinking_score == 1:
            scores[AntigravityState.THINKING] += 0.2

        # Spinner/animation detection: look for "..." or "…" or animated indicators
        if "..." in all_text_lower or "…" in all_text_lower:
            scores[AntigravityState.THINKING] += 0.15

        # Check for execution in progress
        exec_score = self._count_pattern_matches(all_text_lower, self.EXECUTING_PATTERNS)
        if exec_score >= 2:
            scores[AntigravityState.EXECUTING] += 0.6
        elif exec_score == 1:
            scores[AntigravityState.EXECUTING] += 0.25

        # Check for done/completion
        done_score = self._count_pattern_matches(all_text_lower, self.DONE_PATTERNS)
        if done_score >= 2:
            scores[AntigravityState.DONE] += 0.7
        elif done_score == 1:
            scores[AntigravityState.DONE] += 0.3

        # Check for walkthrough specifically
        if "walkthrough" in all_text_lower and ("changes made" in all_text_lower or "what was tested" in all_text_lower):
            scores[AntigravityState.WALKTHROUGH] += 0.8

        # Check for errors
        error_score = self._count_pattern_matches(all_text_lower, self.ERROR_PATTERNS)
        if error_score >= 2:
            scores[AntigravityState.ERROR] += 0.6
        elif error_score == 1:
            scores[AntigravityState.ERROR] += 0.2

        # Check for input waiting state
        input_score = self._count_pattern_matches(all_text_lower, self.INPUT_PATTERNS)
        if input_score >= 1:
            scores[AntigravityState.AWAITING_INPUT] += 0.3
        if submit_coords and not proceed_coords and thinking_score == 0:
            scores[AntigravityState.AWAITING_INPUT] += 0.2

        # Idle: no strong signals for any active state
        active_max = max(
            scores[AntigravityState.THINKING],
            scores[AntigravityState.EXECUTING],
            scores[AntigravityState.IMPLEMENTATION_PLAN],
            scores[AntigravityState.PLAN_WITH_ROADMAP],
        )
        if active_max < 0.2 and input_score >= 1:
            scores[AntigravityState.IDLE] += 0.5

        # ── Phase 3: Select best state ──
        best_state = max(scores, key=scores.get)
        best_score = scores[best_state]

        # If no state scored significantly, default to UNKNOWN
        if best_score < 0.2:
            best_state = AntigravityState.UNKNOWN
            best_score = 0.1

        result.state = best_state
        result.confidence = min(best_score, 1.0)
        result.details = self._build_details(best_state, scores, elements)

        # Extract plan content if applicable
        if best_state == AntigravityState.IMPLEMENTATION_PLAN:
            result.plan_content = self._extract_plan_content(elements)
            if result.plan_content:
                result.plan_content.proceed_coords = proceed_coords

        # Extract roadmap steps if applicable
        if best_state == AntigravityState.PLAN_WITH_ROADMAP:
            result.roadmap_steps = self._extract_roadmap_steps(elements)

        # Find input area coordinates
        result.input_coords = self._find_input_area(elements, screenshot)

        # Track state history
        self._update_history(best_state)

        return result

    def _find_button(
        self, screenshot: Image.Image, button_texts: List[str]
    ) -> Optional[Tuple[float, float]]:
        """Find a button by text on the screenshot."""
        for text in button_texts:
            coords = ScreenGrounder.find_element_by_text(screenshot, text)
            if coords:
                return coords
        return None

    def _count_pattern_matches(self, text: str, patterns: List[str]) -> int:
        """Count how many patterns match in the text."""
        return sum(1 for p in patterns if p in text)

    def _count_numbered_items(self, elements: List[TextElement]) -> int:
        """Count sequential numbered items (1. 2. 3. etc.) in screen elements."""
        count = 0
        for el in elements:
            stripped = el.text.strip()
            if re.match(r"^\d+[\.\)]\s", stripped):
                count += 1
            # Also check for checkbox-style items: [ ], [x], [/]
            if re.match(r"^\[[ x/]\]\s", stripped):
                count += 1
        return count

    def _build_details(
        self, state: AntigravityState, scores: Dict[AntigravityState, float],
        elements: List[TextElement],
    ) -> str:
        """Build a human-readable details string."""
        top_scores = sorted(
            [(s, sc) for s, sc in scores.items() if sc > 0],
            key=lambda x: x[1], reverse=True,
        )[:3]
        score_str = ", ".join(f"{s.value}={sc:.2f}" for s, sc in top_scores)
        return f"Detected: {state.value} ({len(elements)} elements). Scores: [{score_str}]"

    def _extract_plan_content(self, elements: List[TextElement]) -> Optional[PlanContent]:
        """Extract implementation plan content from OCR elements."""
        plan = PlanContent()
        capturing = False
        plan_lines: List[str] = []

        for el in elements:
            text = el.text.strip()
            lower = text.lower()

            if "implementation plan" in lower or "proposed changes" in lower:
                capturing = True
                plan.title = text

            if capturing:
                plan_lines.append(text)

                # Detect sections
                if any(p in lower for p in ["proposed changes", "verification plan", "user review", "open questions"]):
                    plan.sections.append(text)

            if lower in ("proceed", "approve"):
                plan.has_proceed_button = True

        plan.raw_text = "\n".join(plan_lines)
        return plan if plan.title or plan.has_proceed_button else None

    def _extract_roadmap_steps(self, elements: List[TextElement]) -> List[RoadmapStep]:
        """Parse numbered or checkboxed steps from the screen."""
        steps: List[RoadmapStep] = []
        idx = 0

        for el in elements:
            text = el.text.strip()

            # Match numbered items: "1. Do something" or "1) Do something"
            num_match = re.match(r"^(\d+)[\.\)]\s+(.+)", text)
            if num_match:
                idx += 1
                steps.append(RoadmapStep(
                    index=idx,
                    text=num_match.group(2).strip(),
                ))
                continue

            # Match checkbox items: "[ ] Do something", "[x] Done item", "[/] In progress"
            cb_match = re.match(r"^\[([ x/])\]\s+(.+)", text)
            if cb_match:
                idx += 1
                marker = cb_match.group(1)
                steps.append(RoadmapStep(
                    index=idx,
                    text=cb_match.group(2).strip(),
                    is_completed=(marker == "x"),
                ))
                continue

            # Match "Phase N:" or "Step N:" patterns
            phase_match = re.match(r"^(?:Phase|Step|Component)\s+(\d+)[:\-\.]\s*(.+)", text, re.IGNORECASE)
            if phase_match:
                idx += 1
                steps.append(RoadmapStep(
                    index=idx,
                    text=phase_match.group(2).strip(),
                ))

        return steps

    def _find_input_area(
        self, elements: List[TextElement], screenshot: Image.Image
    ) -> Optional[Tuple[float, float]]:
        """Locate the chat input area on screen."""
        # Look for common input placeholder texts
        for el in elements:
            lower = el.text.strip().lower()
            if any(p in lower for p in self.INPUT_PATTERNS):
                return (el.center_x, el.center_y)

        # Fallback: look for text input elements at the bottom of the screen
        # The input is typically in the bottom 15% of the screen
        img_h = screenshot.height
        bottom_threshold = img_h * 0.85
        bottom_elements = [el for el in elements if el.center_y > bottom_threshold]

        if bottom_elements:
            # Find the most central bottom element (likely the input)
            img_w = screenshot.width
            center_x = img_w / 2.0
            closest = min(bottom_elements, key=lambda el: abs(el.center_x - center_x))
            return (closest.center_x, closest.center_y)

        return None

    def _update_history(self, state: AntigravityState) -> None:
        """Track state transitions for debouncing and trend detection."""
        if state == self._last_state:
            self._consecutive_same_state += 1
        else:
            self._consecutive_same_state = 1

        self._last_state = state
        self._state_history.append(state)
        # Keep only last 50 states
        if len(self._state_history) > 50:
            self._state_history = self._state_history[-50:]

    def is_state_stable(self, min_consecutive: int = 2) -> bool:
        """Check if the current state has been consistent for N polls."""
        return self._consecutive_same_state >= min_consecutive

    def get_state_emoji(self, state: Optional[AntigravityState] = None) -> str:
        """Get a display emoji for the given state."""
        s = state or self._last_state
        return {
            AntigravityState.IDLE: "💤",
            AntigravityState.THINKING: "🧠",
            AntigravityState.IMPLEMENTATION_PLAN: "📋",
            AntigravityState.EXECUTING: "⚡",
            AntigravityState.PLAN_WITH_ROADMAP: "🗺️",
            AntigravityState.WALKTHROUGH: "📝",
            AntigravityState.AWAITING_INPUT: "✏️",
            AntigravityState.ERROR: "❌",
            AntigravityState.DONE: "✅",
            AntigravityState.UNKNOWN: "❓",
        }.get(s, "❓")

    def get_state_display(self, state: Optional[AntigravityState] = None) -> str:
        """Get a human-friendly display string for the state."""
        s = state or self._last_state
        emoji = self.get_state_emoji(s)
        return {
            AntigravityState.IDLE: f"{emoji} IDLE — Waiting for input",
            AntigravityState.THINKING: f"{emoji} THINKING — AI is generating...",
            AntigravityState.IMPLEMENTATION_PLAN: f"{emoji} PLAN READY — Implementation plan detected",
            AntigravityState.EXECUTING: f"{emoji} EXECUTING — Running tool calls...",
            AntigravityState.PLAN_WITH_ROADMAP: f"{emoji} ROADMAP — Multi-step plan detected",
            AntigravityState.WALKTHROUGH: f"{emoji} WALKTHROUGH — Summary ready",
            AntigravityState.AWAITING_INPUT: f"{emoji} AWAITING INPUT — Ready for prompt",
            AntigravityState.ERROR: f"{emoji} ERROR — Problem detected",
            AntigravityState.DONE: f"{emoji} DONE — Task completed",
            AntigravityState.UNKNOWN: f"{emoji} UNKNOWN — Cannot determine state",
        }.get(s, f"{emoji} {s.value.upper()}")
