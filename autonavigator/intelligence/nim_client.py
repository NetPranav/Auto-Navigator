"""
NVIDIA NIM API Client for Dynamic Planning and OCR-First Step-by-Step Grounding.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from pydantic import BaseModel, Field

from autonavigator.config import config
from autonavigator.intelligence.grounding import ScreenGrounder, TextElement
from autonavigator.intelligence.prompts import (
    DYNAMIC_PLANNER_PROMPT,
    SCREEN_PERCEPTION_PROMPT,
    ACTION_DECISION_PROMPT,
    COMMENT_SYNTHESIS_PROMPT,
)
from autonavigator.intelligence.planner import PlanChecklist, PlanStep

logger = logging.getLogger(__name__)


class GroundedAction(BaseModel):
    """Structured action output by the reasoning model."""

    thought: str = Field(default="", description="Reasoning behind this action")
    screen_description: Optional[str] = Field(
        default=None, description="OCR element manifest or vision description of screen"
    )
    action: str = Field(
        default="STEP_DONE",
        description="Action type: OPEN_APP, NAVIGATE, SEARCH, CLICK, DOUBLE_CLICK, RIGHT_CLICK, TYPE, PRESS_KEY, HOTKEY, SCROLL, WAIT, OBSTACLE_DETECTED, REPORT, ASK_USER, STEP_DONE, FINISH",
    )
    coordinates: Optional[Dict[str, float]] = Field(
        default=None, description="Coordinates {'x': float, 'y': float}"
    )
    text: Optional[str] = Field(
        default=None, description="Text to type / URL to open / app name / element text to click"
    )
    search_element: Optional[str] = Field(
        default=None, description="Text of search bar element to click before searching"
    )
    key: Optional[str] = Field(default=None, description="Key to press (e.g. Enter, Tab)")
    hotkeys: Optional[List[str]] = Field(default=None, description="Key combinations e.g. ['command', 'space']")
    scroll_direction: Optional[str] = Field(
        default="down", description="Direction: up or down"
    )
    scroll_amount: Optional[int] = Field(default=300, description="Amount in pixels to scroll")
    options: Optional[List[str]] = Field(
        default=None, description="Options for interactive question card"
    )
    is_terminal: bool = Field(
        default=False, description="True if goal has been accomplished"
    )


class LinkedInCommentDraft(BaseModel):
    """Synthesized comment and rationale."""

    draft_comment: str
    rationale: str
    post_summary: Optional[str] = None
    post_author: Optional[str] = None


class NimClient:
    """Client for querying NVIDIA NIM models for OCR-first vision grounding and dynamic reasoning."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        vision_model: Optional[str] = None,
        reasoning_model: Optional[str] = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else config.nvidia_api_key
        self.base_url = base_url or config.nvidia_base_url
        self.vision_model = vision_model or config.default_vision_model
        self.reasoning_model = reasoning_model or config.default_reasoning_model
        self._client = None

    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            if not self.api_key:
                logger.warning(
                    "NVIDIA_API_KEY is not set. Real API calls will fail unless configured."
                )
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key or "placeholder_key",
            )
        return self._client

    @staticmethod
    def pil_to_base64(image: Image.Image) -> str:
        """Convert a PIL Image to a base64 PNG data string."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    def format_ocr_manifest(elements: List[TextElement]) -> str:
        """Format OCR elements into a structured manifest string for the reasoning model."""
        if not elements:
            return "  (No text elements detected on screen)"
        
        lines = []
        for i, el in enumerate(elements, start=1):
            # Truncate very long text to keep manifest readable
            text = el.text.strip()
            if len(text) > 60:
                text = text[:57] + "..."
            lines.append(f"  [{i}] \"{text}\" at ({el.center_x:.0f}, {el.center_y:.0f})")
        
        return "\n".join(lines)

    def call_vision(
        self,
        prompt: str,
        image_base64: str,
        system_prompt: str,
        temperature: float = 0.2,
    ) -> str:
        """Call NVIDIA NIM vision model with image and text prompt with retry logic."""
        if not self.api_key:
            return json.dumps({
                "thought": "[Simulation Mode] Simulating screen action",
                "action": "STEP_DONE",
                "coordinates": None,
                "text": None,
                "is_terminal": False,
            })

        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=self.vision_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_base64}"
                                    },
                                },
                            ],
                        },
                    ],
                    temperature=temperature,
                    max_tokens=1024,
                    timeout=30.0,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"Vision call (attempt {attempt+1}) failed: {e}")
                if attempt == 0:
                    import time
                    time.sleep(1.0)

        return json.dumps({
            "thought": "Vision fallback step completion",
            "action": "STEP_DONE",
            "coordinates": None,
            "text": None,
            "is_terminal": False,
        })

    def call_reasoning(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.3,
    ) -> str:
        """Call NVIDIA NIM text reasoning model with retry logic."""
        if not self.api_key:
            return json.dumps({
                "goal_summary": "Task",
                "steps": [
                    {"step_index": 1, "title": "Open Application", "description": "Open target application"},
                    {"step_index": 2, "title": "Execute Goal", "description": "Perform task"},
                ],
            })

        candidates = [self.reasoning_model, "openai/gpt-oss-120b", "openai/gpt-oss-20b", "meta/llama-3.2-11b-vision-instruct"]
        seen = set()
        for model_cand in candidates:
            if model_cand in seen:
                continue
            seen.add(model_cand)
            try:
                response = self.client.chat.completions.create(
                    model=model_cand,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=1024,
                    timeout=20.0,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"Reasoning call with {model_cand} failed: {e}. Trying next candidate...")

        return ""

    def parse_json_response(self, text: str) -> Dict[str, Any]:
        """Extract and parse JSON from model response text with robust fallback."""
        clean = text.strip()
        if "```json" in clean:
            clean = clean.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in clean:
            clean = clean.split("```", 1)[1].split("```", 1)[0].strip()

        # 1. Direct JSON parse
        try:
            parsed = json.loads(clean)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # 2. Extract balanced JSON object from within text
        start_idx = clean.find("{")
        while start_idx != -1:
            depth = 0
            end_idx = -1
            in_str = False
            escape = False
            for i in range(start_idx, len(clean)):
                c = clean[i]
                if escape:
                    escape = False
                    continue
                if c == "\\":
                    escape = True
                    continue
                if c == '"':
                    in_str = not in_str
                    continue
                if not in_str:
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            end_idx = i
                            break
            if end_idx != -1:
                try:
                    parsed = json.loads(clean[start_idx : end_idx + 1])
                    if isinstance(parsed, dict) and ("action" in parsed or "steps" in parsed or "goal_summary" in parsed):
                        return parsed
                except Exception:
                    pass
            start_idx = clean.find("{", start_idx + 1)

        # Smart Heuristic Parser for GroundedAction from freeform text
        return self._extract_action_heuristic(text)

    def _extract_action_heuristic(self, raw_text: str) -> Dict[str, Any]:
        """Heuristically extract action, coordinates, and text from unstructured output."""
        lower = raw_text.lower()
        
        # Check action keywords
        action = "WAIT"
        options = None
        if (
            "sign in" in lower
            or "login" in lower
            or "verify it's you" in lower
            or "verify" in lower
            or "enter password" in lower
            or "confirmidentifier" in lower
            or "two-factor" in lower
            or "2fa" in lower
            or "captcha" in lower
        ):
            action = "OBSTACLE_DETECTED"
            text_val = "Authentication / Google Identity verification required on screen. Please sign in to continue."
        elif "search" in lower or "search bar" in lower or "find messages" in lower:
            action = "SEARCH"
            q_match = re.search(r"(?:search for|query|find messages from|into the search bar)\s+[\"']?([^\"'\n\.,]+)[\"']?", raw_text, re.IGNORECASE)
            text_val = q_match.group(1).strip() if q_match else "search query"
        elif "open browser" in lower or "open chrome" in lower or "launch browser" in lower:
            action = "OPEN_APP"
            text_val = "Google Chrome"
        elif "navigate to" in lower or "http" in lower or "mail.google.com" in lower or "linkedin.com" in lower or "github.com" in lower:
            action = "NAVIGATE"
            url_match = re.search(r"(https?://[^\s\"']+|mail\.google\.com|linkedin\.com|github\.com[^\s\"']*)", raw_text)
            text_val = url_match.group(1) if url_match else "https://mail.google.com"
        elif "click and type" in lower:
            action = "CLICK_AND_TYPE"
            t_match = re.search(r"type\s+[\"']([^\"']+)[\"']", raw_text, re.IGNORECASE)
            text_val = t_match.group(1) if t_match else ""
        elif "click" in lower:
            action = "CLICK"
            # Try to extract what to click
            t_match = re.search(r"click\s+(?:on\s+)?[\"']([^\"']+)[\"']", raw_text, re.IGNORECASE)
            text_val = t_match.group(1) if t_match else None
        elif "type" in lower:
            action = "TYPE"
            t_match = re.search(r"type\s+[\"']([^\"']+)[\"']", raw_text, re.IGNORECASE)
            text_val = t_match.group(1) if t_match else ""
        elif "read" in lower or "report" in lower or "summary" in lower or "findings" in lower:
            action = "REPORT"
            text_val = raw_text
            options = ["Got it", "Done"]
        elif "ask" in lower or "confirm" in lower:
            action = "ASK_USER"
            text_val = raw_text
            options = ["Approve & Post", "Cancel"]
        elif "done" in lower or "completed" in lower or "next step" in lower:
            action = "STEP_DONE"
            text_val = None
        elif "finish" in lower:
            action = "FINISH"
            text_val = None
        else:
            text_val = None

        # Look for coordinates
        coords = None
        c_match = re.search(r"[\"']?x[\"']?\s*:\s*(\d+(?:\.\d+)?)\s*,\s*[\"']?y[\"']?\s*:\s*(\d+(?:\.\d+)?)", raw_text)
        if c_match:
            coords = {"x": float(c_match.group(1)), "y": float(c_match.group(2))}

        return {
            "thought": raw_text[:200].replace("\n", " ").strip(),
            "action": action,
            "coordinates": coords,
            "text": text_val,
        }

    def generate_plan(self, user_goal: str) -> PlanChecklist:
        """Dynamically generate a step-by-step plan checklist from user prompt."""
        raw = self.call_reasoning(
            prompt=f"User Task: {user_goal}",
            system_prompt=DYNAMIC_PLANNER_PROMPT,
        )
        try:
            data = self.parse_json_response(raw)
            steps_data = data.get("steps", [])
            if not steps_data and isinstance(data, list):
                steps_data = data
            
            steps = []
            for i, s in enumerate(steps_data, start=1):
                if isinstance(s, dict):
                    title = s.get("title") or f"Step {i}"
                    desc = s.get("description") or title
                    steps.append(PlanStep(step_index=i, title=title, description=desc))
                elif isinstance(s, str):
                    steps.append(PlanStep(step_index=i, title=s, description=s))

            if steps:
                steps[0].status = "active"
                return PlanChecklist(
                    goal_summary=data.get("goal_summary", user_goal),
                    steps=steps,
                    active_index=1,
                )
        except Exception as e:
            logger.warning(f"Plan parse fallback: {e}")

        # Smart fallback plan based on instruction
        lower = user_goal.lower()
        if "gmail" in lower:
            steps = [
                PlanStep(step_index=1, title="Open Browser", description="Open Chrome or switch to active browser", status="active"),
                PlanStep(step_index=2, title="Navigate to Gmail", description="Open https://mail.google.com"),
                PlanStep(step_index=3, title="Check Emails", description="View unread emails or search"),
            ]
        elif "linkedin" in lower:
            steps = [
                PlanStep(step_index=1, title="Open Browser", description="Open Chrome or switch to active browser", status="active"),
                PlanStep(step_index=2, title="Navigate to LinkedIn", description="Open https://www.linkedin.com"),
                PlanStep(step_index=3, title="Locate Post & Read Comments", description="Find target user post"),
                PlanStep(step_index=4, title="Draft & Confirm Comment", description="Propose comment for 1-click user approval"),
            ]
        else:
            steps = [
                PlanStep(step_index=1, title="Open Target Application", description="Launch or switch to target app", status="active"),
                PlanStep(step_index=2, title="Perform Action", description=user_goal),
                PlanStep(step_index=3, title="Verify & Finish", description="Confirm completion"),
            ]

        return PlanChecklist(goal_summary=user_goal, steps=steps, active_index=1)

    def describe_screen(self, screenshot_base64: str) -> str:
        """Stage 1 Fallback: Factual screen scene & application perception using vision model.
        Only used when OCR returns zero elements (purely graphical screens)."""
        prompt = (
            "Analyze this computer screenshot and state clearly:\n"
            "1. What app/window is visible in the foreground? (e.g. Antigravity IDE / VS Code, Google Chrome, Desktop, etc.)\n"
            "2. If a browser is open: what website/URL is loaded? (e.g. YouTube, Gmail, New Tab, or not open)\n"
            "3. Are there any search bars, buttons, or login/verification obstacles on screen?"
        )
        raw = self.call_vision(
            prompt=prompt,
            image_base64=screenshot_base64,
            system_prompt=SCREEN_PERCEPTION_PROMPT,
        )
        # Clean up response text
        desc = raw.strip().replace("```", "").strip()
        return desc if desc else "Screen captured (unable to parse perception details)."

    def ground_step_action(
        self,
        screenshot_base64: str,
        goal: str,
        current_step: PlanStep,
        total_steps: int,
        history: Optional[List[str]] = None,
        ocr_elements: Optional[List[TextElement]] = None,
    ) -> GroundedAction:
        """OCR-First Grounding: 1) Build OCR element manifest -> 2) Reasoning Model decides action."""
        
        # 1. Build structured screen element manifest from OCR
        if ocr_elements is not None:
            screen_manifest = self.format_ocr_manifest(ocr_elements)
        else:
            screen_manifest = "  (OCR data not available)"
        
        element_count = len(ocr_elements) if ocr_elements else 0
        logger.info(f"OCR Screen Manifest: {element_count} elements detected")

        # 2. If zero OCR elements, fall back to vision model for description
        if element_count == 0:
            vision_desc = self.describe_screen(screenshot_base64)
            screen_manifest = f"  (No OCR text detected. Vision model says: {vision_desc})"

        # 3. Reasoning Model decides the exact OS action based on the element manifest
        sys_prompt = ACTION_DECISION_PROMPT.format(
            goal=goal,
            step_index=current_step.step_index,
            total_steps=total_steps,
            current_step_title=current_step.title,
            current_step_description=current_step.description,
            screen_elements=screen_manifest,
            history="\n".join(f"- {h}" for h in (history or [])[-5:]) if history else "None",
        )

        user_prompt = (
            f"Goal: {goal}\n"
            f"Current Step: {current_step.title} - {current_step.description}\n"
            f"Screen Elements ({element_count} detected):\n{screen_manifest}\n\n"
            f"What is the exact next JSON action?"
        )

        raw_decision = self.call_reasoning(
            prompt=user_prompt,
            system_prompt=sys_prompt,
        )

        try:
            data = self.parse_json_response(raw_decision)
            if "action" not in data:
                action_inferred = "CLICK" if data.get("x") or data.get("coordinates") else "STEP_DONE"
                data["action"] = action_inferred
                data["thought"] = data.get("thought", "Inferred action")

            action_obj = GroundedAction(**data)
            action_obj.screen_description = f"[{element_count} OCR elements]"
            return action_obj
        except Exception as e:
            logger.warning(f"Reasoning decision fallback: {e}\nRaw: {raw_decision}")
            # Fallback based on step context
            step_lower = current_step.title.lower()
            manifest_lower = screen_manifest.lower()
            
            if ("chrome" not in manifest_lower and "browser" not in manifest_lower) and ("open" in step_lower):
                return GroundedAction(
                    thought="No browser elements detected on screen. Launching Google Chrome.",
                    action="OPEN_APP",
                    text="Google Chrome",
                    screen_description=f"[{element_count} OCR elements]",
                )
            elif "youtube" in step_lower and "youtube" not in manifest_lower:
                return GroundedAction(
                    thought="YouTube not detected in screen elements. Navigating to YouTube.",
                    action="NAVIGATE",
                    text="https://www.youtube.com",
                    screen_description=f"[{element_count} OCR elements]",
                )
            elif "gmail" in step_lower and "gmail" not in manifest_lower and "mail" not in manifest_lower:
                return GroundedAction(
                    thought="Gmail not detected in screen elements. Navigating to Gmail.",
                    action="NAVIGATE",
                    text="https://mail.google.com",
                    screen_description=f"[{element_count} OCR elements]",
                )
            else:
                return GroundedAction(
                    thought=f"Proceeding based on {element_count} screen elements.",
                    action="STEP_DONE",
                    screen_description=f"[{element_count} OCR elements]",
                )

    def synthesize_comment(
        self,
        post_context: str,
        author: Optional[str] = None,
        image_base64: Optional[str] = None,
    ) -> LinkedInCommentDraft:
        """Draft a professional comment using NIM."""
        prompt = f"Author: {author or 'User'}\nPost Content:\n{post_context}\nDraft a thoughtful comment."
        if image_base64:
            raw = self.call_vision(
                prompt=prompt,
                image_base64=image_base64,
                system_prompt=COMMENT_SYNTHESIS_PROMPT,
            )
        else:
            raw = self.call_reasoning(
                prompt=prompt,
                system_prompt=COMMENT_SYNTHESIS_PROMPT,
            )
        try:
            data = self.parse_json_response(raw)
            return LinkedInCommentDraft(**data)
        except Exception:
            return LinkedInCommentDraft(
                draft_comment="Great insights! Thanks for sharing this update.",
                rationale="Fallback comment",
            )
