"""
Screen and UI coordinate grounding, Apple Vision OCR text element detection,
Set-of-Marks (SoM) tagging, and sub-pixel resolution scaling.
"""

from __future__ import annotations

import difflib
import io
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from magnum.intelligence.a11y_tree import A11yElement, A11yEngine, A11yTree

logger = logging.getLogger(__name__)


class NormalizedPoint(BaseModel):
    """Point coordinate normalized or in target screen dimensions."""

    x: float
    y: float


class BoundingBox(BaseModel):
    """Bounding box for an interactive element."""

    id: int
    label: str
    left: float
    top: float
    right: float
    bottom: float

    @property
    def center(self) -> NormalizedPoint:
        return NormalizedPoint(
            x=(self.left + self.right) / 2.0,
            y=(self.top + self.bottom) / 2.0,
        )


class TextElement(BaseModel):
    """Exact on-screen text element recognized by OCR."""

    text: str
    left: float
    top: float
    right: float
    bottom: float
    center_x: float
    center_y: float
    confidence: float = 1.0


class ScreenGrounder:
    """Handles coordinate translation, Apple Vision OCR element detection, and Set-of-Marks overlays."""

    @staticmethod
    def get_mac_retina_scale() -> float:
        """Detect macOS Retina scaling factor (defaults to 2.0 on Mac Retina)."""
        try:
            import Quartz
            main_display = Quartz.CGMainDisplayID()
            mode = Quartz.CGDisplayCopyDisplayMode(main_display)
            pixel_width = Quartz.CGDisplayModeGetPixelWidth(mode)
            point_width = Quartz.CGDisplayPixelsWide(main_display)
            if point_width > 0:
                return float(pixel_width) / float(point_width)
        except Exception:
            pass
        return 1.0

    @staticmethod
    def scale_point(
        point: Dict[str, float],
        source_width: float,
        source_height: float,
        target_width: float,
        target_height: float,
    ) -> NormalizedPoint:
        """Translate coordinates from screenshot dimensions to target display/viewport dimensions."""
        if source_width <= 0 or source_height <= 0:
            return NormalizedPoint(x=point["x"], y=point["y"])

        scale_x = target_width / source_width
        scale_y = target_height / source_height

        return NormalizedPoint(
            x=round(point["x"] * scale_x, 2),
            y=round(point["y"] * scale_y, 2),
        )

    @classmethod
    def extract_screen_text_elements(cls, image: Image.Image) -> List[TextElement]:
        """
        Extract all on-screen text elements and bounding box centers using Apple Vision OCR.
        Coordinates are returned in logical macOS points (accounting for Retina 2x scale).
        """
        elements: List[TextElement] = []
        retina_scale = cls.get_mac_retina_scale()
        img_w, img_h = image.size

        # Temporary file for Apple Vision request
        tmp_path = os.path.join(tempfile.gettempdir(), "_autonav_ocr_temp.png")
        try:
            image.save(tmp_path, format="PNG")

            import Cocoa
            import Vision

            img_url = Cocoa.NSURL.fileURLWithPath_(tmp_path)
            request_handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(img_url, {})

            request = Vision.VNRecognizeTextRequest.alloc().init()
            request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
            request.setUsesLanguageCorrection_(True)

            success = request_handler.performRequests_error_([request], None)
            if not success or not request.results():
                return elements

            for r in request.results():
                top_cand = r.topCandidates_(1)
                if not top_cand:
                    continue
                cand_str = top_cand[0].string().strip()
                if not cand_str:
                    continue

                # Filter out the HUD overlay window itself so agent never targets its own UI
                upper_cand = cand_str.upper()
                if "AI PLAN" in upper_cand or "CHECKLIST" in upper_cand or "AUTO-NAVIGATOR ACTIVE" in upper_cand:
                    continue

                # Filter out agent's own terminal logs so it never clicks its own printed debug output
                if any(cand_str.startswith(prefix) for prefix in (
                    "🎯", "⚡", "🔍", "🧠", "⌨️", "🔔", "🎉", "✓ Action", "✓ Step", "Target:", "Action:", "OCR:", "Step ", "Watching for:"
                )):
                    continue

                if any(k in cand_str for k in ("Auto-Navigator %", "magnum %", "(.venv)", "pranav@")):
                    continue

                bbox = r.boundingBox()
                # Apple Vision normalized coords: origin at bottom-left (0.0 to 1.0)
                # Convert to pixel coords with top-left origin:
                raw_x = bbox.origin.x * img_w
                raw_y = (1.0 - (bbox.origin.y + bbox.size.height)) * img_h
                raw_w = bbox.size.width * img_w
                raw_h = bbox.size.height * img_h

                # Scale to logical macOS points
                left = raw_x / retina_scale
                top = raw_y / retina_scale
                right = (raw_x + raw_w) / retina_scale
                bottom = (raw_y + raw_h) / retina_scale
                center_x = (left + right) / 2.0
                center_y = (top + bottom) / 2.0

                # Filter out elements physically located inside the HUD floating checklist card (top-left: x<340, y: 25..280)
                if left < 340 and 25 < top < 280:
                    # Ignore checklist step icons and lines
                    if any(cand_str.startswith(p) for p in ("•", "v ", "4 ", "| ", "› ", "O ", "V ", "4 1", "• 2", "• 3", "v 1")):
                        continue

                elements.append(
                    TextElement(
                        text=cand_str,
                        left=round(left, 1),
                        top=round(top, 1),
                        right=round(right, 1),
                        bottom=round(bottom, 1),
                        center_x=round(center_x, 1),
                        center_y=round(center_y, 1),
                        confidence=float(top_cand[0].confidence()),
                    )
                )
        except Exception as e:
            logger.debug(f"Apple Vision OCR extraction error: {e}")
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        return elements

    @classmethod
    def find_element_by_text(
        cls, image: Image.Image, query: str, region: Optional[str] = None
    ) -> Optional[Tuple[float, float]]:
        """
        Locate the exact logical center coordinates (x, y) of a target text query on screen.
        Uses exact match, substring match, and fuzzy similarity matching.
        """
        if not query:
            return None

        elements = cls.extract_screen_text_elements(image)
        if not elements:
            return None

        clean_query = query.strip().lower()

        # 1. System menu bar exclusion: NEVER match interactive UI buttons in the macOS top bar (y <= 35)
        elements = [el for el in elements if el.center_y > 35]

        # 2. Exclude terminal area when searching for interactive action buttons
        if clean_query in ("submit", "proceed", "continue", "confirm", "approve", "done", "next"):
            non_terminal = [el for el in elements if not (150 <= el.center_x <= 850 and 450 <= el.center_y <= 1050)]
            if non_terminal:
                elements = non_terminal

        # 3. Apply spatial region constraint if specified (e.g. "bottom_right")
        w, h = image.size
        scale = 2.0 if w > 1920 else 1.0
        sw = w / scale
        sh = h / scale

        if region == "bottom_right":
            elements = [el for el in elements if el.center_x >= sw * 0.45 and el.center_y >= sh * 0.45]
        elif region == "bottom_left":
            elements = [el for el in elements if el.center_x <= sw * 0.55 and el.center_y >= sh * 0.45]
        elif region == "top_right":
            elements = [el for el in elements if el.center_x >= sw * 0.45 and el.center_y <= sh * 0.55]
        elif region == "top_left":
            elements = [el for el in elements if el.center_x <= sw * 0.55 and el.center_y <= sh * 0.55]
        elif region == "bottom":
            elements = [el for el in elements if el.center_y >= sh * 0.50]
        elif region == "top":
            elements = [el for el in elements if el.center_y <= sh * 0.50]
        elif region == "right":
            elements = [el for el in elements if el.center_x >= sw * 0.50]
        elif region == "left":
            elements = [el for el in elements if el.center_x <= sw * 0.50]

        if not elements:
            return None

        # 4. Check learned visual anchors if matching element is near anchor
        from magnum.intelligence.memory import get_memory
        anchor = get_memory().get_visual_anchor(clean_query)
        if anchor:
            for el in elements:
                el_lower = el.text.strip().lower()
                if (clean_query in el_lower or el_lower in clean_query) and abs(el.center_x - anchor.x) < 250 and abs(el.center_y - anchor.y) < 180:
                    return (el.center_x, el.center_y)

        # 5. Smart semantic aliases for specialized UI elements
        if any(w in clean_query for w in ("ai agent text box", "agent text box", "ai text box", "agent input", "chat box")):
            for el in elements:
                el_lower = el.text.strip().lower()
                if "ask anything" in el_lower or "@ to mention" in el_lower:
                    return (el.center_x, el.center_y)

        # 6. Specialized matching for "submit":
        # Specifically match "Submit", "Send", or send icon "→" / "↑", NEVER "Proceed"!
        if clean_query == "submit":
            for el in elements:
                el_lower = el.text.strip().lower()
                if el_lower in ("submit", "send", "→", "↑") or el_lower.startswith("submit"):
                    return (el.center_x, el.center_y)

        # 7. Specialized matching for "proceed":
        if clean_query in ("proceed", "proceed with plan", "proceed to plan"):
            for el in elements:
                el_lower = el.text.strip().lower()
                if "proceed" in el_lower:
                    return (el.center_x, el.center_y)

        # 8. Exact match (case-insensitive)
        for el in elements:
            if el.text.strip().lower() == clean_query:
                return (el.center_x, el.center_y)

        # 9. Substring match
        for el in elements:
            el_text = el.text.strip().lower()
            if clean_query in el_text or el_text in clean_query:
                if len(clean_query) >= 3 and len(el_text) < 40:
                    return (el.center_x, el.center_y)

        # 10. Strict fuzzy similarity match (cutoff 0.82 to prevent false positives)
        texts = [el.text.strip() for el in elements if len(el.text.strip()) >= 3]
        matches = difflib.get_close_matches(query, texts, n=1, cutoff=0.82)
        if matches:
            best_match = matches[0]
            for el in elements:
                if el.text.strip() == best_match:
                    return (el.center_x, el.center_y)

        return None

    @classmethod
    def find_candidates_by_text(
        cls, image: Image.Image, query: str
    ) -> List[Tuple[str, float, float]]:
        """Find all matching text candidates on screen with their names and coordinates."""
        elements = cls.extract_screen_text_elements(image)
        clean_query = query.strip().lower()
        candidates: List[Tuple[str, float, float]] = []

        for el in elements:
            el_text = el.text.strip().lower()
            if (
                clean_query in el_text
                or el_text in clean_query
                or difflib.SequenceMatcher(None, clean_query, el_text).ratio() > 0.5
            ):
                candidates.append((el.text.strip(), el.center_x, el.center_y))

        return candidates

    @classmethod
    def annotate_set_of_marks(
        cls,
        image: Image.Image,
        boxes: Union[List[BoundingBox], List[A11yElement], List[Any]],
    ) -> Image.Image:
        """
        Draw Set-of-Marks (SoM) colored numbered badges on candidate UI elements.
        Supports both BoundingBox, A11yElement, and duck-typed UI markers.
        Automatically scales logical points to screenshot pixel resolution.
        """
        if not boxes:
            return image

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        # Palette of vibrant high-contrast colors
        colors = [
            "#FF0055", "#00E5FF", "#76FF03", "#FFD600",
            "#D500F9", "#FF6D00", "#00B0FF", "#00E676"
        ]

        retina_scale = cls.get_mac_retina_scale()
        img_w, _ = image.size
        # If coordinates are in logical points (< 1920) but image is Retina (e.g. 2880 or 3840), scale up
        scale = retina_scale if (img_w > 1920 and retina_scale > 1.0) else 1.0

        for box in boxes:
            box_id = getattr(box, "id", 1)
            # Extract coordinates
            if hasattr(box, "bounds"):
                b_left, b_top, b_right, b_bottom = box.bounds
            else:
                b_left = getattr(box, "left", 0.0)
                b_top = getattr(box, "top", 0.0)
                b_right = getattr(box, "right", 0.0)
                b_bottom = getattr(box, "bottom", 0.0)

            left = b_left * scale
            top = b_top * scale
            right = b_right * scale
            bottom = b_bottom * scale

            color = colors[box_id % len(colors)]

            # Draw element outline
            draw.rectangle(
                [(left, top), (right, bottom)],
                outline=color,
                width=max(2, int(2 * scale)),
            )

            # Draw Astra-style numbered pill badge
            tag_text = f" {box_id} "
            badge_width = len(tag_text) * int(9 * scale) + int(6 * scale)
            badge_height = int(18 * scale)
            badge_top = max(0, top - badge_height)

            draw.rectangle(
                [(left, badge_top), (left + badge_width, top)],
                fill=color,
            )
            draw.text(
                (left + int(2 * scale), badge_top + int(2 * scale)),
                tag_text,
                fill="#000000",
            )

        return annotated

    @classmethod
    def annotate_a11y_tree(
        cls,
        image: Image.Image,
        tree: A11yTree,
    ) -> Image.Image:
        """Annotate screenshot with Astra-style Set-of-Marks badges for all a11y tree elements."""
        return cls.annotate_set_of_marks(image, tree.elements)

