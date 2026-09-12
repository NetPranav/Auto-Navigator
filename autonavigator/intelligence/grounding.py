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
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

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
        cls, image: Image.Image, query: str
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

        # 1. Exact match (case-insensitive)
        for el in elements:
            if el.text.strip().lower() == clean_query:
                return (el.center_x, el.center_y)

        # 2. Substring match
        for el in elements:
            el_text = el.text.strip().lower()
            if clean_query in el_text or el_text in clean_query:
                return (el.center_x, el.center_y)

        # 3. Fuzzy similarity match
        texts = [el.text.strip() for el in elements]
        matches = difflib.get_close_matches(query, texts, n=1, cutoff=0.55)
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

    @staticmethod
    def annotate_set_of_marks(
        image: Image.Image,
        boxes: List[BoundingBox],
    ) -> Image.Image:
        """
        Draw Set-of-Marks (SoM) colored numbered badges on candidate UI elements.
        This provides unambiguous numerical IDs for the LLM to select.
        """
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        # Palette of vibrant colors for UI boxes
        colors = [
            "#FF0055", "#00E5FF", "#76FF03", "#FFD600",
            "#D500F9", "#FF6D00", "#00B0FF", "#00E676"
        ]

        for box in boxes:
            color = colors[box.id % len(colors)]
            draw.rectangle(
                [(box.left, box.top), (box.right, box.bottom)],
                outline=color,
                width=2,
            )

            tag_text = f" {box.id} "
            badge_width = len(tag_text) * 10 + 6
            badge_height = 18
            draw.rectangle(
                [(box.left, box.top - badge_height), (box.left + badge_width, box.top)],
                fill=color,
            )
            draw.text(
                (box.left + 2, box.top - badge_height + 2),
                tag_text,
                fill="#000000",
            )

        return annotated
