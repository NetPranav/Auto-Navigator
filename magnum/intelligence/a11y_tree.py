"""
Astra-Grade Accessibility Tree (a11y) and Interactive Element Extractor.
Implements the core dual-modal observation pipeline pioneered by ChatGPT Astra (GPT-6 Astra):
extracts semantic interactive element trees (buttons, inputs, comboboxes, links) from browsers
and native OS apps, assigning deterministic IDs for zero-hallucination execution.
"""

from __future__ import annotations

import difflib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class A11yElement:
    """A single semantic interactive or content element in the a11y tree."""

    id: int
    role: str  # button, link, textbox, searchbox, combobox, checkbox, tab, menuitem, dialog
    label: str  # Visible text, aria-label, placeholder, or title
    bounds: Tuple[float, float, float, float]  # (left, top, right, bottom)
    center_x: float
    center_y: float
    value: Optional[str] = None
    placeholder: Optional[str] = None
    is_interactive: bool = True
    is_focused: bool = False
    is_disabled: bool = False
    source: str = "browser_dom"  # "browser_dom" | "apple_ocr" | "macos_ax"
    selector: Optional[str] = None  # CSS selector or native identifier
    native_element: Optional[Any] = None  # Native macOS AXUIElement reference

    @property
    def width(self) -> float:
        return max(0.0, self.bounds[2] - self.bounds[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bounds[3] - self.bounds[1])

    def to_compact_string(self) -> str:
        """Format element in Astra-style compact notation for LLM prompt."""
        parts = [f"[{self.id}] {self.role.upper()} \"{self.label}\""]
        if self.value and self.value != self.label:
            val_clean = self.value[:35] + ("..." if len(self.value) > 35 else "")
            parts.append(f"value=\"{val_clean}\"")
        elif self.placeholder and not self.value:
            parts.append(f"placeholder=\"{self.placeholder}\"")
        if self.is_focused:
            parts.append("[FOCUSED]")
        if self.is_disabled:
            parts.append("[DISABLED]")
        parts.append(f"at ({self.center_x:.0f}, {self.center_y:.0f})")
        return " ".join(parts)


@dataclass
class A11yTree:
    """Collection of interactive elements on screen with lookup and prompt serialization."""

    elements: List[A11yElement] = field(default_factory=list)
    active_app: str = "Desktop"
    url: Optional[str] = None
    title: Optional[str] = None

    def get_element_by_id(self, elem_id: int) -> Optional[A11yElement]:
        """Find element by its integer ID."""
        for el in self.elements:
            if el.id == elem_id:
                return el
        return None

    def find_by_text(self, text_query: str, role: Optional[str] = None) -> Optional[A11yElement]:
        """Find closest element by text query and optional role filter."""
        clean_q = text_query.strip().lower()
        candidates = self.elements
        if role:
            candidates = [el for el in candidates if el.role.lower() == role.lower()]

        # 1. Exact match
        for el in candidates:
            if el.label.strip().lower() == clean_q:
                return el

        # 2. Substring match
        for el in candidates:
            el_lbl = el.label.strip().lower()
            if clean_q in el_lbl or el_lbl in clean_q:
                return el

        # 3. Fuzzy match
        lbls = [el.label.strip() for el in candidates if el.label.strip()]
        matches = difflib.get_close_matches(text_query, lbls, n=1, cutoff=0.7)
        if matches:
            best_lbl = matches[0]
            for el in candidates:
                if el.label.strip() == best_lbl:
                    return el

        return None

    def to_compact_prompt_text(self, max_elements: int = 50) -> str:
        """
        Serialize elements into Astra's compact a11y catalog.
        Categorizes interactive elements clearly so the reasoning model can select target_id.
        """
        if not self.elements:
            return "(No interactive elements detected on screen)"

        lines: List[str] = []
        for el in self.elements[:max_elements]:
            lines.append(el.to_compact_string())

        if len(self.elements) > max_elements:
            lines.append(f"... and {len(self.elements) - max_elements} additional elements.")

        return "\n".join(lines)


class BrowserA11yExtractor:
    """
    Extracts semantic accessibility tree and interactive elements directly from
    Google Chrome or Apple Safari via high-speed JavaScript DOM inspection (<30ms).
    """

    # JavaScript payload injected to query interactive elements and compute page coordinates
    EXTRACTION_JS = """
    (function() {
        var elements = [];
        var idCounter = 1;

        // Clean existing tags
        var existing = document.querySelectorAll('[data-magnum-id]');
        for (var i = 0; i < existing.length; i++) {
            existing[i].removeAttribute('data-magnum-id');
        }

        // Query all interactive and semantic elements
        var selectors = [
            'button',
            'a[href]',
            'input:not([type="hidden"])',
            'textarea',
            'select',
            '[role="button"]',
            '[role="link"]',
            '[role="tab"]',
            '[role="menuitem"]',
            '[role="checkbox"]',
            '[role="radio"]',
            '[role="combobox"]',
            '[role="searchbox"]',
            '[contenteditable="true"]',
            '[tabindex]:not([tabindex="-1"])'
        ];

        var nodes = Array.from(document.querySelectorAll(selectors.join(',')));

        for (var i = 0; i < nodes.length; i++) {
            var node = nodes[i];

            // Filter invisible or zero-size elements
            var rect = node.getBoundingClientRect();
            if (rect.width <= 2 || rect.height <= 2) continue;
            if (rect.bottom < 0 || rect.top > window.innerHeight) continue;
            if (rect.right < 0 || rect.left > window.innerWidth) continue;

            var style = window.getComputedStyle(node);
            if (style.display === 'none' || style.visibility === 'hidden' || parseFloat(style.opacity) < 0.1) continue;

            // Determine role
            var role = node.getAttribute('role') || node.tagName.toLowerCase();
            if (node.tagName.toLowerCase() === 'input') {
                var type = (node.getAttribute('type') || 'text').toLowerCase();
                if (type === 'submit' || type === 'button') role = 'button';
                else if (type === 'checkbox') role = 'checkbox';
                else if (type === 'radio') role = 'radio';
                else if (type === 'search') role = 'searchbox';
                else role = 'textbox';
            } else if (node.tagName.toLowerCase() === 'a') {
                role = 'link';
            } else if (node.tagName.toLowerCase() === 'select') {
                role = 'combobox';
            }

            // Extract accessible label
            var label = '';
            if (node.getAttribute('aria-label')) {
                label = node.getAttribute('aria-label');
            } else if (node.getAttribute('placeholder')) {
                label = node.getAttribute('placeholder');
            } else if (node.getAttribute('title')) {
                label = node.getAttribute('title');
            } else if (node.innerText && node.innerText.trim().length > 0) {
                label = node.innerText.trim();
            } else if (node.value && typeof node.value === 'string' && node.value.trim().length > 0) {
                label = node.value.trim();
            } else if (node.getAttribute('alt')) {
                label = node.getAttribute('alt');
            }

            // Clean multi-line strings
            label = label.replace(/\\s+/g, ' ').trim();
            if (label.length > 70) {
                label = label.substring(0, 67) + '...';
            }

            if (!label && role !== 'textbox' && role !== 'searchbox') {
                continue;
            }

            var currId = idCounter++;
            node.setAttribute('data-magnum-id', currId.toString());

            var val = node.value ? node.value.toString().trim() : null;
            var placeholder = node.getAttribute('placeholder') || null;
            var isFocused = (document.activeElement === node);
            var isDisabled = Boolean(node.disabled || node.getAttribute('aria-disabled') === 'true');

            elements.push({
                id: currId,
                role: role,
                label: label || (placeholder ? ('Input: ' + placeholder) : 'Input field'),
                value: val,
                placeholder: placeholder,
                left: Math.round(rect.left),
                top: Math.round(rect.top),
                right: Math.round(rect.right),
                bottom: Math.round(rect.bottom),
                is_focused: isFocused,
                is_disabled: isDisabled
            });

            if (elements.length >= 60) break;
        }

        return JSON.stringify(elements);
    })();
    """

    @classmethod
    def extract(cls, browser_controller) -> List[A11yElement]:
        """Extract a11y elements from the current frontmost browser tab."""
        raw_json = browser_controller.execute_js(cls.EXTRACTION_JS)
        if not raw_json:
            return []

        try:
            data = json.loads(raw_json)
            elements: List[A11yElement] = []

            for item in data:
                left = float(item["left"])
                top = float(item["top"])
                right = float(item["right"])
                bottom = float(item["bottom"])
                cx = (left + right) / 2.0
                cy = (top + bottom) / 2.0

                el = A11yElement(
                    id=item["id"],
                    role=item["role"],
                    label=item["label"],
                    bounds=(left, top, right, bottom),
                    center_x=cx,
                    center_y=cy,
                    value=item.get("value"),
                    placeholder=item.get("placeholder"),
                    is_focused=bool(item.get("is_focused")),
                    is_disabled=bool(item.get("is_disabled")),
                    source="browser_dom",
                    selector=f'[data-magnum-id="{item["id"]}"]',
                )
                elements.append(el)

            return elements
        except Exception as e:
            logger.debug(f"Failed to parse browser a11y JSON: {e}")
            return []


class A11yEngine:
    """
    Unified Astra-grade Accessibility & Grounding Engine.
    Produces rich a11y trees combining browser DOM inspection and Apple Vision OCR.
    """

    @classmethod
    def build_tree(
        cls,
        image: Image.Image,
        active_app: str,
        browser_controller=None,
        ocr_elements: Optional[List[Any]] = None,
    ) -> A11yTree:
        """
        Build unified a11y tree for the current screen.
        If active_app is Chrome or Safari, uses browser DOM a11y extraction.
        Otherwise, converts Apple Vision OCR elements into the a11y tree.
        """
        tree = A11yTree(active_app=active_app)
        elements: List[A11yElement] = []

        is_browser = any(b in active_app.lower() for b in ("chrome", "safari", "brave", "arc", "edge", "firefox"))

        # ── TIER 1: BROWSER DOM INSPECTION ──
        if is_browser and browser_controller:
            try:
                browser_elements = BrowserA11yExtractor.extract(browser_controller)
                if browser_elements:
                    elements.extend(browser_elements)
            except Exception as e:
                logger.debug(f"Browser a11y extraction failed: {e}")

        # ── TIER 2: NATIVE MACOS AXUIELEMENT INSPECTION ──
        if not is_browser or not elements:
            try:
                from magnum.driver.macos_ax import MacOSAccessibilityDriver
                if MacOSAccessibilityDriver.is_available():
                    app_info = MacOSAccessibilityDriver.find_app_by_name(active_app) or MacOSAccessibilityDriver.get_frontmost_app()
                    if app_info:
                        ax_nodes = MacOSAccessibilityDriver.extract_ax_elements(app_info[1])
                        for ax in ax_nodes:
                            elements.append(
                                A11yElement(
                                    id=len(elements) + 1,
                                    role=ax.role,
                                    label=ax.label,
                                    bounds=ax.bounds,
                                    center_x=ax.center_x,
                                    center_y=ax.center_y,
                                    value=ax.value,
                                    source="macos_ax",
                                    native_element=ax.raw_element,
                                )
                            )
            except Exception as e:
                logger.debug(f"Native macOS AX extraction failed: {e}")

        # ── TIER 3: VISUAL APPLE VISION OCR SUPPLEMENT ──
        if ocr_elements:
            # Add OCR elements if they don't strongly overlap with existing Tier 1/2 elements
            existing_bounds = [el.bounds for el in elements]
            for ocr_el in ocr_elements:
                # Check overlap with existing elements
                overlaps = False
                for eb in existing_bounds:
                    if abs(ocr_el.center_x - (eb[0] + eb[2]) / 2.0) < 30 and abs(ocr_el.center_y - (eb[1] + eb[3]) / 2.0) < 15:
                        overlaps = True
                        break
                if overlaps:
                    continue

                txt = ocr_el.text.strip()
                lower = txt.lower()
                role = "content"
                if any(b in lower for b in ("button", "submit", "login", "sign in", "search", "ok", "cancel", "next", "continue", "done", "save")):
                    role = "button"
                elif "http" in lower or ".com" in lower or ".org" in lower:
                    role = "link"
                elif any(inp in lower for inp in ("enter", "type", "search...", "username", "password", "email")):
                    role = "textbox"

                elements.append(
                    A11yElement(
                        id=len(elements) + 1,
                        role=role,
                        label=txt,
                        bounds=(ocr_el.left, ocr_el.top, ocr_el.right, ocr_el.bottom),
                        center_x=ocr_el.center_x,
                        center_y=ocr_el.center_y,
                        source="apple_ocr",
                    )
                )

        # Normalize sequential IDs from 1..N
        for i, el in enumerate(elements, start=1):
            el.id = i

        tree.elements = elements
        return tree
