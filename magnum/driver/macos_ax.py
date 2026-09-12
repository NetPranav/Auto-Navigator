"""
Native macOS Accessibility Driver for Auto-Navigator (Magnum).
Direct programmatic UI inspection and non-intrusive action dispatch via macOS Accessibility
(AXUIElement) and AppKit. Directly references architectures from open-codex-computer-use and trycua.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try importing macOS native accessibility frameworks
try:
    import ApplicationServices as AS
    import AppKit
    HAS_MACOS_AX = True
except ImportError:
    HAS_MACOS_AX = False
    logger.warning("PyObjC ApplicationServices/AppKit not available. Native macOS AX driver disabled.")


@dataclass
class NativeAXElement:
    """A native macOS UI element queried directly from the OS Accessibility tree."""

    id: int
    role: str  # e.g. "button", "textfield", "checkbox", "popupbutton", "link"
    raw_role: str  # e.g. "AXButton", "AXTextField"
    label: str  # Title, description, or help string
    bounds: Tuple[float, float, float, float]  # (left, top, right, bottom)
    center_x: float
    center_y: float
    value: Optional[str] = None
    is_enabled: bool = True
    raw_element: Any = None  # Native AXUIElement reference

    @property
    def width(self) -> float:
        return max(0.0, self.bounds[2] - self.bounds[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bounds[3] - self.bounds[1])


class MacOSAccessibilityDriver:
    """Native macOS Accessibility automation driver using AXUIElement."""

    # Mapping of raw AX roles to standardized role names
    ROLE_MAP = {
        "AXButton": "button",
        "AXTextField": "textbox",
        "AXTextArea": "textbox",
        "AXCheckBox": "checkbox",
        "AXRadioButton": "radio",
        "AXPopUpButton": "combobox",
        "AXComboBox": "combobox",
        "AXMenuItem": "menuitem",
        "AXLink": "link",
        "AXTabGroup": "tab",
        "AXRadioButtonGroup": "radiogroup",
        "AXStaticText": "text",
        "AXImage": "image",
    }

    @classmethod
    def is_available(cls) -> bool:
        """Check if native macOS accessibility is supported in this environment."""
        return HAS_MACOS_AX

    @classmethod
    def get_frontmost_app(cls) -> Optional[Tuple[str, int, str]]:
        """Return (name, pid, bundle_id) of the active frontmost macOS application."""
        if not HAS_MACOS_AX:
            return None
        try:
            workspace = AppKit.NSWorkspace.sharedWorkspace()
            app = workspace.frontmostApplication()
            if app:
                name = str(app.localizedName() or "")
                pid = int(app.processIdentifier())
                bundle_id = str(app.bundleIdentifier() or "")
                return (name, pid, bundle_id)
        except Exception as e:
            logger.debug(f"Error getting frontmost app: {e}")
        return None

    @classmethod
    def find_app_by_name(cls, query_name: str) -> Optional[Tuple[str, int, str]]:
        """Find running macOS application matching name."""
        if not HAS_MACOS_AX:
            return None
        try:
            workspace = AppKit.NSWorkspace.sharedWorkspace()
            clean_query = query_name.lower().strip()
            for app in workspace.runningApplications():
                loc_name = (app.localizedName() or "").lower()
                bundle_id = (app.bundleIdentifier() or "").lower()
                if clean_query == loc_name or clean_query in loc_name or clean_query in bundle_id:
                    return (str(app.localizedName()), int(app.processIdentifier()), str(app.bundleIdentifier() or ""))
        except Exception as e:
            logger.debug(f"Error finding app '{query_name}': {e}")
        return None

    @classmethod
    def extract_ax_elements(cls, pid: int, max_elements: int = 120) -> List[NativeAXElement]:
        """
        Recursively extract interactive UI elements from an application's focused window.
        Returns elements with exact screen coordinates and native AXUIElement handles.
        """
        if not HAS_MACOS_AX:
            return []

        elements: List[NativeAXElement] = []
        try:
            app_ref = AS.AXUIElementCreateApplication(pid)

            # Get focused window or main window
            err, window = AS.AXUIElementCopyAttributeValue(app_ref, AS.kAXFocusedWindowAttribute, None)
            if err != 0 or not window:
                err, windows = AS.AXUIElementCopyAttributeValue(app_ref, AS.kAXWindowsAttribute, None)
                if windows and len(windows) > 0:
                    window = windows[0]

            if not window:
                return []

            counter = [1]
            visited = set()

            def _traverse(node: Any, depth: int = 0):
                if depth > 12 or len(elements) >= max_elements:
                    return

                # Prevent cycle loops
                node_hash = hash(node)
                if node_hash in visited:
                    return
                visited.add(node_hash)

                err_role, raw_role = AS.AXUIElementCopyAttributeValue(node, AS.kAXRoleAttribute, None)
                if err_role == 0 and raw_role:
                    role_str = str(raw_role)
                    # Check if interactive or interesting
                    if role_str in cls.ROLE_MAP or role_str.startswith("AX"):
                        # Extract title or description
                        label = ""
                        err_title, title_val = AS.AXUIElementCopyAttributeValue(node, AS.kAXTitleAttribute, None)
                        if err_title == 0 and title_val:
                            label = str(title_val).strip()

                        if not label:
                            err_desc, desc_val = AS.AXUIElementCopyAttributeValue(node, AS.kAXDescriptionAttribute, None)
                            if err_desc == 0 and desc_val:
                                label = str(desc_val).strip()

                        if not label:
                            err_help, help_val = AS.AXUIElementCopyAttributeValue(node, AS.kAXHelpAttribute, None)
                            if err_help == 0 and help_val:
                                label = str(help_val).strip()

                        # Extract value (for text fields)
                        value_str = None
                        err_val, val_obj = AS.AXUIElementCopyAttributeValue(node, AS.kAXValueAttribute, None)
                        if err_val == 0 and val_obj is not None:
                            value_str = str(val_obj).strip()

                        # Extract position and size
                        err_pos, pos_val = AS.AXUIElementCopyAttributeValue(node, AS.kAXPositionAttribute, None)
                        err_sz, sz_val = AS.AXUIElementCopyAttributeValue(node, AS.kAXSizeAttribute, None)

                        if err_pos == 0 and err_sz == 0 and pos_val and sz_val:
                            ok_pt, pt = AS.AXValueGetValue(pos_val, AS.kAXValueCGPointType, None)
                            ok_sz, sz = AS.AXValueGetValue(sz_val, AS.kAXValueCGSizeType, None)

                            if ok_pt and ok_sz and sz.width > 2 and sz.height > 2:
                                left = float(pt.x)
                                top = float(pt.y)
                                right = left + float(sz.width)
                                bottom = top + float(sz.height)
                                cx = (left + right) / 2.0
                                cy = (top + bottom) / 2.0

                                # Only include elements with labels or inputs
                                role_clean = cls.ROLE_MAP.get(role_str, "content")
                                is_interactive = role_clean in ("button", "textbox", "checkbox", "combobox", "link", "menuitem")

                                if label or is_interactive:
                                    if not label:
                                        label = value_str or f"{role_clean.title()} control"

                                    elements.append(
                                        NativeAXElement(
                                            id=counter[0],
                                            role=role_clean,
                                            raw_role=role_str,
                                            label=label[:70],
                                            bounds=(left, top, right, bottom),
                                            center_x=cx,
                                            center_y=cy,
                                            value=value_str,
                                            raw_element=node,
                                        )
                                    )
                                    counter[0] += 1

                # Traverse children
                err_ch, children = AS.AXUIElementCopyAttributeValue(node, AS.kAXChildrenAttribute, None)
                if err_ch == 0 and children:
                    for child in children:
                        _traverse(child, depth + 1)

            _traverse(window)

        except Exception as e:
            logger.debug(f"Native AX tree extraction error for PID {pid}: {e}")

        return elements

    @classmethod
    def press_ax_element(cls, native_element: Any) -> bool:
        """
        Directly invoke the native action (kAXPressAction) on an AXUIElement.
        Clicks the native button/control without moving or hijacking the system mouse cursor!
        """
        if not HAS_MACOS_AX or not native_element:
            return False
        try:
            err = AS.AXUIElementPerformAction(native_element, AS.kAXPressAction)
            if err == 0:
                logger.info("✓ Direct AXPressAction succeeded.")
                return True
            # Try AXPick or AXConfirm
            for fallback_action in ("AXPick", "AXConfirm", "AXShowMenu"):
                err2 = AS.AXUIElementPerformAction(native_element, fallback_action)
                if err2 == 0:
                    logger.info(f"✓ Direct {fallback_action} succeeded.")
                    return True
        except Exception as e:
            logger.debug(f"Direct AXPressAction failed: {e}")
        return False

    @classmethod
    def set_ax_element_value(cls, native_element: Any, text: str) -> bool:
        """
        Directly set the text value of a native text field via AXUIElement.
        Instantly populates inputs without requiring keystroke typing delays!
        """
        if not HAS_MACOS_AX or not native_element:
            return False
        try:
            err = AS.AXUIElementSetAttributeValue(native_element, AS.kAXValueAttribute, text)
            if err == 0:
                logger.info(f"✓ Direct AX set value succeeded for '{text[:20]}'")
                return True
        except Exception as e:
            logger.debug(f"Direct AX set value failed: {e}")
        return False
