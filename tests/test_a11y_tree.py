"""
Unit tests for Astra-grade Accessibility Tree, Set-of-Marks annotation, and deterministic element ID dispatch.
"""

import pytest
from PIL import Image
from magnum.intelligence.a11y_tree import A11yElement, A11yEngine, A11yTree, BrowserA11yExtractor
from magnum.intelligence.grounding import ScreenGrounder, TextElement
from magnum.intelligence.nim_client import GroundedAction, NimClient


def test_a11y_element_compact_notation():
    el = A11yElement(
        id=1,
        role="button",
        label="Search Flights",
        bounds=(100.0, 200.0, 250.0, 240.0),
        center_x=175.0,
        center_y=220.0,
    )
    assert el.width == 150.0
    assert el.height == 40.0
    s = el.to_compact_string()
    assert '[1] BUTTON "Search Flights"' in s
    assert "at (175, 220)" in s


def test_a11y_tree_lookup_and_prompt():
    el1 = A11yElement(
        id=1,
        role="textbox",
        label="From",
        value="New York",
        bounds=(50.0, 100.0, 200.0, 140.0),
        center_x=125.0,
        center_y=120.0,
    )
    el2 = A11yElement(
        id=2,
        role="button",
        label="Search",
        bounds=(300.0, 100.0, 400.0, 140.0),
        center_x=350.0,
        center_y=120.0,
    )

    tree = A11yTree(elements=[el1, el2], active_app="Google Chrome")
    assert tree.get_element_by_id(1) == el1
    assert tree.get_element_by_id(2) == el2
    assert tree.get_element_by_id(99) is None

    # Text search
    assert tree.find_by_text("Search") == el2
    assert tree.find_by_text("from") == el1

    prompt_txt = tree.to_compact_prompt_text()
    assert '[1] TEXTBOX "From"' in prompt_txt
    assert '[2] BUTTON "Search"' in prompt_txt


def test_a11y_engine_with_ocr_fallback():
    img = Image.new("RGB", (800, 600), color="white")
    ocr_items = [
        TextElement(text="Google Chrome", left=10, top=10, right=100, bottom=30, center_x=55, center_y=20),
        TextElement(text="Submit Form", left=200, top=300, right=300, bottom=340, center_x=250, center_y=320),
        TextElement(text="Enter Name", left=200, top=200, right=300, bottom=240, center_x=250, center_y=220),
    ]

    from unittest.mock import patch
    with patch("magnum.driver.macos_ax.MacOSAccessibilityDriver.find_app_by_name", return_value=None), \
         patch("magnum.driver.macos_ax.MacOSAccessibilityDriver.get_frontmost_app", return_value=None):
        tree = A11yEngine.build_tree(
            image=img,
            active_app="MockNonExistentAppXYZ",
            browser_controller=None,
            ocr_elements=ocr_items,
        )

    assert len(tree.elements) == 3
    # Check that roles are heuristically mapped
    assert tree.elements[1].role == "button"
    assert tree.elements[2].role == "textbox"


def test_annotate_a11y_tree_som():
    img = Image.new("RGB", (800, 600), color="white")
    el1 = A11yElement(id=1, role="button", label="OK", bounds=(50, 50, 150, 90), center_x=100, center_y=70)
    el2 = A11yElement(id=2, role="button", label="Cancel", bounds=(200, 50, 300, 90), center_x=250, center_y=70)
    tree = A11yTree(elements=[el1, el2])

    annotated = ScreenGrounder.annotate_a11y_tree(img, tree)
    assert annotated.size == img.size


def test_grounded_action_target_id():
    act = GroundedAction(
        thought="Click the search button",
        action="CLICK",
        target_id=4,
    )
    assert act.target_id == 4
    assert act.action == "CLICK"


def test_ground_step_action_with_a11y_resolution():
    client = NimClient(api_key="mock_key")
    # Test JSON parsing with target_id
    raw_json = '{"thought": "Click search button", "action": "CLICK", "target_id": 2}'
    parsed = client.parse_json_response(raw_json)
    assert parsed.get("target_id") == 2
    assert parsed.get("action") == "CLICK"

    # Test heuristic fallback with element ID in text
    heur = client._extract_action_heuristic("I will click element [3] on the screen")
    assert heur["action"] == "CLICK"
    assert heur["target_id"] == 3
