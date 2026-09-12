"""
Unit tests for Native macOS Accessibility Driver (AXUIElement).
"""

import pytest
from magnum.driver.macos_ax import MacOSAccessibilityDriver, NativeAXElement


def test_macos_ax_availability():
    # In macOS environment, PyObjC ApplicationServices should be available
    assert MacOSAccessibilityDriver.is_available() is True


def test_macos_ax_get_frontmost_app():
    app_info = MacOSAccessibilityDriver.get_frontmost_app()
    assert app_info is not None
    name, pid, bundle_id = app_info
    assert len(name) > 0
    assert pid > 0


def test_macos_ax_find_app():
    # Finder is always running on macOS
    finder_info = MacOSAccessibilityDriver.find_app_by_name("Finder")
    assert finder_info is not None
    name, pid, bundle_id = finder_info
    assert "finder" in name.lower()
    assert pid > 0


def test_macos_ax_extract_elements_from_frontmost():
    app_info = MacOSAccessibilityDriver.get_frontmost_app()
    assert app_info is not None
    _, pid, _ = app_info

    elements = MacOSAccessibilityDriver.extract_ax_elements(pid, max_elements=50)
    assert isinstance(elements, list)
    if elements:
        el = elements[0]
        assert isinstance(el, NativeAXElement)
        assert el.width >= 0
        assert el.height >= 0
        assert el.center_x >= 0
        assert el.center_y >= 0


def test_native_ax_element_properties():
    el = NativeAXElement(
        id=1,
        role="button",
        raw_role="AXButton",
        label="Save Document",
        bounds=(100.0, 50.0, 200.0, 90.0),
        center_x=150.0,
        center_y=70.0,
    )
    assert el.width == 100.0
    assert el.height == 40.0
    assert el.role == "button"
