"""Tests for Jarvis Browser Controller module."""

from unittest.mock import patch, MagicMock
from magnum.browser.controller import JarvisBrowserController, BrowserTab, get_browser_controller


def test_browser_controller_singleton():
    c1 = get_browser_controller()
    c2 = get_browser_controller()
    assert c1 is c2
    assert isinstance(c1, JarvisBrowserController)


def test_search_engine_url_construction():
    ctrl = JarvisBrowserController()
    assert "google.com/search?q=machine+learning" in ctrl.SEARCH_ENGINES["google"].format(query="machine+learning")
    assert "youtube.com/results?search_query=lofi" in ctrl.SEARCH_ENGINES["youtube"].format(query="lofi")
    assert "github.com/search?q=fastapi" in ctrl.SEARCH_ENGINES["github"].format(query="fastapi")


def test_switch_to_tab_fuzzy_matching():
    ctrl = JarvisBrowserController()
    dummy_tabs = [
        BrowserTab(window_id=1, tab_index=1, title="GitHub - openai/whisper", url="https://github.com/openai/whisper"),
        BrowserTab(window_id=1, tab_index=2, title="YouTube - Lo-Fi Beats", url="https://youtube.com/watch?v=123"),
        BrowserTab(window_id=1, tab_index=3, title="Google Search", url="https://google.com"),
    ]

    with patch.object(ctrl, "list_tabs", return_value=dummy_tabs), \
         patch("subprocess.run") as mock_sub:
        # Match by keyword
        matched = ctrl.switch_to_tab("youtube")
        assert matched is not None
        assert matched.tab_index == 2
        assert "YouTube" in matched.title

        # Match by fuzzy title
        matched_gh = ctrl.switch_to_tab("whisper")
        assert matched_gh is not None
        assert matched_gh.tab_index == 1


def test_scroll_js_generation():
    ctrl = JarvisBrowserController()
    with patch.object(ctrl, "execute_js", return_value="OK") as mock_js:
        res = ctrl.scroll(direction="down", amount=500)
        assert res is True
        assert mock_js.called
        assert "scrollBy" in mock_js.call_args[0][0]

        res_bottom = ctrl.scroll(direction="bottom")
        assert res_bottom is True
        assert "scrollHeight" in mock_js.call_args[0][0]


def test_navigate_actions():
    ctrl = JarvisBrowserController()
    with patch.object(ctrl, "execute_js", return_value="OK") as mock_js:
        assert ctrl.navigate("reload") is True
        assert "reload" in mock_js.call_args[0][0]


def test_click_and_type_by_id():
    ctrl = JarvisBrowserController()
    with patch.object(ctrl, "execute_js", return_value="CLICKED_ID: 5") as mock_js:
        assert ctrl.click_element_by_id(5) is True
        assert "data-magnum-id=\"5\"" in mock_js.call_args[0][0]

    with patch.object(ctrl, "execute_js", return_value="TYPED_ID: 3") as mock_js:
        assert ctrl.type_element_by_id(3, "Delhi") is True
        assert "Delhi" in mock_js.call_args[0][0]


def test_select_dropdown_and_checkbox():
    ctrl = JarvisBrowserController()
    with patch.object(ctrl, "execute_js", return_value="SELECTED_OPTION: Business Class") as mock_js:
        assert ctrl.select_dropdown_option(2, "Business Class") is True
        assert "business class" in mock_js.call_args[0][0].lower()

    with patch.object(ctrl, "execute_js", return_value="CHECKBOX_SET: true") as mock_js:
        assert ctrl.set_checkbox(4, checked=True) is True
        assert "true" in mock_js.call_args[0][0]


def test_scroll_element_and_batch_fill():
    ctrl = JarvisBrowserController()
    with patch.object(ctrl, "execute_js", return_value="SCROLLED_INTO_VIEW") as mock_js:
        assert ctrl.scroll_element_into_view(7) is True
        assert "data-magnum-id=\"7\"" in mock_js.call_args[0][0]

    with patch.object(ctrl, "execute_js", return_value="FILLED_COUNT: 3") as mock_js:
        count = ctrl.batch_fill_form({"from": "NYC", "to": "LAX", "passengers": "2"})
        assert count == 3
