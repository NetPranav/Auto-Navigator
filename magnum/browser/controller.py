"""
Jarvis Seamless Browser Controller for Auto-Navigator (Magnum).
Direct programmatic control over Google Chrome and Apple Safari on macOS via native AppleScript
and JavaScript injection. Provides instant tab management, DOM extraction, smart page summarization,
scrolling, and form interaction without relying on screen OCR.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import subprocess
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BrowserTab:
    window_id: int
    tab_index: int
    title: str
    url: str
    browser: str = "Google Chrome"


class JarvisBrowserController:
    """Native macOS browser automation engine for Google Chrome and Safari."""

    SEARCH_ENGINES = {
        "google": "https://www.google.com/search?q={query}",
        "youtube": "https://www.youtube.com/results?search_query={query}",
        "github": "https://github.com/search?q={query}&type=repositories",
        "reddit": "https://www.reddit.com/search/?q={query}",
        "amazon": "https://www.amazon.com/s?k={query}",
        "wikipedia": "https://en.wikipedia.org/wiki/Special:Search?search={query}",
        "duckduckgo": "https://duckduckgo.com/?q={query}",
    }

    def __init__(self, preferred_browser: str = "Google Chrome") -> None:
        self.preferred_browser = preferred_browser

    def detect_active_browser(self) -> str:
        """Detect which browser is currently frontmost or running."""
        script = """
        tell application "System Events"
            set frontProc to name of first application process whose frontmost is true
            if frontProc is "Google Chrome" or frontProc is "Safari" then
                return frontProc
            end if
            set isChrome to exists (processes where name is "Google Chrome")
            if isChrome then
                return "Google Chrome"
            end if
            set isSafari to exists (processes where name is "Safari")
            if isSafari then
                return "Safari"
            end if
            return "Google Chrome"
        end tell
        """
        try:
            res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=False)
            out = res.stdout.strip()
            return out if out in ("Google Chrome", "Safari") else self.preferred_browser
        except Exception:
            return self.preferred_browser

    # ── URL & Search Navigation ──

    def open_url(self, url: str, new_tab: bool = False, browser: Optional[str] = None) -> bool:
        """Open or navigate to a URL in the browser."""
        b = browser or self.detect_active_browser()
        clean_url = url.strip()
        if not clean_url.startswith("http://") and not clean_url.startswith("https://") and not clean_url.startswith("file://"):
            clean_url = f"https://{clean_url}"

        if b == "Google Chrome":
            if new_tab:
                script = f'''
                tell application "Google Chrome"
                    activate
                    if (count of windows) is 0 then
                        make new window
                        set URL of active tab of front window to "{clean_url}"
                    else
                        tell front window to make new tab with properties {{URL:"{clean_url}"}}
                    end if
                end tell
                '''
            else:
                script = f'''
                tell application "Google Chrome"
                    activate
                    if (count of windows) is 0 then
                        make new window
                    end if
                    set URL of active tab of front window to "{clean_url}"
                end tell
                '''
        else:  # Safari
            if new_tab:
                script = f'''
                tell application "Safari"
                    activate
                    if (count of windows) is 0 then
                        make new document with properties {{URL:"{clean_url}"}}
                    else
                        tell front window
                            set current tab to (make new tab with properties {{URL:"{clean_url}"}})
                        end tell
                    end if
                end tell
                '''
            else:
                script = f'''
                tell application "Safari"
                    activate
                    if (count of windows) is 0 then
                        make new document with properties {{URL:"{clean_url}"}}
                    else
                        set URL of current tab of front window to "{clean_url}"
                    end if
                end tell
                '''

        res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=False)
        return res.returncode == 0

    def search(self, query: str, engine: str = "google", new_tab: bool = False) -> bool:
        """Search query using the specified engine."""
        eng = engine.lower().strip()
        template = self.SEARCH_ENGINES.get(eng, self.SEARCH_ENGINES["google"])
        encoded_query = urllib.parse.quote_plus(query.strip())
        target_url = template.format(query=encoded_query)
        return self.open_url(target_url, new_tab=new_tab)

    # ── Tab Inspection & Management ──

    def list_tabs(self, browser: Optional[str] = None) -> List[BrowserTab]:
        """List all open tabs in the browser."""
        b = browser or self.detect_active_browser()
        tabs: List[BrowserTab] = []

        if b == "Google Chrome":
            script = """
            set outText to ""
            tell application "Google Chrome"
                if (count of windows) > 0 then
                    repeat with wIdx from 1 to count of windows
                        set tabList to tabs of window wIdx
                        repeat with tIdx from 1 to count of tabList
                            set t to item tIdx of tabList
                            set outText to outText & wIdx & "\t" & tIdx & "\t" & (title of t) & "\t" & (URL of t) & "\n"
                        end repeat
                    end repeat
                end if
            end tell
            return outText
            """
        else:  # Safari
            script = """
            set outText to ""
            tell application "Safari"
                if (count of windows) > 0 then
                    repeat with wIdx from 1 to count of windows
                        set tabList to tabs of window wIdx
                        repeat with tIdx from 1 to count of tabList
                            set t to item tIdx of tabList
                            set outText to outText & wIdx & "\t" & tIdx & "\t" & (name of t) & "\t" & (URL of t) & "\n"
                        end repeat
                    end repeat
                end if
            end tell
            return outText
            """

        res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=False)
        if res.returncode == 0 and res.stdout.strip():
            for line in res.stdout.strip().split("\n"):
                parts = line.split("\t")
                if len(parts) >= 4:
                    try:
                        tabs.append(BrowserTab(
                            window_id=int(parts[0]),
                            tab_index=int(parts[1]),
                            title=parts[2].strip(),
                            url=parts[3].strip(),
                            browser=b,
                        ))
                    except ValueError:
                        continue
        return tabs

    def switch_to_tab(self, query: str, browser: Optional[str] = None) -> Optional[BrowserTab]:
        """Switch to a tab by title, domain, or fuzzy name."""
        b = browser or self.detect_active_browser()
        tabs = self.list_tabs(b)
        if not tabs:
            return None

        clean_q = query.lower().strip()
        matched_tab: Optional[BrowserTab] = None

        # 1. Exact or substring match in title or URL
        for t in tabs:
            if clean_q in t.title.lower() or clean_q in t.url.lower():
                matched_tab = t
                break

        # 2. Fuzzy match on titles
        if not matched_tab:
            titles = [t.title for t in tabs]
            matches = difflib.get_close_matches(query, titles, n=1, cutoff=0.45)
            if matches:
                matched_tab = next(t for t in tabs if t.title == matches[0])

        if not matched_tab:
            return None

        # Execute switch
        if b == "Google Chrome":
            script = f'''
            tell application "Google Chrome"
                activate
                set active tab index of window {matched_tab.window_id} to {matched_tab.tab_index}
                set index of window {matched_tab.window_id} to 1
            end tell
            '''
        else:  # Safari
            script = f'''
            tell application "Safari"
                activate
                set current tab of window {matched_tab.window_id} to tab {matched_tab.tab_index} of window {matched_tab.window_id}
                set index of window {matched_tab.window_id} to 1
            end tell
            '''

        subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=False)
        return matched_tab

    def close_tab(self, query: Optional[str] = None, browser: Optional[str] = None) -> bool:
        """Close the active tab or a matched tab."""
        b = browser or self.detect_active_browser()
        if query:
            tab = self.switch_to_tab(query, b)
            if not tab:
                return False

        if b == "Google Chrome":
            script = """
            tell application "Google Chrome"
                if (count of windows) > 0 then
                    tell front window to close active tab
                    return "OK"
                end if
            end tell
            """
        else:
            script = """
            tell application "Safari"
                if (count of windows) > 0 then
                    tell front window to close current tab
                    return "OK"
                end if
            end tell
            """
        res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=False)
        return "OK" in res.stdout

    # ── Page Scrolling & Navigation ──

    def scroll(self, direction: str = "down", amount: int = 600, browser: Optional[str] = None) -> bool:
        """Smoothly scroll the active webpage."""
        b = browser or self.detect_active_browser()
        delta = amount if direction.lower() in ("down", "bottom") else -amount

        if direction.lower() == "bottom":
            js = "window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'}); 'OK'"
        elif direction.lower() == "top":
            js = "window.scrollTo({top: 0, behavior: 'smooth'}); 'OK'"
        else:
            js = f"window.scrollBy({{top: {delta}, left: 0, behavior: 'smooth'}}); 'OK'"

        return self.execute_js(js, b) is not None

    def navigate(self, action: str, browser: Optional[str] = None) -> bool:
        """Navigate: back, forward, reload, zoom_in, zoom_out."""
        b = browser or self.detect_active_browser()
        act = action.lower().strip()

        if act in ("reload", "refresh"):
            js = "location.reload(); 'OK'"
            return self.execute_js(js, b) is not None
        elif act in ("back", "previous"):
            js = "history.back(); 'OK'"
            return self.execute_js(js, b) is not None
        elif act in ("forward", "next"):
            js = "history.forward(); 'OK'"
            return self.execute_js(js, b) is not None
        elif act in ("top", "page top"):
            return self.scroll("top", browser=b)
        elif act in ("bottom", "page bottom"):
            return self.scroll("bottom", browser=b)

        return False

    # ── DOM Extraction & JavaScript Execution ──

    def execute_js(self, javascript: str, browser: Optional[str] = None) -> Optional[str]:
        """Execute arbitrary JavaScript inside the active browser tab."""
        b = browser or self.detect_active_browser()
        escaped_js = javascript.replace('\\', '\\\\').replace('"', '\\"')

        if b == "Google Chrome":
            script = f'''
            tell application "Google Chrome"
                if (count of windows) > 0 then
                    tell active tab of front window
                        return execute javascript "{escaped_js}"
                    end tell
                end if
            end tell
            '''
        else:
            script = f'''
            tell application "Safari"
                if (count of windows) > 0 then
                    tell current tab of front window
                        return do JavaScript "{escaped_js}"
                    end tell
                end if
            end tell
            '''
        try:
            res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=False)
            if res.returncode == 0:
                return res.stdout.strip()
            return None
        except Exception as e:
            logger.debug(f"execute_js error: {e}")
            return None

    def extract_page_content(self, browser: Optional[str] = None, max_chars: int = 4000) -> Dict[str, str]:
        """Extract clean title, URL, and body text from active browser tab in <50ms."""
        b = browser or self.detect_active_browser()
        js_extract = """
        (function() {
            var title = document.title || '';
            var url = location.href || '';
            var body = document.body ? (document.body.innerText || '') : '';
            // Remove excessive blank lines and whitespace
            var cleanBody = body.replace(/\\s*\\n\\s*/g, '\\n').trim();
            return JSON.stringify({
                title: title,
                url: url,
                text: cleanBody.substring(0, 4000)
            });
        })();
        """
        raw_res = self.execute_js(js_extract, b)
        if raw_res:
            try:
                data = json.loads(raw_res)
                return {
                    "title": data.get("title", ""),
                    "url": data.get("url", ""),
                    "text": data.get("text", ""),
                }
            except Exception:
                pass

        # Fallback to AppleScript properties if JS execution is disabled
        if b == "Google Chrome":
            fb_script = """
            tell application "Google Chrome"
                if (count of windows) > 0 then
                    tell active tab of front window
                        return (title) & "\t" & (URL)
                    end tell
                end if
            end tell
            """
        else:
            fb_script = """
            tell application "Safari"
                if (count of windows) > 0 then
                    tell current tab of front window
                        return (name) & "\t" & (URL)
                    end tell
                end if
            end tell
            """
        res = subprocess.run(["osascript", "-e", fb_script], capture_output=True, text=True, check=False)
        parts = res.stdout.strip().split("\t")
        return {
            "title": parts[0].strip() if len(parts) > 0 else "Active Page",
            "url": parts[1].strip() if len(parts) > 1 else "",
            "text": "",
        }

    # ── Smart Page Summarization with NVIDIA NIM ──

    async def summarize_active_page(self, nim_client) -> str:
        """Extract active page and summarize with NVIDIA NIM reasoning model."""
        content = self.extract_page_content()
        title = content.get("title", "Active Page")
        url = content.get("url", "")
        text = content.get("text", "")

        if not text and not url:
            return "No active webpage content found to summarize. Please ensure Chrome or Safari is open."

        prompt = f"""
You are Jarvis, an advanced AI desktop executive assistant.
Summarize the following webpage that the user is currently viewing in their browser.

Page Title: {title}
URL: {url}
Content Snippet:
{text[:3000]}

Provide:
1. A 1-sentence executive overview of what this page or article is about.
2. 3 key bullet points capturing the core insights, facts, or actionable takeaways.
Keep it crisp, professional, and directly useful.
"""
        try:
            summary = await nim_client.chat_reasoning(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=400,
            )
            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating page summary: {e}")
            return f"Page: '{title}' ({url})\nContent preview: {text[:300]}..."

    # ── DOM-Level Link Click & Form Filling ──

    def click_link_by_text(self, text: str, browser: Optional[str] = None) -> bool:
        """Find and click an anchor, button, or clickable element by text directly via DOM."""
        b = browser or self.detect_active_browser()
        clean_text = text.lower().strip()
        js_template = """
        (function() {
            var target = "__TARGET__";
            var clickables = Array.from(document.querySelectorAll('a, button, [role="button"], input[type="submit"], input[type="button"]'));
            for (var i = 0; i < clickables.length; i++) {
                var el = clickables[i];
                var t = (el.innerText || el.value || el.getAttribute('aria-label') || '').toLowerCase().trim();
                if (t === target || t.indexOf(target) !== -1) {
                    el.scrollIntoView({behavior: 'smooth', block: 'center'});
                    el.click();
                    return "CLICKED: " + t;
                }
            }
            return "";
        })();
        """
        js = js_template.replace("__TARGET__", clean_text.replace('\\', '\\\\').replace('"', '\\"'))
        res = self.execute_js(js, b)
        return bool(res and "CLICKED" in res)

    def fill_search_or_input(self, value: str, browser: Optional[str] = None) -> bool:
        """Focus the first search box or primary input field and set its value."""
        b = browser or self.detect_active_browser()
        escaped_val = value.replace('\\', '\\\\').replace('"', '\\"')
        js_template = """
        (function() {
            var inputs = Array.from(document.querySelectorAll('input[type="search"], input[type="text"], input[name*="search" i], input[name*="q" i], textarea'));
            for (var i = 0; i < inputs.length; i++) {
                var el = inputs[i];
                if (el.offsetParent !== null) {
                    el.focus();
                    el.value = "__VAL__";
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                    return "FILLED";
                }
            }
            return "";
        })();
        """
    def click_element_by_id(self, target_id: int, browser: Optional[str] = None) -> bool:
        """Click an element tagged with data-magnum-id in the active browser tab via direct DOM dispatch."""
        b = browser or self.detect_active_browser()
        js = f"""
        (function() {{
            var el = document.querySelector('[data-magnum-id="{target_id}"]');
            if (el) {{
                el.scrollIntoView({{behavior: 'smooth', block: 'center'}});
                el.click();
                return "CLICKED_ID: {target_id}";
            }}
            return "";
        }})();
        """
        res = self.execute_js(js, b)
        return bool(res and "CLICKED_ID" in res)

    def type_element_by_id(self, target_id: int, text: str, browser: Optional[str] = None) -> bool:
        """Focus an input element tagged with data-magnum-id and type/set its value."""
        b = browser or self.detect_active_browser()
        escaped_val = text.replace('\\', '\\\\').replace('"', '\\"')
        js = f"""
        (function() {{
            var el = document.querySelector('[data-magnum-id="{target_id}"]');
            if (el) {{
                el.scrollIntoView({{behavior: 'smooth', block: 'center'}});
                el.focus();
                el.value = "{escaped_val}";
                el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                return "TYPED_ID: {target_id}";
            }}
            return "";
        }})();
        """
        res = self.execute_js(js, b)
        return bool(res and "TYPED_ID" in res)

    def select_dropdown_option(self, target_id: int, option_text_or_value: str, browser: Optional[str] = None) -> bool:
        """Select an option in a <select> or ARIA combobox by text or value."""
        b = browser or self.detect_active_browser()
        clean_opt = option_text_or_value.lower().strip()
        escaped_opt = clean_opt.replace('\\', '\\\\').replace('"', '\\"')
        js = f"""
        (function() {{
            var el = document.querySelector('[data-magnum-id="{target_id}"]');
            if (!el) return "";

            // 1. Native HTML <select>
            if (el.tagName.toLowerCase() === 'select') {{
                for (var i = 0; i < el.options.length; i++) {{
                    var opt = el.options[i];
                    var t = (opt.text || opt.value || '').toLowerCase().trim();
                    if (t === "{escaped_opt}" || t.indexOf("{escaped_opt}") !== -1) {{
                        el.selectedIndex = i;
                        el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        return "SELECTED_OPTION: " + opt.text;
                    }}
                }}
            }}

            // 2. Custom ARIA dropdown / combobox
            el.scrollIntoView({{behavior: 'smooth', block: 'center'}});
            el.click();
            setTimeout(function() {{
                var items = Array.from(document.querySelectorAll('[role="option"], li, .dropdown-item, .select-option'));
                for (var j = 0; j < items.length; j++) {{
                    var it = items[j];
                    if (it.innerText && it.innerText.toLowerCase().indexOf("{escaped_opt}") !== -1) {{
                        it.click();
                        break;
                    }}
                }}
            }}, 200);
            return "SELECTED_ARIA_DROPDOWN";
        }})();
        """
        res = self.execute_js(js, b)
        return bool(res and "SELECTED" in res)

    def set_checkbox(self, target_id: int, checked: bool = True, browser: Optional[str] = None) -> bool:
        """Set checkbox or radio button state."""
        b = browser or self.detect_active_browser()
        chk_str = "true" if checked else "false"
        js = f"""
        (function() {{
            var el = document.querySelector('[data-magnum-id="{target_id}"]');
            if (el) {{
                el.scrollIntoView({{behavior: 'smooth', block: 'center'}});
                el.checked = {chk_str};
                el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                return "CHECKBOX_SET: {chk_str}";
            }}
            return "";
        }})();
        """
        res = self.execute_js(js, b)
        return bool(res and "CHECKBOX_SET" in res)

    def scroll_element_into_view(self, target_id: int, browser: Optional[str] = None) -> bool:
        """Scroll an element directly into the center of the browser viewport."""
        b = browser or self.detect_active_browser()
        js = f"""
        (function() {{
            var el = document.querySelector('[data-magnum-id="{target_id}"]');
            if (el) {{
                el.scrollIntoView({{behavior: 'smooth', block: 'center'}});
                return "SCROLLED_INTO_VIEW";
            }}
            return "";
        }})();
        """
        res = self.execute_js(js, b)
        return bool(res and "SCROLLED_INTO_VIEW" in res)

    def batch_fill_form(self, fields: Dict[str, str], browser: Optional[str] = None) -> int:
        """Fill multiple form inputs in a single instant pass."""
        b = browser or self.detect_active_browser()
        json_fields = json.dumps(fields).replace('\\', '\\\\').replace('"', '\\"')
        js = f"""
        (function() {{
            var payload = JSON.parse("{json_fields}");
            var filledCount = 0;
            for (var key in payload) {{
                var val = payload[key];
                var el = null;
                if (/^\\d+$/.test(key)) {{
                    el = document.querySelector('[data-magnum-id="' + key + '"]');
                }}
                if (!el) {{
                    el = document.querySelector('input[name="' + key + '"], input[placeholder*="' + key + '" i], textarea[name="' + key + '"]');
                }}
                if (el) {{
                    el.focus();
                    el.value = val;
                    el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    filledCount++;
                }}
            }}
            return "FILLED_COUNT: " + filledCount;
        }})();
        """
        res = self.execute_js(js, b)
        if res and "FILLED_COUNT:" in res:
            try:
                return int(res.split("FILLED_COUNT:")[1].strip())
            except ValueError:
                pass
        return 0


_global_browser_controller: Optional[JarvisBrowserController] = None


def get_browser_controller() -> JarvisBrowserController:
    """Singleton getter for JarvisBrowserController."""
    global _global_browser_controller
    if _global_browser_controller is None:
        _global_browser_controller = JarvisBrowserController()
    return _global_browser_controller
