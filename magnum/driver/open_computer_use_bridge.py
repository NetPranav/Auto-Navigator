"""
Deep-Level Integration Bridge for Open Computer Use (open-codex-computer-use).
Interfaces directly with the native Open Computer Use (ocu) engine, enabling
Auto-Navigator (Magnum) to execute full Astra / Codex Computer Use tool calls.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class OpenComputerUseBridge:
    """Direct Python bridge to the Open Computer Use (open-codex-computer-use) engine."""

    _binary_path: Optional[str] = None

    @classmethod
    def get_binary_path(cls) -> Optional[str]:
        """Resolve path to open-computer-use or ocu binary."""
        if cls._binary_path and os.path.exists(cls._binary_path):
            return cls._binary_path

        # 1. Check local node_modules in project
        workspace_dir = Path(__file__).parent.parent.parent
        local_bin = workspace_dir / "node_modules" / ".bin" / "ocu"
        if local_bin.exists() and os.access(local_bin, os.X_OK):
            cls._binary_path = str(local_bin)
            return cls._binary_path

        local_ocu = workspace_dir / "node_modules" / ".bin" / "open-computer-use"
        if local_ocu.exists() and os.access(local_ocu, os.X_OK):
            cls._binary_path = str(local_ocu)
            return cls._binary_path

        # 2. Check system PATH
        for cmd in ("ocu", "open-computer-use"):
            found = shutil.which(cmd)
            if found:
                cls._binary_path = found
                return cls._binary_path

        return None

    @classmethod
    def is_available(cls) -> bool:
        """Return True if Open Computer Use binary is accessible."""
        return cls.get_binary_path() is not None

    @classmethod
    def call_tool(cls, tool_name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute an Open Computer Use tool call via the CLI.
        Returns parsed JSON result dictionary.
        """
        binary = cls.get_binary_path()
        if not binary:
            return {"isError": True, "error": "Open Computer Use binary not found."}

        cmd = [binary, "call", tool_name]
        if args:
            cmd.extend(["--args", json.dumps(args)])

        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            raw_out = res.stdout.strip()
            if not raw_out and res.stderr:
                return {"isError": True, "error": res.stderr.strip()}

            # Try parsing JSON output
            try:
                data = json.loads(raw_out)
                return data
            except json.JSONDecodeError:
                return {
                    "isError": res.returncode != 0,
                    "content": [{"type": "text", "text": raw_out}],
                }
        except subprocess.TimeoutExpired:
            return {"isError": True, "error": f"Tool '{tool_name}' timed out after 30s."}
        except Exception as e:
            return {"isError": True, "error": str(e)}

    @classmethod
    def list_apps(cls) -> List[Dict[str, Any]]:
        """List running and recently used applications with metadata."""
        res = cls.call_tool("list_apps")
        apps = []
        if not res.get("isError") and res.get("content"):
            raw_text = res["content"][0].get("text", "")
            for line in raw_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Format: "AppName — com.bundle.id [frontmost, running, uses=X]"
                parts = line.split(" — ", 1)
                name = parts[0].strip()
                meta = parts[1].strip() if len(parts) > 1 else ""
                bundle_id = meta.split(" ")[0].strip() if meta else ""
                is_frontmost = "[frontmost" in meta
                is_running = "running" in meta

                apps.append({
                    "name": name,
                    "bundle_id": bundle_id,
                    "is_frontmost": is_frontmost,
                    "is_running": is_running,
                    "raw": line,
                })
        return apps

    @classmethod
    def get_app_state(
        cls, app: str, text_limit: Union[int, str] = 1000, max_tree_nodes: int = 1200
    ) -> Dict[str, Any]:
        """Capture screenshot and accessibility tree for target app."""
        args: Dict[str, Any] = {"app": app, "text_limit": text_limit, "max_tree_nodes": max_tree_nodes}
        return cls.call_tool("get_app_state", args)

    @classmethod
    def click(
        cls,
        app: str,
        element_index: Optional[str] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        click_count: int = 1,
        mouse_button: str = "left",
        click_method: str = "auto",
    ) -> bool:
        """Click an element by element_index or pixel coordinates."""
        args: Dict[str, Any] = {
            "app": app,
            "click_count": click_count,
            "mouse_button": mouse_button,
            "click_method": click_method,
        }
        if element_index is not None:
            args["element_index"] = str(element_index)
        if x is not None and y is not None:
            args["x"] = x
            args["y"] = y

        res = cls.call_tool("click", args)
        return not res.get("isError", False)

    @classmethod
    def type_text(cls, app: str, text: str) -> bool:
        """Type text into active app window."""
        res = cls.call_tool("type_text", {"app": app, "text": text})
        return not res.get("isError", False)

    @classmethod
    def set_value(cls, app: str, element_index: str, value: str) -> bool:
        """Directly set text value of an accessible element without typing delay."""
        res = cls.call_tool("set_value", {"app": app, "element_index": str(element_index), "value": value})
        return not res.get("isError", False)

    @classmethod
    def press_key(cls, app: str, key: str) -> bool:
        """Press key or key combination."""
        res = cls.call_tool("press_key", {"app": app, "key": key})
        return not res.get("isError", False)

    @classmethod
    def scroll(cls, app: str, direction: str = "down", pages: int = 1) -> bool:
        """Scroll in direction by number of pages."""
        res = cls.call_tool("scroll", {"app": app, "direction": direction, "pages": pages})
        return not res.get("isError", False)

    @classmethod
    def drag(cls, app: str, from_x: float, from_y: float, to_x: float, to_y: float) -> bool:
        """Drag from one coordinate to another."""
        args = {"app": app, "from_x": from_x, "from_y": from_y, "to_x": to_x, "to_y": to_y}
        res = cls.call_tool("drag", args)
        return not res.get("isError", False)
