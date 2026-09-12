"""
Magnum Persistent Memory & Knowledge Store.
Learns from user corrections, app name clarifications, and past interactions.
Persists across restarts in ~/.magnum_memory.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
MEMORY_FILE = Path.home() / ".magnum_memory.json"


class LearnedApp(BaseModel):
    name: str
    actual_name: str
    process_name: Optional[str] = None
    is_already_open: bool = False
    notes: Optional[str] = None


class VisualAnchor(BaseModel):
    label: str
    x: float
    y: float
    app_name: Optional[str] = None
    surrounding_text: Optional[str] = None


class MagnumMemory(BaseModel):
    app_mappings: Dict[str, str] = Field(
        default_factory=lambda: {
            "antigravity": "Antigravity IDE",
            "antigravity ide": "Antigravity IDE",
            "vscode": "Visual Studio Code",
            "vs code": "Visual Studio Code",
            "chrome": "Google Chrome",
            "terminal": "Terminal",
            "iterm": "iTerm2",
        }
    )
    user_rules: List[str] = Field(
        default_factory=lambda: [
            "Antigravity is a desktop IDE, never open browser or Google for it.",
            "Antigravity is typically already running in the current workspace.",
            "When asked to wait for or watch a button (like Submit), run a background watcher, do not click checklist text.",
        ]
    )
    visual_anchors: Dict[str, VisualAnchor] = Field(default_factory=dict)
    custom_knowledge: Dict[str, str] = Field(default_factory=dict)

    def save(self) -> None:
        try:
            MEMORY_FILE.write_text(self.model_dump_json(indent=2))
        except Exception as e:
            logger.warning(f"Could not save memory: {e}")

    @classmethod
    def load(cls) -> MagnumMemory:
        if MEMORY_FILE.exists():
            try:
                data = json.loads(MEMORY_FILE.read_text())
                return cls(**data)
            except Exception as e:
                logger.warning(f"Could not load memory, using defaults: {e}")
        inst = cls()
        inst.save()
        return inst

    def learn_app_mapping(self, alias: str, real_name: str) -> None:
        """Learn that alias refers to real_name."""
        self.app_mappings[alias.lower().strip()] = real_name.strip()
        logger.info(f"🧠 Learned: '{alias}' -> '{real_name}'")
        self.save()

    def learn_user_rule(self, rule: str) -> None:
        """Add a learned rule."""
        if rule not in self.user_rules:
            self.user_rules.append(rule)
            logger.info(f"🧠 Learned rule: {rule}")
            self.save()

    def learn_visual_anchor(
        self,
        label: str,
        x: float,
        y: float,
        app_name: Optional[str] = None,
        surrounding_text: Optional[str] = None,
    ) -> None:
        """Learn physical button coordinates and anchor."""
        clean_key = label.lower().strip()
        self.visual_anchors[clean_key] = VisualAnchor(
            label=label,
            x=x,
            y=y,
            app_name=app_name,
            surrounding_text=surrounding_text,
        )
        logger.info(f"🎯 Learned Visual Anchor: '{label}' at ({x:.0f}, {y:.0f})")
        self.save()

    def get_visual_anchor(self, label: str) -> Optional[VisualAnchor]:
        """Look up learned visual anchor coordinates."""
        return self.visual_anchors.get(label.lower().strip())

    def resolve_app_name(self, name: str) -> str:
        """Resolve app name using learned knowledge."""
        lower = name.lower().strip()
        return self.app_mappings.get(lower, name)

    def get_prompt_context(self) -> str:
        """Format learned knowledge for LLM system prompts."""
        lines = ["[LEARNED SYSTEM KNOWLEDGE & CORRECTIONS]"]
        for alias, real in self.app_mappings.items():
            lines.append(f"- App '{alias}' is '{real}'")
        for rule in self.user_rules:
            lines.append(f"- RULE: {rule}")
        for k, v in self.custom_knowledge.items():
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)


# Singleton
_memory_instance: Optional[MagnumMemory] = None


def get_memory() -> MagnumMemory:
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = MagnumMemory.load()
    return _memory_instance
