"""
Workflow Manager for Magnum AI Assistant.
Provides persistent storage, fuzzy retrieval, and execution schemas for custom user workflows.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
from pathlib import Path
import time
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_WORKFLOWS_PATH = Path.home() / ".magnum_workflows.json"


class WorkflowStep(BaseModel):
    """A single step in a multi-step workflow."""
    step_number: int
    instruction: str


class Workflow(BaseModel):
    """A named multi-step sequence of instructions."""
    name: str
    description: str = ""
    steps: List[WorkflowStep] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)


class WorkflowManager:
    """
    Manages custom workflows stored persistently in JSON.
    Supports fuzzy matching so spoken variations correctly resolve to the intended workflow.
    """

    _instance: Optional[WorkflowManager] = None

    def __init__(self, filepath: Optional[Path] = None) -> None:
        self.filepath = filepath or DEFAULT_WORKFLOWS_PATH
        self._workflows: Dict[str, Workflow] = {}
        self._load()

    @classmethod
    def get_instance(cls, filepath: Optional[Path] = None) -> WorkflowManager:
        if cls._instance is None:
            cls._instance = cls(filepath)
        return cls._instance

    def _load(self) -> None:
        """Load workflows from persistent file."""
        if not self.filepath.exists():
            self._save()
            return

        try:
            data = json.loads(self.filepath.read_text(encoding="utf-8"))
            self._workflows = {}
            for item in data.get("workflows", []):
                wf = Workflow.model_validate(item)
                self._workflows[wf.name.lower().strip()] = wf
            logger.info(f"Loaded {len(self._workflows)} custom workflow(s)")
        except Exception as e:
            logger.warning(f"Error loading workflows: {e}")
            self._workflows = {}

    def _save(self) -> None:
        """Save workflows to persistent file."""
        try:
            data = {
                "version": "1.0",
                "workflows": [wf.model_dump() for wf in self._workflows.values()],
            }
            self.filepath.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Error saving workflows: {e}")

    def save_workflow(self, name: str, steps: List[str], description: str = "") -> Workflow:
        """Create or update a workflow with the given sequential step instructions."""
        clean_name = name.strip()
        wf_steps = [
            WorkflowStep(step_number=i + 1, instruction=step.strip())
            for i, step in enumerate(steps)
            if step.strip()
        ]
        wf = Workflow(
            name=clean_name,
            description=description.strip(),
            steps=wf_steps,
            created_at=time.time(),
        )
        self._workflows[clean_name.lower()] = wf
        self._save()
        logger.info(f"Saved workflow '{clean_name}' with {len(wf_steps)} step(s)")
        return wf

    def get_workflow(self, name: str) -> Optional[Workflow]:
        """Look up a workflow by exact or case-insensitive name."""
        return self._workflows.get(name.lower().strip())

    def find_workflow(self, query: str) -> Optional[Workflow]:
        """
        Fuzzy lookup for a workflow name matching the user's spoken query.
        Handles variations like:
        - 'run antigravity phases' -> 'Antigravity phases and auto submit'
        - 'meeting ready' -> 'Getting meeting ready'
        """
        q = query.lower().strip()
        # 1. Exact match
        if q in self._workflows:
            return self._workflows[q]

        # 2. Substring match or keyword overlap with ratio gating
        q_tokens = set(q.split())
        best_overlap_wf: Optional[Workflow] = None
        best_overlap_score = 0.0

        for key, wf in self._workflows.items():
            wf_tokens = set(key.split())
            overlap = len(q_tokens & wf_tokens)
            if not overlap or not q_tokens or not wf_tokens:
                continue

            q_ratio = overlap / len(q_tokens)
            wf_ratio = overlap / len(wf_tokens)

            # Require significant overlap in both query and workflow name
            # Disallows long instructional sentences (e.g. 15 words) from falsely matching a 2-word workflow
            if wf_ratio >= 0.6 and q_ratio >= 0.30:
                score = q_ratio + wf_ratio
                if score > best_overlap_score:
                    best_overlap_score = score
                    best_overlap_wf = wf

        if best_overlap_wf:
            return best_overlap_wf

        # 3. Fuzzy string similarity via difflib (only for short queries close in length to workflow name)
        if len(q.split()) <= 6:
            names = list(self._workflows.keys())
            matches = difflib.get_close_matches(q, names, n=1, cutoff=0.65)
            if matches:
                return self._workflows[matches[0]]

        return None

    def list_workflows(self) -> List[Workflow]:
        """Return all saved workflows."""
        return list(self._workflows.values())

    def delete_workflow(self, name: str) -> bool:
        """Delete a workflow by name."""
        clean = name.lower().strip()
        if clean in self._workflows:
            del self._workflows[clean]
            self._save()
            logger.info(f"Deleted workflow '{name}'")
            return True
        return False


def get_workflow_manager() -> WorkflowManager:
    """Convenience accessor for the WorkflowManager singleton."""
    return WorkflowManager.get_instance()
