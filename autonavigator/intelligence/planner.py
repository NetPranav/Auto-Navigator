"""
Dynamic AI Planning and Step Checklist Management.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

from autonavigator.intelligence.prompts import DYNAMIC_PLANNER_PROMPT

logger = logging.getLogger(__name__)


class PlanStep(BaseModel):
    """A single actionable step in the plan checklist."""

    step_index: int
    title: str
    description: str
    status: Literal["pending", "active", "completed", "failed"] = "pending"


class PlanChecklist(BaseModel):
    """The master checklist of steps generated dynamically by AI."""

    goal_summary: str
    steps: List[PlanStep]
    active_index: int = 1

    @property
    def current_step(self) -> Optional[PlanStep]:
        for s in self.steps:
            if s.step_index == self.active_index:
                return s
        return None

    @property
    def is_finished(self) -> bool:
        return all(s.status == "completed" for s in self.steps) or self.active_index > len(self.steps)

    def advance_to_next_step(self) -> Optional[PlanStep]:
        """Mark current step as completed and move to next step."""
        current = self.current_step
        if current:
            current.status = "completed"
        self.active_index += 1
        next_step = self.current_step
        if next_step:
            next_step.status = "active"
        return next_step

    def get_titles_list(self) -> List[str]:
        """Return list of step titles for UI display."""
        return [f"{s.step_index}. {s.title}" for s in self.steps]
