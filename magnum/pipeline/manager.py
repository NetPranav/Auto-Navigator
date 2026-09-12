"""
Phased Pipeline Autopilot for Auto-Navigator / Magnum.
Automates multi-phase development in Antigravity IDE:
1. Focus Antigravity IDE
2. Click AI Agent input box and paste Phase prompt
3. Wait for 'Proceed' (implementation plan) and click it
4. Watch for 'Submit' button until finished
5. Move to next Phase until entire project is built!
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PipelineStep(BaseModel):
    phase_number: int
    prompt: str
    status: str = "pending"  # pending, active, waiting_proceed, waiting_submit, completed, failed


class ProjectPipeline(BaseModel):
    name: str
    steps: List[PipelineStep] = Field(default_factory=list)


class PipelineManager:
    """Manages project pipelines and autonomous execution."""

    @staticmethod
    def parse_phases_file(filepath: Path) -> List[str]:
        """Parse phases from a markdown or text file."""
        if not filepath.exists():
            raise FileNotFoundError(f"Phases file not found: {filepath}")

        content = filepath.read_text(encoding="utf-8").strip()
        return PipelineManager.parse_phases_text(content)

    @staticmethod
    def parse_phases_text(content: str) -> List[str]:
        """Split text into distinct phase prompts."""
        def _clean_phase(p: str) -> str:
            p = re.sub(r"^[-=_\s]+", "", p)
            p = re.sub(r"[-=_\s]+$", "", p)
            p = re.sub(r"^(?:###?\s*)?Phase\s+\d+[:\-\.]?\s*", "", p, flags=re.IGNORECASE)
            return p.strip()

        # 1. Check for '---' dividers first if present
        if "---" in content:
            div_splits = [_clean_phase(p) for p in content.split("---") if _clean_phase(p)]
            if len(div_splits) >= 2:
                return div_splits

        # 2. Check for 'Phase 1:', 'Phase 2:', etc. (with optional indentation or markdown headers)
        phase_splits = re.split(r"(?i)(?:^|\n+)\s*(?:###?\s*)?Phase\s+\d+[:\-\.]?\s*", content)
        cleaned = [_clean_phase(p) for p in phase_splits if _clean_phase(p)]
        if len(cleaned) >= 2:
            return cleaned

        # 3. Check for double blank lines
        paragraphs = [_clean_phase(p) for p in re.split(r"\n\s*\n\s*\n", content) if _clean_phase(p)]
        if len(paragraphs) >= 2:
            return paragraphs

        # Single phase fallback
        c = _clean_phase(content)
        return [c] if c else []

    async def run_pipeline(self, pipeline: ProjectPipeline, agent) -> bool:
        """Run the multi-phase pipeline autonomously."""
        from magnum.notifications import send_notification
        overlay = agent.overlay
        voice = agent.voice_engine
        driver = agent.driver

        total_phases = len(pipeline.steps)
        logger.info(f"Starting pipeline '{pipeline.name}' ({total_phases} phases)")
        voice.speak(f"Starting pipeline {pipeline.name} with {total_phases} phases.", blocking=False)
        send_notification("🚀 Pipeline Started", f"'{pipeline.name}' ({total_phases} phases)")

        overlay.set_status(f"PIPELINE: {pipeline.name.upper()}")

        step_titles = [f"Phase {s.phase_number}" for s in pipeline.steps]
        completed_indices: List[int] = []

        for idx, step in enumerate(pipeline.steps, start=1):
            overlay.update_plan(step_titles, active_idx=idx, completed=completed_indices)
            step.status = "active"
            voice.speak(f"Beginning Phase {step.phase_number}.", blocking=False)

            # 1. Bring Antigravity to foreground
            await driver.navigate("Antigravity IDE")
            await asyncio.sleep(1.0)

            # 2. Click AI agent input box and paste prompt
            overlay.set_status(f"PHASE {idx}/{total_phases}: PASTING PROMPT")
            
            # Click bottom-left prompt area
            screen_w, screen_h = 1440.0, 900.0
            await driver.click(screen_w * 0.45, screen_h * 0.92)
            await asyncio.sleep(0.3)
            
            # Type/Paste phase prompt
            await driver.type_text(step.prompt)
            await asyncio.sleep(0.5)
            await driver.press_key("return")
            await asyncio.sleep(2.0)

            # 3. Wait for 'Proceed' button (implementation plan)
            overlay.set_status(f"PHASE {idx}/{total_phases}: WAITING FOR PROCEED")
            step.status = "waiting_proceed"
            logger.info(f"Phase {idx}: Waiting for 'Proceed' button...")

            proceed_clicked = False
            # Poll for 'Proceed' button for up to 60 seconds
            for _ in range(30):
                screenshot = driver.take_screenshot()
                from magnum.grounding import ScreenGrounder
                coords = ScreenGrounder.find_element_by_text(screenshot, "Proceed")
                if coords:
                    px, py = coords
                    logger.info(f"Phase {idx}: 'Proceed' found at ({px:.0f}, {py:.0f}) — clicking!")
                    await driver.click(px, py)
                    voice.play_chime("/System/Library/Sounds/Tink.aiff")
                    proceed_clicked = True
                    break
                await asyncio.sleep(2.0)

            # 4. Watch for 'Submit' button until finished
            overlay.set_status(f"PHASE {idx}/{total_phases}: WATCHING SUBMIT")
            step.status = "waiting_submit"
            logger.info(f"Phase {idx}: Watching for 'Submit' button...")

            submit_clicked_count = 0
            # Wait for generation to progress and click Submit
            for _ in range(45):  # up to 90 seconds per phase
                screenshot = driver.take_screenshot()
                from magnum.grounding import ScreenGrounder
                s_coords = ScreenGrounder.find_element_by_text(screenshot, "Submit")
                if s_coords:
                    sx, sy = s_coords
                    logger.info(f"Phase {idx}: 'Submit' found at ({sx:.0f}, {sy:.0f}) — clicking!")
                    await driver.click(sx, sy)
                    voice.play_chime("/System/Library/Sounds/Tink.aiff")
                    submit_clicked_count += 1
                    await asyncio.sleep(3.0)
                    break
                await asyncio.sleep(2.0)

            # Phase completed!
            step.status = "completed"
            completed_indices.append(idx)
            voice.play_chime("/System/Library/Sounds/Glass.aiff")
            voice.speak(f"Phase {step.phase_number} complete.", blocking=False)
            send_notification(f"✓ Phase {idx} Complete", f"Phase {idx}/{total_phases} finished in {pipeline.name}")
            await asyncio.sleep(2.0)

        # All phases complete!
        overlay.update_plan(step_titles, active_idx=total_phases, completed=completed_indices)
        voice.play_chime("/System/Library/Sounds/Glass.aiff")
        finish_msg = f"Project pipeline '{pipeline.name}' completed all {total_phases} phases!"
        voice.speak(finish_msg, blocking=False)
        send_notification("🎉 Pipeline Completed", finish_msg, sound="Hero")
        return True


_global_pipeline_manager: Optional[PipelineManager] = None


def get_pipeline_manager() -> PipelineManager:
    global _global_pipeline_manager
    if _global_pipeline_manager is None:
        _global_pipeline_manager = PipelineManager()
    return _global_pipeline_manager
