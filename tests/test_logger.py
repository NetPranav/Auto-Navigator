import pytest
import os
from pathlib import Path
from magnum.logger import (
    setup_magnum_logging,
    log_instruction,
    log_plan,
    log_step_start,
    log_perception,
    log_action_execution,
    log_user_interaction,
    log_failure,
    log_task_complete,
    get_recent_logs,
    get_log_file_path,
)


def test_logger_setup_and_events(tmp_path):
    log_file = setup_magnum_logging(log_dir=tmp_path)
    assert log_file.exists()
    alias_file = tmp_path / "magnm.log"
    assert alias_file.exists()

    log_instruction("Test Unit Instruction", mode="desktop")
    log_plan("Test Goal", ["Step 1: Check screen", "Step 2: Click button"])
    log_step_start(1, 2, "Check screen", "Inspect elements")
    log_perception("Safari", ocr_count=15, a11y_tree_count=8)
    log_action_execution("CLICK", "Tier 2: OpenComputerUse", {"target_id": 4, "app": "Safari"}, success=True)
    log_user_interaction("Should I click submit?", "Yes, go ahead")
    log_failure("Unit Test Failure Context", "Element 4 obscured by modal", step_info="[2/2] Click button")
    log_task_complete("Test Unit Instruction", success=True, duration_seconds=2.4)

    content = log_file.read_text(encoding="utf-8")
    assert "Test Unit Instruction" in content
    assert "Test Goal" in content
    assert "Step 1: Check screen" in content
    assert "Safari" in content
    assert "MAGNUM FAILURE REPORT" in content
    assert "Element 4 obscured by modal" in content
    assert "COMPLETED SUCCESSFULLY" in content


def test_get_recent_logs():
    logs = get_recent_logs(lines=10)
    assert isinstance(logs, str)
    assert len(logs) > 0


def test_task_flight_recorder_audit():
    from magnum.logger import flight_recorder, get_latest_run_summary
    from magnum.intelligence.planner import PlanStep

    flight_recorder.start_task("Test flight task", mode="desktop")
    flight_recorder.record_plan("Test flight task", [
        PlanStep(step_index=1, title="Test Step 1", description="Description 1"),
    ])
    flight_recorder.start_step(1, 1, "Test Step 1", "Description 1")
    flight_recorder.record_attempt(
        step_index=1,
        attempt_number=1,
        active_app="Antigravity IDE",
        screenshot="test_step.png",
        ocr_count=10,
        a11y_targets_count=5,
        ai_thought="Need to click button #3",
        chosen_action="CLICK",
        execution_tier="Tier 3: Native Cocoa AX",
        target_id=3,
        target_label="Run",
        execution_success=True,
        verification_confirmed=True,
        verification_note="Run button pressed",
    )
    flight_recorder.complete_step(1, success=True)
    flight_recorder.finish_task(success=True)

    summary = get_latest_run_summary()
    assert "MAGNUM TASK FLIGHT RECORDER" in summary
    assert "Test flight task" in summary
    assert "Test Step 1" in summary
    assert "Need to click button #3" in summary
    assert "Tier 3: Native Cocoa AX" in summary
    assert "Run button pressed" in summary

