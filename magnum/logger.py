"""
Magnum Diagnostic & Activity Logging System with Comprehensive Flight Recorder.
Maintains persistent, structured logs in 'magnum.log' (and 'magnm.log')
capturing exact task instructions, dynamic plans, step-by-step perception,
internal AI reasoning, execution tiers, attempts, and failure root-cause analysis.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# Primary log file path in workspace root (or current working directory)
LOG_FILE_NAME = "magnum.log"
ALIAS_LOG_FILE_NAME = "magnm.log"
RUNS_DIR_NAME = "runs"
LATEST_RUN_FILE = "latest_run.md"


class MagnumLogFormatter(logging.Formatter):
    """Clean, high-visibility log format for Magnum execution tracing."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        level = record.levelname.ljust(5)
        module = f"{record.name}:{record.lineno}"
        msg = record.getMessage()
        if record.exc_info:
            exc_text = "".join(traceback.format_exception(*record.exc_info)).strip()
            return f"[{timestamp}] [{level}] [{module}] {msg}\n{exc_text}"
        return f"[{timestamp}] [{level}] [{module}] {msg}"


class AutoFlushFileHandler(logging.FileHandler):
    """FileHandler that automatically flushes on every emit for real-time log tracking."""
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


_logger_initialized = False
_file_handler: Optional[AutoFlushFileHandler] = None


def _ensure_initialized() -> None:
    global _logger_initialized
    if not _logger_initialized:
        setup_magnum_logging()


def setup_magnum_logging(
    log_dir: Optional[Path] = None,
    log_level: int = logging.INFO,
    force: bool = False,
) -> Path:
    """
    Initialize persistent file logging for all magnum modules.
    Writes to magnum.log and creates magnm.log alias in the workspace root.
    """
    global _logger_initialized, _file_handler

    target_dir = log_dir or Path.cwd()
    target_dir.mkdir(parents=True, exist_ok=True)
    log_file = (target_dir / LOG_FILE_NAME).resolve()
    alias_file = (target_dir / ALIAS_LOG_FILE_NAME).resolve()

    if _logger_initialized and _file_handler and not log_dir and not force:
        return Path(_file_handler.baseFilename)

    formatter = MagnumLogFormatter()

    root_logger = logging.getLogger()
    magnum_logger = logging.getLogger("magnum")
    autonav_logger = logging.getLogger("autonavigator")

    if _file_handler:
        for l in (root_logger, magnum_logger, autonav_logger):
            if _file_handler in l.handlers:
                l.removeHandler(_file_handler)
        try:
            _file_handler.close()
        except Exception:
            pass

    # File handler writing append-only logs with instant autoflush
    handler = AutoFlushFileHandler(log_file, encoding="utf-8", mode="a")
    handler.setLevel(log_level)
    handler.setFormatter(formatter)
    _file_handler = handler

    for l in (root_logger, magnum_logger, autonav_logger):
        if handler not in l.handlers:
            l.addHandler(handler)
        if l.level > log_level or l.level == 0:
            l.setLevel(log_level)

    # Maintain magnm.log alias via symlink or duplicate reference
    try:
        if not alias_file.exists():
            if hasattr(os, "symlink"):
                try:
                    os.symlink(log_file.name, alias_file)
                except OSError:
                    alias_file.write_text(f"See {LOG_FILE_NAME}\n", encoding="utf-8")
    except Exception:
        pass

    _logger_initialized = True
    magnum_logger.info(f"=== Magnum Logging Session Initialized at {log_file} ===")
    return log_file


def get_log_file_path() -> Path:
    """Return the absolute path of the active magnum.log file."""
    return Path.cwd() / LOG_FILE_NAME


def get_latest_run_file_path() -> Path:
    """Return the path to runs/latest_run.md."""
    runs_dir = Path.cwd() / RUNS_DIR_NAME
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir / LATEST_RUN_FILE


# ── COMPREHENSIVE TASK FLIGHT RECORDER ──

class AttemptAudit(BaseModel):
    """Exact audit of a single step attempt."""
    attempt_number: int
    timestamp: str = ""
    active_app: str = "Unknown"
    screenshot: str = ""
    ocr_count: int = 0
    a11y_targets_count: int = 0
    ai_thought: str = ""
    chosen_action: str = ""
    target_id: Optional[int] = None
    target_label: Optional[str] = None
    target_coords: Optional[Dict[str, float]] = None
    input_text: Optional[str] = None
    key_pressed: Optional[str] = None
    execution_tier: str = ""
    execution_success: bool = False
    verification_confirmed: bool = False
    verification_note: str = ""
    failure_reason: Optional[str] = None


class StepAudit(BaseModel):
    """Exact audit of a planned step."""
    step_index: int
    title: str
    description: str = ""
    status: str = "PENDING"  # PENDING, IN_PROGRESS, COMPLETED, FAILED
    attempts: List[AttemptAudit] = Field(default_factory=list)


class TaskFlightAudit(BaseModel):
    """Complete audit trail of an entire task execution."""
    task_id: str = ""
    instruction: str = ""
    mode: str = "desktop"
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    status: str = "RUNNING"  # RUNNING, SUCCESS, FAILED
    plan_checklist: List[str] = Field(default_factory=list)
    steps: List[StepAudit] = Field(default_factory=list)
    failure_context: Optional[str] = None
    failure_root_cause: Optional[str] = None
    last_known_screenshot: Optional[str] = None


class TaskFlightRecorder:
    """
    Flight Recorder that captures the exact planning, perception, reasoning,
    execution attempts, and failure points of every task.
    """

    def __init__(self) -> None:
        self.current_audit: Optional[TaskFlightAudit] = None
        self._start_epoch: float = 0.0

    def start_task(self, instruction: str, mode: str = "desktop") -> None:
        """Start recording a new task."""
        _ensure_initialized()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._start_epoch = time.time()
        task_id = f"task_{int(self._start_epoch)}"
        self.current_audit = TaskFlightAudit(
            task_id=task_id,
            instruction=instruction,
            mode=mode,
            start_time=now_str,
            status="RUNNING",
        )
        log_instruction(instruction, mode=mode)
        self._save_checkpoint()

    def record_plan(self, goal: str, steps: List[Any]) -> None:
        """Record the exact generated dynamic plan."""
        if not self.current_audit:
            return
        step_descs = []
        for i, s in enumerate(steps, start=1):
            title = getattr(s, "title", str(s))
            desc = getattr(s, "description", "")
            step_descs.append(f"[{i}] {title}" + (f" — {desc}" if desc else ""))
            self.current_audit.steps.append(
                StepAudit(step_index=i, title=title, description=desc, status="PENDING")
            )
        self.current_audit.plan_checklist = step_descs
        log_plan(goal, step_descs)
        self._save_checkpoint()

    def start_step(self, step_index: int, total_steps: int, title: str, description: str = "") -> None:
        """Mark a step as in progress."""
        if not self.current_audit:
            return
        for s in self.current_audit.steps:
            if s.step_index == step_index:
                s.status = "IN_PROGRESS"
                break
        log_step_start(step_index, total_steps, title, description)
        self._save_checkpoint()

    def record_attempt(
        self,
        step_index: int,
        attempt_number: int,
        active_app: str,
        screenshot: str,
        ocr_count: int,
        a11y_targets_count: int,
        ai_thought: str,
        chosen_action: str,
        execution_tier: str,
        target_id: Optional[int] = None,
        target_label: Optional[str] = None,
        target_coords: Optional[Dict[str, float]] = None,
        input_text: Optional[str] = None,
        key_pressed: Optional[str] = None,
        execution_success: bool = True,
        verification_confirmed: bool = False,
        verification_note: str = "",
        failure_reason: Optional[str] = None,
    ) -> None:
        """Record the full details of an execution attempt."""
        if not self.current_audit:
            return

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        attempt_audit = AttemptAudit(
            attempt_number=attempt_number,
            timestamp=now_str,
            active_app=active_app,
            screenshot=screenshot,
            ocr_count=ocr_count,
            a11y_targets_count=a11y_targets_count,
            ai_thought=ai_thought,
            chosen_action=chosen_action,
            target_id=target_id,
            target_label=target_label,
            target_coords=target_coords,
            input_text=input_text,
            key_pressed=key_pressed,
            execution_tier=execution_tier,
            execution_success=execution_success,
            verification_confirmed=verification_confirmed,
            verification_note=verification_note,
            failure_reason=failure_reason,
        )

        for s in self.current_audit.steps:
            if s.step_index == step_index:
                s.attempts.append(attempt_audit)
                break

        if screenshot:
            self.current_audit.last_known_screenshot = screenshot

        self._save_checkpoint()

    def complete_step(self, step_index: int, success: bool = True) -> None:
        """Mark step as complete or failed."""
        if not self.current_audit:
            return
        for s in self.current_audit.steps:
            if s.step_index == step_index:
                s.status = "COMPLETED" if success else "FAILED"
                break
        self._save_checkpoint()

    def record_failure(
        self,
        context: str,
        error: Any,
        step_info: Optional[str] = None,
        screenshot_path: Optional[str] = None,
    ) -> None:
        """Record failure root cause and details."""
        if self.current_audit:
            self.current_audit.status = "FAILED"
            self.current_audit.failure_context = context
            self.current_audit.failure_root_cause = str(error)
            if screenshot_path:
                self.current_audit.last_known_screenshot = screenshot_path
        log_failure(context, error, step_info=step_info, screenshot_path=screenshot_path)
        self._save_checkpoint()

    def finish_task(self, success: bool) -> None:
        """Conclude the task and generate the full flight audit report."""
        if not self.current_audit:
            return
        self.current_audit.status = "SUCCESS" if success else "FAILED"
        self.current_audit.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self._start_epoch:
            self.current_audit.duration_seconds = round(time.time() - self._start_epoch, 2)

        log_task_complete(
            self.current_audit.instruction,
            success=success,
            duration_seconds=self.current_audit.duration_seconds,
        )
        self._save_final_report()

    def render_markdown_report(self) -> str:
        """Render the complete, detailed markdown audit breakdown."""
        if not self.current_audit:
            return "No task audit currently recorded."

        a = self.current_audit
        status_emoji = "✅ SUCCESS" if a.status == "SUCCESS" else ("❌ FAILED" if a.status == "FAILED" else "⏳ RUNNING")

        lines = [
            "================================================================================",
            "🛸 MAGNUM TASK FLIGHT RECORDER — COMPLETE EXECUTION BREAKDOWN",
            "================================================================================",
            f"**Task Instruction** : {a.instruction}",
            f"**Execution Mode**   : {a.mode.upper()}",
            f"**Started At**       : {a.start_time}",
            f"**Ended At**         : {a.end_time or 'In Progress...'}",
            f"**Total Duration**   : {a.duration_seconds}s",
            f"**Overall Status**   : {status_emoji}",
            "",
            "--------------------------------------------------------------------------------",
            "📋 1. THE EXACT PLAN GENERATED",
            "--------------------------------------------------------------------------------",
        ]

        if not a.steps:
            lines.append("No steps were planned or recorded.")
        else:
            for s in a.steps:
                status_icon = "✓" if s.status == "COMPLETED" else ("✗" if s.status == "FAILED" else "•")
                lines.append(f"[{status_icon}] Step {s.step_index}: {s.title}")
                if s.description:
                    lines.append(f"    Description: {s.description}")
                lines.append(f"    Status: {s.status}")

        lines.extend([
            "",
            "--------------------------------------------------------------------------------",
            "🔬 2. STEP-BY-STEP EXECUTION AUDIT (HOW IT TRIED TO DO IT)",
            "--------------------------------------------------------------------------------",
        ])

        for s in a.steps:
            lines.append(f"\n▶ STEP {s.step_index}: {s.title} ({s.status})")
            if not s.attempts:
                lines.append("  └─ (No execution attempts recorded for this step)")
                continue

            for att in s.attempts:
                lines.append(f"  ├─ Attempt {att.attempt_number} [{att.timestamp}]")
                lines.append(f"  │  ├─ Perception    : Active App='{att.active_app}' | OCR Elements={att.ocr_count} | A11y Targets={att.a11y_targets_count}")
                if att.screenshot:
                    lines.append(f"  │  ├─ Screenshot    : {att.screenshot}")
                if att.ai_thought:
                    lines.append(f"  │  ├─ AI Thought    : \"{att.ai_thought}\"")
                lines.append(f"  │  ├─ Chosen Action : {att.chosen_action}")
                if att.target_id is not None or att.target_label:
                    lines.append(f"  │  ├─ Target Element: ID=[{att.target_id}] Label='{att.target_label or ''}' Coords={att.target_coords or ''}")
                if att.input_text:
                    lines.append(f"  │  ├─ Text Input    : \"{att.input_text}\"")
                if att.key_pressed:
                    lines.append(f"  │  ├─ Key Pressed   : {att.key_pressed}")
                lines.append(f"  │  ├─ Execution Tier: {att.execution_tier}")
                exec_status = "SUCCESS" if att.execution_success else "FAILED"
                lines.append(f"  │  ├─ Tool Outcome  : {exec_status}")
                if att.verification_note:
                    lines.append(f"  │  ├─ Verification  : {att.verification_note}")
                if att.failure_reason:
                    lines.append(f"  │  └─ Failure Reason: {att.failure_reason}")
                else:
                    lines.append(f"  │  └─ Status        : {'Verified' if att.verification_confirmed else 'Attempted'}")

        if a.status == "FAILED" or a.failure_root_cause:
            lines.extend([
                "",
                "--------------------------------------------------------------------------------",
                "🚨 3. ROOT CAUSE & FAILURE ANALYSIS",
                "--------------------------------------------------------------------------------",
                f"Failure Context  : {a.failure_context or 'Unknown failure point'}",
                f"Root Cause Error : {a.failure_root_cause or 'Step incomplete after max attempts'}",
                f"Last Screenshot  : {a.last_known_screenshot or 'None'}",
            ])

        lines.extend([
            "================================================================================",
            "",
        ])
        return "\n".join(lines)

    def _save_checkpoint(self) -> None:
        """Save active state to latest_run.md."""
        try:
            report_text = self.render_markdown_report()
            get_latest_run_file_path().write_text(report_text, encoding="utf-8")
        except Exception:
            pass

    def _save_final_report(self) -> None:
        """Save the final report to runs/ and append to magnum.log."""
        try:
            report_text = self.render_markdown_report()
            # 1. Update runs/latest_run.md
            get_latest_run_file_path().write_text(report_text, encoding="utf-8")

            # 2. Archive to timestamped run file
            if self.current_audit and self.current_audit.task_id:
                runs_dir = Path.cwd() / RUNS_DIR_NAME
                archive_file = runs_dir / f"{self.current_audit.task_id}.md"
                archive_file.write_text(report_text, encoding="utf-8")

            # 3. Append the full narrative breakdown to magnum.log
            logger = logging.getLogger("magnum.flight_recorder")
            logger.info(f"\n{report_text}\n")
        except Exception as e:
            logger = logging.getLogger("magnum.flight_recorder")
            logger.warning(f"Could not persist flight report: {e}")


# Global flight recorder singleton
flight_recorder = TaskFlightRecorder()


# ── HELPER LOGGING FUNCTIONS ──

def log_instruction(instruction: str, mode: str = "desktop") -> None:
    """Log the start of a user instruction."""
    _ensure_initialized()
    logger = logging.getLogger("magnum.agent")
    sep = "=" * 70
    logger.info(f"\n{sep}\n▶ NEW USER INSTRUCTION [{mode.upper()}]: '{instruction}'\n{sep}")


def log_plan(goal: str, steps: List[str]) -> None:
    """Log the generated dynamic plan and steps."""
    _ensure_initialized()
    logger = logging.getLogger("magnum.planner")
    step_lines = "\n".join(f"   [{i+1}] {s}" for i, s in enumerate(steps))
    logger.info(f"📋 DYNAMIC PLAN for '{goal}':\n{step_lines}")


def log_step_start(step_index: int, total_steps: int, title: str, description: str = "") -> None:
    """Log the beginning of an individual plan step."""
    _ensure_initialized()
    logger = logging.getLogger("magnum.agent")
    desc = f" - {description}" if description else ""
    logger.info(f"👉 STEP [{step_index}/{total_steps}]: {title}{desc}")


def log_perception(active_app: str, ocr_count: int, a11y_tree_count: int) -> None:
    """Log screen perception and accessibility analysis."""
    _ensure_initialized()
    logger = logging.getLogger("magnum.perception")
    logger.info(f"👁️ SCREEN PERCEPTION: Active App='{active_app}' | OCR Items={ocr_count} | Astra A11y Targets={a11y_tree_count}")


def log_action_execution(
    action_type: str,
    tier: str,
    details: Dict[str, Any],
    success: bool = True,
) -> None:
    """Log a specific action execution."""
    _ensure_initialized()
    logger = logging.getLogger("magnum.execution")
    status = "SUCCESS" if success else "FAILED"
    det_str = " | ".join(f"{k}={v}" for k, v in details.items() if v is not None)
    logger.info(f"⚡ ACTION [{tier}] {action_type} -> [{status}] ({det_str})")


def log_user_interaction(prompt_text: str, user_response: str) -> None:
    """Log HITL questions asked to the user and their responses."""
    _ensure_initialized()
    logger = logging.getLogger("magnum.hitl")
    logger.info(f"💬 HITL INTERACTION: Agent Asked='{prompt_text}' -> User Replied='{user_response}'")


def log_failure(
    context: str,
    error: Any,
    step_info: Optional[str] = None,
    screenshot_path: Optional[str] = None,
) -> None:
    """Log a prominent failure block into magnum.log for user diagnostics."""
    _ensure_initialized()
    logger = logging.getLogger("magnum.failure")
    sep = "!" * 70
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    err_msg = str(error)
    tb_str = ""
    if isinstance(error, BaseException):
        tb_str = "".join(traceback.format_exception(type(error), error, error.__traceback__)).strip()

    lines = [
        f"\n{sep}",
        f"🚨 MAGNUM FAILURE REPORT [{timestamp}]",
        f"Context    : {context}",
    ]
    if step_info:
        lines.append(f"Step       : {step_info}")
    if screenshot_path:
        lines.append(f"Screenshot : {screenshot_path}")
    lines.append(f"Error      : {err_msg}")
    if tb_str:
        lines.append(f"Traceback  :\n{tb_str}")
    lines.append(f"{sep}\n")

    logger.error("\n".join(lines))


def log_task_complete(instruction: str, success: bool, duration_seconds: float = 0.0) -> None:
    """Log task completion summary."""
    _ensure_initialized()
    logger = logging.getLogger("magnum.agent")
    status = "COMPLETED SUCCESSFULLY ✅" if success else "FAILED / STOPPED ❌"
    dur = f" in {duration_seconds:.1f}s" if duration_seconds > 0 else ""
    logger.info(f"🏁 TASK RESULT: {status}{dur} for instruction: '{instruction}'\n")


def get_latest_run_summary() -> str:
    """Return the complete markdown audit report of the latest task run."""
    run_file = get_latest_run_file_path()
    if run_file.exists():
        return run_file.read_text(encoding="utf-8")
    return flight_recorder.render_markdown_report()


def get_recent_logs(lines: int = 50) -> str:
    """Read the last N lines from magnum.log."""
    log_path = get_log_file_path()
    if not log_path.exists():
        return "No magnum.log found yet. Run an instruction to generate logs."
    try:
        all_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        recent = all_lines[-lines:] if len(all_lines) > lines else all_lines
        return "\n".join(recent)
    except Exception as e:
        return f"Error reading log file: {e}"
