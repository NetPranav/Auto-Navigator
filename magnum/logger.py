"""
Magnum Diagnostic & Activity Logging System.
Maintains persistent, structured logs in 'magnum.log' (and 'magnm.log')
capturing all user instructions, AI perception, planning, tool executions,
and detailed failure diagnostics.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Primary log file path in workspace root (or current working directory)
LOG_FILE_NAME = "magnum.log"
ALIAS_LOG_FILE_NAME = "magnm.log"


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
    """Log a specific action execution (DOM, OpenComputerUse, Native AX, or Desktop)."""
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
    """
    Log a prominent failure block into magnum.log for user diagnostics.
    """
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
