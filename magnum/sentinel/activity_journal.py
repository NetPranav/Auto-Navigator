"""
Activity Journal — Persistent activity log with AI-powered recall.
Records what the user does, saves to disk, and enables natural language queries
about past activity (e.g. "What was the most important thing I was doing?").
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

JOURNAL_FILE = Path.home() / ".magnum_activity_journal.jsonl"
JOURNAL_MAX_AGE_DAYS = 7  # Auto-rotate entries older than 7 days


class ActivityEntry(BaseModel):
    """A single activity log entry."""
    timestamp: str = ""
    epoch: float = 0.0
    app_name: str = ""
    activity_summary: str = ""
    importance: str = "medium"     # critical, high, medium, low
    importance_score: int = 5      # 1-10
    context: str = ""              # file/page/conversation open
    suggested_next: str = ""
    screen_text_preview: str = ""  # First 200 chars of OCR text
    duration_seconds: float = 0.0


class ActivityJournal:
    """
    Persistent activity log that records what the user does.
    Saves to ~/.magnum_activity_journal.jsonl (append-only, JSONL format).
    Supports natural language recall via AI reasoning.
    """

    def __init__(self) -> None:
        self._entries_cache: List[ActivityEntry] = []
        self._last_activity: Optional[ActivityEntry] = None
        self._last_log_time: float = 0.0
        self._session_start: float = time.time()

    def log_activity(self, entry: ActivityEntry) -> None:
        """Append an activity entry to the journal."""
        if not entry.timestamp:
            entry.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not entry.epoch:
            entry.epoch = time.time()

        # Calculate duration since last entry
        if self._last_activity and self._last_activity.epoch:
            entry.duration_seconds = round(entry.epoch - self._last_activity.epoch, 1)

        # Don't log duplicate activities (same app + same summary within 30s)
        if self._last_activity:
            if (
                entry.app_name == self._last_activity.app_name
                and entry.activity_summary == self._last_activity.activity_summary
                and (entry.epoch - self._last_activity.epoch) < 30.0
            ):
                return

        self._last_activity = entry
        self._last_log_time = entry.epoch
        self._entries_cache.append(entry)

        # Keep cache reasonable
        if len(self._entries_cache) > 500:
            self._entries_cache = self._entries_cache[-500:]

        # Write to disk
        try:
            with open(JOURNAL_FILE, "a", encoding="utf-8") as f:
                f.write(entry.model_dump_json() + "\n")
        except Exception as e:
            logger.warning(f"Could not write to activity journal: {e}")

    def get_recent(self, hours: int = 24) -> List[ActivityEntry]:
        """Get activity entries from the last N hours."""
        cutoff = time.time() - (hours * 3600)

        # Try loading from cache first
        cached = [e for e in self._entries_cache if e.epoch >= cutoff]
        if cached:
            return cached

        # Load from disk
        return self._load_from_disk(cutoff_epoch=cutoff)

    def get_by_importance(self, min_level: str = "high") -> List[ActivityEntry]:
        """Get entries filtered by minimum importance level."""
        level_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        min_val = level_order.get(min_level, 2)

        entries = self.get_recent(hours=24)
        return [e for e in entries if level_order.get(e.importance, 2) >= min_val]

    def get_journal_text(self, hours: int = 24, max_entries: int = 50) -> str:
        """Get a formatted text representation of recent journal entries for AI recall."""
        entries = self.get_recent(hours=hours)
        if not entries:
            return "(No activity recorded in the last {hours} hours)"

        # Take the most recent entries, up to max_entries
        entries = entries[-max_entries:]

        lines = []
        for e in entries:
            duration_str = f" ({e.duration_seconds:.0f}s)" if e.duration_seconds > 0 else ""
            importance_emoji = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🟢",
            }.get(e.importance, "⚪")

            line = (
                f"[{e.timestamp}] {importance_emoji} {e.importance.upper()}{duration_str} "
                f"| {e.app_name}: {e.activity_summary}"
            )
            if e.context:
                line += f" ({e.context})"
            lines.append(line)

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get activity statistics for the current session."""
        entries = self.get_recent(hours=24)
        if not entries:
            return {"total_entries": 0, "session_duration": 0}

        app_counts: Dict[str, int] = {}
        importance_counts: Dict[str, int] = {}
        total_duration = 0.0

        for e in entries:
            app_counts[e.app_name] = app_counts.get(e.app_name, 0) + 1
            importance_counts[e.importance] = importance_counts.get(e.importance, 0) + 1
            total_duration += e.duration_seconds

        top_apps = sorted(app_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_entries": len(entries),
            "session_duration": round(time.time() - self._session_start),
            "top_apps": top_apps,
            "importance_breakdown": importance_counts,
            "total_tracked_time": round(total_duration),
        }

    def rotate(self) -> int:
        """Remove entries older than JOURNAL_MAX_AGE_DAYS. Returns count removed."""
        if not JOURNAL_FILE.exists():
            return 0

        cutoff = time.time() - (JOURNAL_MAX_AGE_DAYS * 86400)
        kept: List[str] = []
        removed = 0

        try:
            with open(JOURNAL_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if data.get("epoch", 0) >= cutoff:
                            kept.append(line)
                        else:
                            removed += 1
                    except json.JSONDecodeError:
                        kept.append(line)  # Keep unparseable lines

            if removed > 0:
                with open(JOURNAL_FILE, "w", encoding="utf-8") as f:
                    f.write("\n".join(kept) + "\n" if kept else "")
                logger.info(f"Activity journal rotated: removed {removed} old entries")

        except Exception as e:
            logger.warning(f"Journal rotation failed: {e}")

        return removed

    def _load_from_disk(self, cutoff_epoch: float = 0.0) -> List[ActivityEntry]:
        """Load entries from the JSONL file."""
        entries: List[ActivityEntry] = []

        if not JOURNAL_FILE.exists():
            return entries

        try:
            with open(JOURNAL_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        entry = ActivityEntry(**data)
                        if entry.epoch >= cutoff_epoch:
                            entries.append(entry)
                    except (json.JSONDecodeError, Exception):
                        continue
        except Exception as e:
            logger.warning(f"Could not read activity journal: {e}")

        return entries


# Singleton
_journal_instance: Optional[ActivityJournal] = None


def get_activity_journal() -> ActivityJournal:
    """Get the global ActivityJournal singleton."""
    global _journal_instance
    if _journal_instance is None:
        _journal_instance = ActivityJournal()
    return _journal_instance
