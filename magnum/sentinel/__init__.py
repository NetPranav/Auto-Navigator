"""
Magnum Sentinel — Universal Intelligent App Monitor, Suggestion Engine & Activity Journal.

The Sentinel system adds three major capabilities to Magnum:
1. AppSentinel — Watch any app (Gmail, WhatsApp, Slack, etc.) with AI-driven content rules
2. SuggestionEngine — Observe user activity and proactively suggest next tasks
3. ActivityJournal — Persistent activity log with AI-powered recall

Usage:
    from magnum.sentinel import AppSentinel, SuggestionEngine, ActivityJournal
"""

from magnum.sentinel.app_monitor import AppSentinel, AppMonitorRule, MonitorMatch
from magnum.sentinel.content_analyzer import ContentAnalyzer, ContentMatch
from magnum.sentinel.suggestion_engine import SuggestionEngine, Suggestion
from magnum.sentinel.activity_journal import ActivityJournal, ActivityEntry

__all__ = [
    "AppSentinel",
    "AppMonitorRule",
    "MonitorMatch",
    "ContentAnalyzer",
    "ContentMatch",
    "SuggestionEngine",
    "Suggestion",
    "ActivityJournal",
    "ActivityEntry",
]
