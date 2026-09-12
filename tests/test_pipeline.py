"""Tests for PipelineManager and Phased Pipeline Autopilot."""

import tempfile
from pathlib import Path
from magnum.pipeline.manager import PipelineManager, ProjectPipeline, PipelineStep


def test_parse_phases_text_headers():
    text = """
    Phase 1: Setup database and models.
    Create schema and connect SQLite.

    Phase 2: Add REST API routes.
    Add endpoints for users and posts.

    Phase 3: Write automated unit tests.
    Ensure 100% test coverage.
    """
    phases = PipelineManager.parse_phases_text(text)
    assert len(phases) == 3
    assert "Setup database" in phases[0]
    assert "Add REST API" in phases[1]
    assert "Write automated unit tests" in phases[2]


def test_parse_phases_text_dividers():
    text = """
    First, build the UI layout in vanilla HTML and CSS.
    ---
    Second, wire JavaScript interactions and event listeners.
    ---
    Third, verify responsiveness on mobile and desktop.
    """
    phases = PipelineManager.parse_phases_text(text)
    assert len(phases) == 3
    assert "build the UI layout" in phases[0]
    assert "wire JavaScript" in phases[1]
    assert "verify responsiveness" in phases[2]


def test_parse_phases_file():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as tf:
        tf.write("Phase 1: Step A\n\nPhase 2: Step B")
        tf_path = Path(tf.name)

    try:
        phases = PipelineManager.parse_phases_file(tf_path)
        assert len(phases) == 2
        assert "Step A" in phases[0]
        assert "Step B" in phases[1]
    finally:
        if tf_path.exists():
            tf_path.unlink()
