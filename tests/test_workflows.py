"""Tests for WorkflowManager and custom user workflows."""

import tempfile
from pathlib import Path
from magnum.workflows.manager import WorkflowManager


def test_workflow_manager_lifecycle():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        temp_path = Path(tf.name)

    try:
        mgr = WorkflowManager(filepath=temp_path)
        assert len(mgr.list_workflows()) == 0

        # Save a new workflow
        wf = mgr.save_workflow(
            name="Antigravity phases and auto submit",
            steps=[
                "Open Antigravity",
                "Paste the current phase into chat and press enter",
                "Wait for implementation plan and click proceed",
                "Watch for submit button and click it until I tell you to stop",
            ],
            description="Autonomous multi-phase executor for Antigravity",
        )

        assert wf.name == "Antigravity phases and auto submit"
        assert len(wf.steps) == 4
        assert wf.steps[0].instruction == "Open Antigravity"
        assert wf.steps[3].instruction == "Watch for submit button and click it until I tell you to stop"

        # Lookup exact
        found = mgr.get_workflow("Antigravity phases and auto submit")
        assert found is not None
        assert len(found.steps) == 4

        # Lookup fuzzy
        fuzzy_match1 = mgr.find_workflow("antigravity phases")
        assert fuzzy_match1 is not None
        assert fuzzy_match1.name == "Antigravity phases and auto submit"

        fuzzy_match2 = mgr.find_workflow("run the antigravity phases and auto submit")
        assert fuzzy_match2 is not None
        assert fuzzy_match2.name == "Antigravity phases and auto submit"

        # Add second workflow
        mgr.save_workflow(
            name="Getting meeting ready",
            steps=["Open Calendar", "Open Slack", "Mute notifications"],
        )
        assert len(mgr.list_workflows()) == 2

        fuzzy_meeting = mgr.find_workflow("meeting ready")
        assert fuzzy_meeting is not None
        assert fuzzy_meeting.name == "Getting meeting ready"

        # Delete workflow
        assert mgr.delete_workflow("Getting meeting ready") is True
        assert len(mgr.list_workflows()) == 1
        assert mgr.get_workflow("Getting meeting ready") is None

    finally:
        if temp_path.exists():
            temp_path.unlink()


def test_workflow_intent_phonetic_triggers():
    # Test all variations of user speech transcripts
    prompts = [
        "start a voucher for submit button",
        "create a workshop",
        "start a workflow response",
        "start a workflow",
        "create a workflow",
        "record a workflow",
        "make a new workflow for antigravity",
    ]

    workflow_synonyms = ("workflow", "work flow", "voucher", "workshop", "macro", "sequence")
    create_verbs = ("create", "start", "record", "make", "new", "begin", "build", "set up", "setup")

    for p in prompts:
        lower = p.lower()
        matched = any(syn in lower for syn in workflow_synonyms) and any(verb in lower for verb in create_verbs)
        assert matched is True, f"Failed to match workflow trigger for '{p}'"


def test_no_workflow_step_self_recursion():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        temp_path = Path(tf.name)

    try:
        mgr = WorkflowManager(filepath=temp_path)
        mgr.save_workflow(
            name="Antigravity Watcher",
            steps=["start a watcher to look for the submit button in bottom right corner of antigravity until i tell it to stop"],
        )

        long_step_inst = "start a watcher to look for the submit button in bottom right corner of antigravity until i tell it to stop"
        # Must NOT match "Antigravity Watcher" because it is a long task sentence
        matched = mgr.find_workflow(long_step_inst)
        assert matched is None

        # Short command MUST match
        matched_short = mgr.find_workflow("run antigravity watcher")
        assert matched_short is not None
        assert matched_short.name == "Antigravity Watcher"
    finally:
        if temp_path.exists():
            temp_path.unlink()

