import pytest
from autonavigator.intelligence.planner import PlanChecklist, PlanStep


def test_plan_checklist_lifecycle():
    steps = [
        PlanStep(step_index=1, title="Open Browser", description="Open Google Chrome", status="active"),
        PlanStep(step_index=2, title="Search LinkedIn", description="Search for Yash Rai"),
        PlanStep(step_index=3, title="Post Comment", description="Post the approved comment"),
    ]
    checklist = PlanChecklist(goal_summary="Comment on LinkedIn", steps=steps, active_index=1)

    assert checklist.current_step.title == "Open Browser"
    assert checklist.is_finished is False
    assert len(checklist.get_titles_list()) == 3

    # Advance to step 2
    checklist.advance_to_next_step()
    assert checklist.current_step.title == "Search LinkedIn"
    assert checklist.steps[0].status == "completed"
    assert checklist.steps[1].status == "active"

    # Advance to step 3
    checklist.advance_to_next_step()
    assert checklist.current_step.title == "Post Comment"

    # Finish
    checklist.advance_to_next_step()
    assert checklist.is_finished is True
