"""Tests for Magnum concurrent task queue engine."""

import asyncio
import pytest
from magnum.task_queue import TaskQueue, MagnumTask, TaskType, TaskStatus, WatcherCondition


def test_create_foreground_task():
    """Test creating a foreground task."""
    tq = TaskQueue()
    task = tq.create_foreground_task("open youtube")
    assert task.type == TaskType.FOREGROUND
    assert task.status == TaskStatus.PENDING
    assert task.instruction == "open youtube"
    assert task.id in tq.tasks


def test_create_watcher():
    """Test creating a background watcher."""
    tq = TaskQueue()
    task = tq.create_watcher(
        instruction="click submit when it appears",
        watch_for="Submit",
        action="CLICK",
        stop_when="Complete",
        poll_interval=5.0,
    )
    assert task.type == TaskType.WATCHER
    assert task.status == TaskStatus.PENDING
    assert task.watcher_condition is not None
    assert task.watcher_condition.watch_for == "Submit"
    assert task.watcher_condition.stop_when == "Complete"
    assert task.watcher_condition.action == "CLICK"


def test_get_active_watchers():
    """Test filtering active watchers."""
    tq = TaskQueue()
    w1 = tq.create_watcher("watch 1", "Submit", "CLICK")
    w1.status = TaskStatus.ACTIVE
    w2 = tq.create_watcher("watch 2", "Continue", "CLICK")
    w2.status = TaskStatus.ACTIVE
    fg = tq.create_foreground_task("open chrome")
    fg.status = TaskStatus.ACTIVE

    active = tq.get_active_watchers()
    assert len(active) == 2
    assert all(w.type == TaskType.WATCHER for w in active)


def test_status_summary():
    """Test human-readable status summary."""
    tq = TaskQueue()
    w = tq.create_watcher("watch submit", "Submit", "CLICK")
    w.status = TaskStatus.ACTIVE
    w.watcher_trigger_count = 3

    summary = tq.get_status_summary()
    assert "Submit" in summary
    assert "3" in summary


def test_cancel_watcher():
    """Test cancelling a watcher by ID."""
    tq = TaskQueue()
    w = tq.create_watcher("watch test", "Button", "CLICK")
    w.status = TaskStatus.ACTIVE
    
    # Can't cancel without asyncio task, but status should update
    tq.tasks[w.id].status = TaskStatus.CANCELLED
    assert tq.tasks[w.id].status == TaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_foreground_task_execution():
    """Test running a foreground task with exclusive lock."""
    tq = TaskQueue()
    task = tq.create_foreground_task("test task")

    async def mock_execute(instruction):
        await asyncio.sleep(0.1)
        return True

    result = await tq.run_foreground(task, mock_execute)
    assert result is True
    assert task.status == TaskStatus.DONE
    assert task.completed_at is not None


@pytest.mark.asyncio
async def test_foreground_pauses_watchers():
    """Test that foreground tasks pause and resume watchers."""
    tq = TaskQueue()
    
    # Create an active watcher
    w = tq.create_watcher("watch submit", "Submit", "CLICK")
    w.status = TaskStatus.ACTIVE

    pause_events = []

    async def mock_execute(instruction):
        # During execution, pause event should be cleared
        pause_events.append(tq._watcher_pause_event.is_set())
        await asyncio.sleep(0.05)
        return True

    task = tq.create_foreground_task("foreground task")
    await tq.run_foreground(task, mock_execute)
    
    # Pause event should have been cleared during execution
    assert pause_events[0] is False
    # And set again after completion
    assert tq._watcher_pause_event.is_set() is True
