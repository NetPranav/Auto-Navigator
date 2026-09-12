import pytest
from magnum.agent import MagnumAgent


def test_conversational_watcher_parsing():
    agent = MagnumAgent.__new__(MagnumAgent)
    agent.config = type("Config", (), {"watcher_poll_interval": 5.0})()

    long_prompt = (
        "in the right side of the anti gravity there is a submit button it will come "
        "in some frequency again and again and again you are supposed to click on the "
        "submit button as soon as you find it it will not be once it would be multiple "
        "times so click on it until i tell you to stop it"
    )

    config = agent._parse_watcher_instruction(long_prompt)
    assert config is not None
    assert config["watch_for"] == "Submit"
    assert config["action"] == "CLICK"
    assert config["stop_when"] is None  # Runs continuously until user stops


def test_combined_watcher_phrase():
    agent = MagnumAgent.__new__(MagnumAgent)
    agent.config = type("Config", (), {"watcher_poll_interval": 5.0})()

    prompt = "whenever the submit buttons come on the right bottom click on it until i tell you to stop"
    config = agent._parse_watcher_instruction(prompt)
    assert config is not None
    assert config["watch_for"] == "Submit"
    assert config["action"] == "CLICK"
    assert config["stop_when"] is None


def test_user_actual_voice_prompt():
    agent = MagnumAgent.__new__(MagnumAgent)
    agent.config = type("Config", (), {"watcher_poll_interval": 5.0})()

    # Exact transcript that previously produced "Again And Again" and stop_when="complete"
    prompt = "in the right side of the integrity there is a submit button click on it again and again whenever it appears until i tell you to stop"
    config = agent._parse_watcher_instruction(prompt)
    assert config is not None
    assert config["watch_for"] == "Submit"
    assert config["action"] == "CLICK"
    assert config["stop_when"] is None  # Must NOT be 'complete'!


def test_turn_watcher_on_phrase():
    agent = MagnumAgent.__new__(MagnumAgent)
    agent.config = type("Config", (), {"watcher_poll_interval": 5.0})()

    prompt = "turn the watcher on to click on the submit button again and again until I tell you to stop"
    config = agent._parse_watcher_instruction(prompt)
    assert config is not None
    assert config["watch_for"] == "Submit"
    assert config["action"] == "CLICK"
    assert config["stop_when"] is None


def test_watcher_spatial_region_parsing():
    agent = MagnumAgent.__new__(MagnumAgent)
    agent.config = type("Config", (), {"watcher_poll_interval": 5.0})()

    prompt = "start a watcher to look for the submit button in bottom right corner of antigravity unitl i tell it to stop"
    config = agent._parse_watcher_instruction(prompt)
    assert config is not None
    assert config["watch_for"] == "Submit"
    assert config["action"] == "CLICK"
    assert config["stop_when"] is None
    assert config["region"] == "bottom_right"


@pytest.mark.asyncio
async def test_stop_watcher_natural_language_commands():
    from unittest.mock import MagicMock
    agent = MagnumAgent.__new__(MagnumAgent)
    agent.task_queue = MagicMock()
    agent.overlay = MagicMock()
    agent.voice_engine = MagicMock()

    # When active watchers exist
    agent.task_queue.get_active_watchers.return_value = [MagicMock()]

    for stop_phrase in ("stop the watcher", "please stop watcher", "turn off the watcher", "cancel watchers", "kill the watcher", "stop"):
        processed = await agent.process_instruction(stop_phrase)
        assert processed is True, f"Failed to stop watcher for: {stop_phrase}"
        assert agent.task_queue.cancel_all_watchers.called
