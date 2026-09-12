"""Tests for QuickLauncher global hotkey module."""

from magnum.ui.quick_launcher import QuickLauncher, get_quick_launcher


def test_quick_launcher_instance():
    ql = QuickLauncher()
    assert ql.on_command is None
    assert ql._active is False


def test_quick_launcher_singleton():
    def dummy_callback(cmd):
        pass

    ql1 = get_quick_launcher(on_command=dummy_callback)
    ql2 = get_quick_launcher()
    assert ql1 is ql2
    assert ql1.on_command == dummy_callback
