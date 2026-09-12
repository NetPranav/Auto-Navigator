import pytest
from autonavigator.config import AppConfig


def test_app_config_defaults():
    cfg = AppConfig()
    assert cfg.default_execution_mode in ("headless", "desktop")
    assert cfg.viewport_width == 1280
    assert cfg.viewport_height == 800
    assert cfg.max_steps == 30
    assert "nvidia.com" in cfg.nvidia_base_url


def test_app_config_ensure_directories(tmp_path):
    cfg = AppConfig(
        browser_profile_path=tmp_path / "browser_profile",
        screenshots_dir=tmp_path / "screenshots",
        save_debug_screenshots=True,
    )
    cfg.ensure_directories()
    assert cfg.browser_profile_path.exists()
    assert cfg.screenshots_dir.exists()
