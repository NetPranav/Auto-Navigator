"""Tests for perception caching and screen diff calculation."""

from PIL import Image
from magnum.task_queue import compute_screen_diff


def test_compute_screen_diff_identical():
    img1 = Image.new("RGB", (400, 300), color=(100, 150, 200))
    img2 = Image.new("RGB", (400, 300), color=(100, 150, 200))
    diff = compute_screen_diff(img1, img2)
    assert diff == 0.0


def test_compute_screen_diff_different():
    img1 = Image.new("RGB", (400, 300), color=(0, 0, 0))
    img2 = Image.new("RGB", (400, 300), color=(255, 255, 255))
    diff = compute_screen_diff(img1, img2)
    assert diff > 0.9
