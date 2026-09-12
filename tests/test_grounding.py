import pytest
from PIL import Image, ImageDraw
from autonavigator.intelligence.grounding import (
    ScreenGrounder,
    BoundingBox,
    NormalizedPoint,
    TextElement,
)


def test_scale_point_identity():
    pt = {"x": 500, "y": 300}
    scaled = ScreenGrounder.scale_point(pt, 1000, 600, 1000, 600)
    assert scaled.x == 500
    assert scaled.y == 300


def test_scale_point_downscaling():
    # 2x Retina downscale from 2560x1600 to 1280x800
    pt = {"x": 1000, "y": 600}
    scaled = ScreenGrounder.scale_point(pt, 2560, 1600, 1280, 800)
    assert scaled.x == 500.0
    assert scaled.y == 300.0


def test_bounding_box_center():
    box = BoundingBox(id=1, label="Search Bar", left=100, top=50, right=300, bottom=90)
    center = box.center
    assert center.x == 200.0
    assert center.y == 70.0


def test_annotate_set_of_marks():
    img = Image.new("RGB", (400, 300), color="white")
    boxes = [
        BoundingBox(id=1, label="Button 1", left=50, top=50, right=150, bottom=90),
        BoundingBox(id=2, label="Button 2", left=200, top=50, right=300, bottom=90),
    ]
    annotated = ScreenGrounder.annotate_set_of_marks(img, boxes)
    assert annotated.size == (400, 300)
    assert annotated != img


def test_apple_vision_ocr_text_extraction():
    # Create an image with rendered text
    img = Image.new("RGB", (600, 200), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "File", fill="black")
    draw.text((200, 50), "Open Folder", fill="black")

    coords = ScreenGrounder.find_element_by_text(img, "File")
    assert coords is not None
    assert len(coords) == 2
    assert coords[0] > 0
    assert coords[1] > 0


def test_find_element_by_text_fuzzy():
    img = Image.new("RGB", (600, 200), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((100, 80), "college_python_scraper", fill="black")

    coords = ScreenGrounder.find_element_by_text(img, "college python scrapper")
    assert coords is not None
    assert coords[0] > 0
