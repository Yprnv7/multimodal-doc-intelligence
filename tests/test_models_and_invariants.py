"""Tests and property-based invariants for bounding boxes and spatial models."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.parser.models import BoundingBox


def test_bounding_box_valid() -> None:
    bbox = BoundingBox(ymin=100, xmin=150, ymax=300, xmax=450)
    assert bbox.width == 300
    assert bbox.height == 200
    assert bbox.area == 60000


def test_bounding_box_invalid_raises() -> None:
    with pytest.raises(ValueError, match="ymin .* must be strictly less than ymax"):
        BoundingBox(ymin=300, xmin=100, ymax=200, xmax=400)

    with pytest.raises(ValueError, match="xmin .* must be strictly less than xmax"):
        BoundingBox(ymin=100, xmin=500, ymax=300, xmax=200)


def test_bounding_box_intersection_and_iou() -> None:
    box_a = BoundingBox(ymin=100, xmin=100, ymax=300, xmax=300)
    box_b = BoundingBox(ymin=200, xmin=200, ymax=400, xmax=400)

    inter = box_a.intersection(box_b)
    assert inter is not None
    assert inter.ymin == 200
    assert inter.xmin == 200
    assert inter.ymax == 300
    assert inter.xmax == 300
    assert inter.area == 10000

    iou = box_a.iou(box_b)
    # Area A = 40000, Area B = 40000, Inter = 10000, Union = 70000 => 10000 / 70000 = 1/7
    assert round(iou, 4) == round(1.0 / 7.0, 4)


def test_bounding_box_disjoint_returns_none() -> None:
    box_a = BoundingBox(ymin=100, xmin=100, ymax=200, xmax=200)
    box_b = BoundingBox(ymin=500, xmin=500, ymax=600, xmax=600)
    assert box_a.intersection(box_b) is None
    assert box_a.iou(box_b) == 0.0


@given(
    ymin=st.integers(min_value=0, max_value=499),
    xmin=st.integers(min_value=0, max_value=499),
    delta_y=st.integers(min_value=1, max_value=500),
    delta_x=st.integers(min_value=1, max_value=500),
)
def test_hypothesis_bounding_box_invariants(ymin: int, xmin: int, delta_y: int, delta_x: int) -> None:
    ymax = ymin + delta_y
    xmax = xmin + delta_x
    bbox = BoundingBox(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax)

    assert bbox.ymin < bbox.ymax
    assert bbox.xmin < bbox.xmax
    assert bbox.width > 0
    assert bbox.height > 0
    assert bbox.area > 0
    # Reflexive IoU must be 1.0
    assert round(bbox.iou(bbox), 4) == 1.0000
