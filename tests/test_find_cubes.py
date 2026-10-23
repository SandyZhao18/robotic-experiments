import math

import numpy as np
import pytest

from robotic_experiments.blocks.find_cubes import Found, square_rotation


@pytest.mark.parametrize("angle", [0.0, 7.0, -12.0, 30.0, 44.0, 90.0, -80.0])
def test_square_rotation_is_folded_into_plus_minus_45(angle):
    t = math.radians(angle)
    corners = np.array([[math.cos(t) * x - math.sin(t) * y, math.sin(t) * x + math.cos(t) * y]
                        for x, y in ((-15, -15), (15, -15), (15, 15), (-15, 15))])
    edges = np.vstack([np.linspace(a, b, 20) for a, b in zip(corners, np.roll(corners, -1, axis=0))])
    rot, (w, h) = square_rotation(edges)
    expected = (angle + 45.0) % 90.0 - 45.0
    assert rot == pytest.approx(expected, abs=0.5)
    assert (w, h) == pytest.approx((30.0, 30.0), abs=0.5)


def test_found_reports_nearest_square_and_offset():
    c = Found("red", f=0.05, r=2.1, rotation_deg=0.0, top_size_mm=(30, 30), top_px=(0, 0))
    assert c.square == "a3"
    assert c.offset_mm(57.0) == pytest.approx((2.85, 5.7))
    assert Found("red", -0.8, 0.0, 0.0, (30, 30), (0, 0)).square is None


def test_usable_checks_supply_area_reach_and_twist():
    import dataclasses

    from robotic_experiments.blocks import workspace

    ws = dataclasses.replace(workspace.load("tests/workspace_fixture.toml"), supply_files="fgh", max_reach_mm=10_000.0)
    ok = Found("red", f=6.0, r=1.0, rotation_deg=5.0, top_size_mm=(30, 30), top_px=(0, 0))  # g2
    assert ok.usable(ws) is None
    assert "supply" in Found("red", 2.0, 1.0, 0.0, (30, 30), (0, 0)).usable(ws)  # c2
    assert "twisted" in Found("red", 6.0, 1.0, 35.0, (30, 30), (0, 0)).usable(ws)
    assert "reach" in ok.usable(dataclasses.replace(ws, max_reach_mm=100.0))
    assert "off the mat" in Found("red", 8.6, 1.0, 0.0, (30, 30), (0, 0)).usable(ws)
