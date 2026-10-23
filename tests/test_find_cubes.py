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


def test_usable_checks_supply_area_and_reach():
    import dataclasses

    from robotic_experiments.blocks import workspace

    ws = dataclasses.replace(workspace.load("tests/workspace_fixture.toml"), supply_files="fgh", max_reach_mm=10_000.0)
    ok = Found("red", f=6.0, r=1.0, rotation_deg=5.0, top_size_mm=(30, 30), top_px=(0, 0))  # g2
    assert ok.usable(ws) is None
    assert "supply" in Found("red", 2.0, 1.0, 0.0, (30, 30), (0, 0)).usable(ws)  # c2
    assert Found("red", 6.0, 1.0, 35.0, (30, 30), (0, 0)).usable(ws) is None  # the wrist turns to match
    assert "reach" in ok.usable(dataclasses.replace(ws, max_reach_mm=100.0))
    assert "off the mat" in Found("red", 8.6, 1.0, 0.0, (30, 30), (0, 0)).usable(ws)


def test_fragments_of_one_cube_are_merged():
    from robotic_experiments.blocks import workspace
    from robotic_experiments.blocks.find_cubes import merge_fragments

    ws = workspace.load("tests/workspace_fixture.toml")  # 40 mm blocks, 57 mm squares
    cube = Found("yellow", 7.0, 2.0, 21.0, (40, 38), (0, 0))
    side = Found("orange", 6.9, 2.3, 23.0, (11, 32), (0, 0))  # sliver: too narrow
    shadow = Found("orange", 7.2, 2.1, 24.0, (30, 28), (0, 0))  # whole-sized but on top of the cube
    other = Found("red", 5.0, 0.0, 0.0, (39, 40), (0, 0))
    kept = merge_fragments([side, shadow, cube, other], ws)
    assert sorted(kept, key=lambda c: c.color) == [other, cube]


@pytest.mark.parametrize("angle", [0.0, 8.0, -15.0, 25.0, -40.0])
def test_edge_rotation_from_a_drawn_square(angle):
    import cv2

    from robotic_experiments.blocks.find_cubes import edge_rotation

    img = np.full((200, 200), 60, np.uint8)
    box = cv2.boxPoints(((100.0, 100.0), (40.0, 40.0), angle)).astype(np.int32)
    cv2.fillPoly(img, [box], 220)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [box], 1)
    # image y points down; mat ranks/files here are the image axes with y flipped
    rot = edge_rotation(img, mask.astype(bool), np.array([[1.0, 0.0], [0.0, -1.0]]))
    assert rot == pytest.approx((-angle + 45.0) % 90.0 - 45.0, abs=1.5)


def test_oversized_detection_is_not_a_cube():
    from robotic_experiments.blocks import workspace
    from robotic_experiments.blocks.find_cubes import merge_fragments

    ws = workspace.load("tests/workspace_fixture.toml")  # 40 mm blocks
    hand = Found("orange", 7.0, 0.0, 36.0, (54, 122), (0, 0))
    cube = Found("red", 5.0, 0.0, 0.0, (39, 40), (0, 0))
    assert merge_fragments([hand, cube], ws) == [cube]


def test_only_cube_height_counts_as_a_block():
    from robotic_experiments.blocks.locate_mat import Block

    mk = lambda h: Block("orange", (0, 0), np.zeros((2, 2), bool), np.zeros(3), h)
    assert mk(30.0).cube_height
    assert not mk(1.0).cube_height  # a light mat square under warm light
    assert not mk(150.0).cube_height  # a hand reaching in


def test_classify_shapes_by_top_and_height():
    from robotic_experiments.blocks.find_cubes import classify, long_axis

    assert classify((29, 27), 25) == "cube"
    assert classify((57, 27), 24) == "brick"
    assert classify((25, 24), 54) == "cylinder"
    assert classify((11, 32), 25) is None  # a sliver of side face
    assert classify((54, 122), 30) is None  # a hand
    pts = np.array([[x, y] for x in range(-30, 31, 5) for y in (-15, 15)], float)
    assert long_axis(pts) == "files" and long_axis(pts[:, ::-1]) == "ranks"


def test_shape_comes_from_the_blob_not_a_clipped_top_face():
    from robotic_experiments.blocks.find_cubes import classify

    # a brick far away: depth noise clipped its top face to 23x19, but the blob is the whole brick
    assert classify((27, 60), 22) == "brick"
    # a cube's blob with its visible sides is still well under a brick's length
    assert classify((34, 38), 25) == "cube"
    # a tall blob as wide as a brick is not a cylinder
    assert classify((55, 60), 55) is None
