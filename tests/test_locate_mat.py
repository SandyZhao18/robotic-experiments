import math

import numpy as np
import pytest

from robotic_experiments.blocks.locate_mat import LocateError, fit_board

SQUARE = 57.0
A1 = np.array([300.0, 200.0])
MAT_Z = -38.0


def board_corners(yaw_deg: float, files=range(7), ranks=range(7)) -> np.ndarray:
    """Inner corners (rows=ranks, cols=files, 3) of a board with a1 at A1, files at yaw_deg."""
    t = math.radians(yaw_deg)
    fs = SQUARE * np.array([math.cos(t), math.sin(t)])
    rs = SQUARE * np.array([-math.sin(t), math.cos(t)])  # 90° CCW from files
    c00 = A1 + 0.5 * fs + 0.5 * rs
    return np.array([[[*(c00 + i * fs + j * rs), MAT_Z] for i in files] for j in ranks])


def block_on_a1() -> np.ndarray:
    return np.array([*A1, MAT_Z + 30.0])


@pytest.mark.parametrize("yaw", [0.0, 90.0, -90.0, 180.0, -33.0])
def test_fit_recovers_board_in_any_orientation(yaw):
    fit = fit_board(board_corners(yaw), block_on_a1())
    assert fit.a1 == pytest.approx(tuple(A1))
    assert math.dist(fit.h1, fit.a1) == pytest.approx(7 * SQUARE)
    assert fit.file_yaw_deg == pytest.approx(yaw if yaw != -180 else 180)


@pytest.mark.parametrize("flip_rows,flip_cols,transpose", [(1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1)])
def test_fit_ignores_detection_order(flip_rows, flip_cols, transpose):
    lat = board_corners(20.0)
    if flip_rows:
        lat = lat[::-1]
    if flip_cols:
        lat = lat[:, ::-1]
    if transpose:
        lat = lat.transpose(1, 0, 2)
    fit = fit_board(lat, block_on_a1())
    assert fit.a1 == pytest.approx(tuple(A1))
    assert fit.file_yaw_deg == pytest.approx(20.0)


def test_partial_view_containing_a1_corner():
    fit = fit_board(board_corners(0.0, files=range(5), ranks=range(7)), block_on_a1())
    assert fit.a8 == pytest.approx((A1[0], A1[1] + 7 * SQUARE))


def test_refuses_when_a1_corner_not_in_view():
    # detected patch starts one corner in, so the block is a full square from the computed a1
    lat = board_corners(0.0, files=range(1, 6), ranks=range(7))
    with pytest.raises(LocateError, match="not on a1"):
        fit_board(lat, block_on_a1())


def test_refuses_noisy_fit():
    lat = board_corners(0.0)
    lat[3, 3, :2] += 15.0
    with pytest.raises(LocateError, match="residual"):
        fit_board(lat, block_on_a1())


def test_block_top_ignores_visible_side_face():
    from robotic_experiments.blocks.locate_mat import find_block_top, find_red_block

    bgr = np.full((200, 200, 3), (120, 160, 200), np.uint8)  # mat-ish, not saturated red
    depth = np.full((200, 200), 450.0)
    bgr[80:140, 80:140] = (0, 0, 230)  # top face: 60x60 px, centered at (109.5, 109.5)
    depth[80:140, 80:140] = 420.0
    bgr[80:140, 60:80] = (0, 0, 200)  # side face seen at an angle, farther away
    depth[80:140, 60:80] = np.linspace(422.0, 448.0, 20)[None, :][:, ::-1]
    (u, v), d = find_block_top(bgr, depth)
    # The side face's strip nearest the top edge is within the depth tolerance, so a sliver
    # (~2 px of a 60 px face, ~1 mm on the real cube) still counts; the whole blob is ~10 px off.
    assert (u, v) == pytest.approx((109.5, 109.5), abs=2.5)
    assert d == pytest.approx(420.0)
    assert find_red_block(bgr)[0] < 105  # the whole blob is pulled toward the side face


def test_yellow_and_red_blocks_are_told_apart():
    from robotic_experiments.blocks.locate_mat import block_mask

    bgr = np.full((200, 200, 3), (150, 170, 190), np.uint8)  # light wood-ish
    bgr[20:60, 20:60] = (40, 40, 200)  # red block
    bgr[120:170, 120:170] = (40, 200, 220)  # yellow block (BGR)
    red, yellow = block_mask(bgr, "red"), block_mask(bgr, "yellow")
    assert red[40, 40] and not red[140, 140]
    assert yellow[140, 140] and not yellow[40, 40]


def test_red_and_orange_named_by_median_hue():
    import cv2

    from robotic_experiments.blocks.locate_mat import block_blobs

    hsv = np.zeros((200, 300, 3), np.uint8)
    hsv[...] = (11, 40, 150)  # light wood square: reddish but dull
    hsv[20:80, 20:80] = (1, 210, 140)  # red cube
    hsv[20:80, 120:180] = (6, 160, 160)  # orange cube
    hsv[120:180, 120:180] = (85, 240, 120)  # green cube
    hsv[120:180, 220:280] = (98, 250, 130)  # blue tape: must not count as green
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    found = {c: tuple(np.argwhere(m).mean(0).round()) for c, m in block_blobs(bgr, ["red", "orange", "green", "yellow"])}
    assert found == {"red": (49.0, 49.0), "orange": (49.0, 149.0), "green": (149.0, 149.0)}
