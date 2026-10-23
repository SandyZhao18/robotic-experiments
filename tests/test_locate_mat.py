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
