import dataclasses

import numpy as np
import pytest

from robotic_experiments.blocks import workspace
from robotic_experiments.blocks.check_placement import measure
from robotic_experiments.blocks.locate_mat import LocateError

WS = workspace.load("tests/workspace_fixture.toml")


def lattice(ws, files, ranks, shift=(0.0, 0.0)):
    """Inner corners for the given corner indices, as the camera would see them."""
    a1 = np.array(ws.a1) + shift
    fs, rs = np.array(ws.file_step), np.array(ws.rank_step)
    return np.array([[[*(a1 + (f + 0.5) * fs + (r + 0.5) * rs), -35.0] for f in files] for r in ranks])


def test_block_on_center_reads_zero():
    lat = lattice(WS, range(2, 6), range(1, 6))  # a1 not in view
    p = measure(lat, WS.measured_square_xy("e5"), WS, "e5")
    assert p.distance_mm == pytest.approx(0.0, abs=1e-6)


def test_offset_is_reported_along_files_and_ranks():
    lat = lattice(WS, range(0, 5), range(0, 4))
    x, y = WS.measured_square_xy("a3")
    fu, ru = np.array(WS.file_unit), np.array(WS.rank_unit)
    block = np.array([x, y]) + 5.0 * fu - 22.0 * ru
    p = measure(lat, block, WS, "a3")
    assert (p.along_files_mm, p.along_ranks_mm) == pytest.approx((5.0, -22.0))


def test_offset_is_measured_against_the_mat_in_view():
    # the whole view is shifted 8 mm (camera mount error); block sits on the real a3 center
    shift = np.array([8.0, 0.0])
    lat = lattice(WS, range(0, 5), range(0, 4), shift=shift)
    p = measure(lat, np.array(WS.measured_square_xy("a3")) + shift, WS, "a3")
    assert p.distance_mm == pytest.approx(0.0, abs=1e-6)
    assert p.view_shift_mm == pytest.approx(8.0)


def test_placement_offset_does_not_change_measurement():
    ws = dataclasses.replace(WS, placement_offset_mm=(0.0, 22.3))
    lat = lattice(ws, range(0, 5), range(0, 4))
    p = measure(lat, ws.measured_square_xy("a3"), ws, "a3")
    assert p.distance_mm == pytest.approx(0.0, abs=1e-6)


def test_refuses_when_mat_has_moved():
    lat = lattice(WS, range(0, 5), range(0, 4), shift=(WS.square_size_mm / 2, 0.0))
    with pytest.raises(LocateError, match="has the mat moved"):
        measure(lat, WS.measured_square_xy("a3"), WS, "a3")
