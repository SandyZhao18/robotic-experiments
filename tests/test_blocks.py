import dataclasses

import pytest

from robotic_experiments.blocks import workspace
from robotic_experiments.blocks.design import Design
from robotic_experiments.blocks.planner import PlanError, plan

WS = workspace.load("tests/workspace_fixture.toml")


def test_parse_matches_render():
    d = Design.parse("■□□\n■□□\n■■■")
    assert d == Design.preset("L")
    assert d.render() == "■ □ □\n■ □ □\n■ ■ ■"


@pytest.mark.parametrize("bad", ["#..|#..", "#...|#..|###", "#..|#..|#?#", ""])
def test_parse_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        Design.parse(bad)


def test_filled_is_row_by_row():
    assert Design.preset("L").filled() == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]


def test_square_mapping_hits_recorded_corners():
    assert WS.square_xy("a1") == WS.a1
    assert WS.square_xy("h1") == pytest.approx(WS.h1)
    assert WS.square_xy("a8") == pytest.approx(WS.a8)


def test_design_displays_as_seen_from_white():
    # top_left = c7, stride 2: top row is rank 7, bottom row rank 3, columns c/e/g
    assert [WS.cell_square(0, c) for c in range(3)] == ["c7", "e7", "g7"]
    assert [WS.cell_square(r, 0) for r in range(3)] == ["c7", "c5", "c3"]
    assert WS.cell_xy(1, 1) == pytest.approx(WS.square_xy("e5"))


def test_placeholder_workspace_is_internally_consistent():
    assert WS.problems() == []


def test_plan_is_deterministic():
    assert plan(Design.preset("cross"), WS) == plan(Design.preset("cross"), WS)


def test_one_block_sequence():
    p = plan(Design.preset("one"), WS)
    (a,) = p.assignments
    assert a.pickup_square == "a1" and a.cell == (1, 1) and a.cell_square == "e5"
    kinds = [s.kind for s in p.steps]
    assert kinds == ["move", "move", "move", "open", "move", "grab", "move", "move",
                     "move", "move", "move", "release", "move", "move", "move"]
    grip = next(s for s in p.steps if s.label.startswith("descend"))
    assert (grip.target.x, grip.target.y, grip.target.z) == (*WS.pickups[0], WS.grip_z)
    lower = next(s for s in p.steps if s.label.startswith("lower"))
    assert (lower.target.x, lower.target.y) == WS.cell_xy(1, 1)
    assert lower.target.z == WS.grip_z + WS.place_release_offset_mm


def test_rejects_empty_design():
    with pytest.raises(PlanError, match="no filled cells"):
        plan(Design.parse("...|...|..."), WS)


def test_rejects_too_few_pickups():
    ws = dataclasses.replace(WS, pickup_squares=WS.pickup_squares[:2])
    with pytest.raises(PlanError, match="only 2 pickup spots"):
        plan(Design.preset("line"), ws)


def test_rejects_out_of_bounds():
    ws = dataclasses.replace(WS, bounds_max=(500.0, 250.0, 600.0))  # e5 is at x=528
    with pytest.raises(PlanError, match="outside configured workspace bounds"):
        plan(Design.preset("one"), ws)


def test_rejects_tight_pitch():
    ws = dataclasses.replace(WS, grid_stride=1)
    with pytest.raises(PlanError, match="too tight"):
        plan(Design.preset("one"), ws)


def test_rejects_swapped_corners():
    ws = dataclasses.replace(WS, a8=WS.h1, h1=WS.a8)
    assert ws.problems() == []  # a mirror image is still square...
    ws = dataclasses.replace(WS, a8=(WS.a8[0], WS.h1[1]))  # ...but a8 on the h-file is not
    assert any("90°" in p for p in ws.problems())


def test_rejects_corner_disagreeing_with_ruler():
    ws = dataclasses.replace(WS, square_size_mm=50.0)
    assert any("recheck the recorded corners" in p for p in ws.problems())


def test_rejects_grid_off_mat():
    ws = dataclasses.replace(WS, grid_top_left="g7")
    with pytest.raises(PlanError, match="falls off the mat"):
        plan(Design.preset("one"), ws)


def test_rejects_pickup_inside_grid():
    ws = dataclasses.replace(WS, pickup_squares=("e5",) + WS.pickup_squares[1:])
    with pytest.raises(PlanError, match="inside the build grid"):
        plan(Design.preset("one"), ws)


def test_rejects_pickups_too_close():
    ws = dataclasses.replace(WS, pickup_squares=("a1", "b1"))
    with pytest.raises(PlanError, match=r"pickup 1 \(b1\) is"):
        plan(Design.preset("one"), ws)


def test_placement_offset_shifts_targets_not_measurements():
    ws = dataclasses.replace(WS, placement_offset_mm=(0.0, 10.0))
    assert ws.measured_square_xy("e5") == WS.square_xy("e5")
    mx, my = ws.measured_square_xy("e5")
    x, y = ws.square_xy("e5")
    assert (x - mx, y - my) == pytest.approx((10 * ws.rank_unit[0], 10 * ws.rank_unit[1]))
