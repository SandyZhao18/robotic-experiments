import dataclasses

import pytest

from robotic_experiments.blocks import workspace
from robotic_experiments.blocks.design import Design
from robotic_experiments.blocks.planner import PlanError, plan

WS = workspace.load("config/workspace.toml")


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


def test_cell_mapping_hits_measured_corners():
    assert WS.cell_xy(0, 0) == WS.top_left
    assert WS.cell_xy(0, 2) == WS.top_right
    assert WS.cell_xy(2, 0) == WS.bottom_left
    assert WS.cell_xy(1, 1) == (490.0, 0.0)


def test_placeholder_workspace_is_internally_consistent():
    assert WS.problems() == []


def test_plan_is_deterministic():
    assert plan(Design.preset("cross"), WS) == plan(Design.preset("cross"), WS)


def test_one_block_sequence():
    p = plan(Design.preset("one"), WS)
    (a,) = p.assignments
    assert a.pickup_index == 0 and a.cell == (1, 1)
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
    ws = dataclasses.replace(WS, pickups=WS.pickups[:2])
    with pytest.raises(PlanError, match="only 2 pickup spots"):
        plan(Design.preset("line"), ws)


def test_rejects_out_of_bounds():
    ws = dataclasses.replace(WS, bounds_max=(450.0, 350.0, 600.0))  # center cell is at x=490
    with pytest.raises(PlanError, match="outside configured workspace bounds"):
        plan(Design.preset("one"), ws)


def test_rejects_tight_pitch():
    ws = dataclasses.replace(WS, top_right=(580.0, -10.0), bottom_left=(500.0, 90.0))
    with pytest.raises(PlanError, match="too tight"):
        plan(Design.preset("one"), ws)


def test_rejects_swapped_corners():
    # bottom_left measured on the same row as top_right -> axes not perpendicular
    ws = dataclasses.replace(WS, bottom_left=(400.0, -90.0))
    assert any("90°" in p for p in ws.problems())


def test_rejects_pickup_inside_grid():
    ws = dataclasses.replace(WS, pickups=((490.0, 0.0),) + WS.pickups[1:])
    with pytest.raises(PlanError, match="pickup 0 is"):
        plan(Design.preset("one"), ws)
