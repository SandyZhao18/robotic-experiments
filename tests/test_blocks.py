import dataclasses

import pytest

from robotic_experiments.blocks import workspace
from robotic_experiments.blocks.design import Design
from robotic_experiments.blocks.planner import PlanError, plan, plan_transfer

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
                     "move", "move", "move", "release", "move", "check_released", "move", "move"]
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


def test_pickups_side_by_side_across_the_finger_opening_are_allowed():
    ws = dataclasses.replace(WS, theta_deg=-91.0)  # fingers open along ranks
    assert ws.fingers_open_along_ranks
    ok = dataclasses.replace(ws, pickup_squares=("a1", "b1", "c1", "d1", "e1"))
    assert not [p for p in ok.layout_problems() if "pickup" in p]
    bad = dataclasses.replace(ws, pickup_squares=("a1", "a2"))
    with pytest.raises(PlanError, match=r"pickup 1 \(a2\)"):
        plan(Design.preset("one"), bad)


def test_finger_direction_follows_theta():
    along_ranks = dataclasses.replace(WS, theta_deg=-91.0)
    along_files = dataclasses.replace(WS, theta_deg=-1.0)
    assert along_ranks.finger_conflict("a3", "a4", 100.0) and not along_ranks.finger_conflict("a3", "b3", 100.0)
    assert along_files.finger_conflict("a3", "b3", 100.0) and not along_files.finger_conflict("a3", "a4", 100.0)


def test_placement_offset_shifts_targets_not_measurements():
    ws = dataclasses.replace(WS, placement_offset_mm=(0.0, 10.0))
    assert ws.measured_square_xy("e5") == WS.square_xy("e5")
    mx, my = ws.measured_square_xy("e5")
    x, y = ws.square_xy("e5")
    assert (x - mx, y - my) == pytest.approx((10 * ws.rank_unit[0], 10 * ws.rank_unit[1]))


def test_transfer_refuses_adjacent_squares_with_fully_open_fingers():
    # span 100 mm -> a neighbour along the opening needs 75 mm; adjacent squares are 57 mm apart
    ws = dataclasses.replace(WS, theta_deg=-91.0)
    with pytest.raises(PlanError, match="along the finger opening"):
        plan_transfer(ws, [("a1", "a3"), ("c1", "a4")])


def test_transfer_allows_adjacent_squares_with_partly_open_fingers():
    ws = dataclasses.replace(WS, theta_deg=-91.0, open_position=400.0, open_finger_span_mm=65.0, finger_clearance_mm=8.0)
    p = plan_transfer(ws, [("a1", "a3"), ("c1", "a4")])
    assert [a.cell_square for a in p.assignments] == ["a3", "a4"]


def test_transfer_refuses_dropping_next_to_a_waiting_block():
    ws = dataclasses.replace(WS, open_position=400.0, open_finger_span_mm=65.0, finger_clearance_mm=30.0)
    with pytest.raises(PlanError, match="block on c1"):
        plan_transfer(ws, [("a1", "b1"), ("c1", "c3")])


def test_preflight_treats_partial_opening_as_empty():
    from types import SimpleNamespace

    from robotic_experiments.blocks.executor import parked_open

    ws = dataclasses.replace(WS, open_position=360.0)
    assert parked_open(SimpleNamespace(meta={"position": 362.0}), ws)
    assert not parked_open(SimpleNamespace(meta={"position": 220.0}), ws)  # a cube in the fingers
    assert not parked_open(SimpleNamespace(meta={}), ws)
    # release openings per block size are parking spots too (30 mm -> 360, 60 mm -> 660) ...
    assert parked_open(SimpleNamespace(meta={"position": 662.0}), dataclasses.replace(WS, open_position=None))
    # ... but fingers closed on a brick's 60 mm length (~520) are holding it
    assert not parked_open(SimpleNamespace(meta={"position": 520.0}), dataclasses.replace(WS, open_position=None))


def test_plan_from_loose_blocks_uses_pickable_ones_in_order():
    from robotic_experiments.blocks.planner import Loose

    ws = dataclasses.replace(WS, theta_deg=-91.0)
    loose = [Loose("twisted", 6.0, 0.0, pickable=False), Loose("red", 6.0, 3.0), Loose("blue", 7.0, 5.0)]
    p = plan(Design.parse("...|...|##."), ws, loose)
    assert [a.pickup_square for a in p.assignments] == ["red", "blue"]
    assert p.assignments[0].pickup_xy == ws.xy_at(6.0, 3.0)


def test_plan_from_loose_blocks_refuses_crowded_pickups():
    from robotic_experiments.blocks.planner import Loose

    ws = dataclasses.replace(WS, theta_deg=-91.0)
    # 0.8 squares apart along ranks (the finger opening): too close for fully open fingers,
    # even though the neighbour itself is not chosen
    loose = [Loose("red", 6.0, 1.0), Loose("twisted", 6.0, 1.8, pickable=False)]
    with pytest.raises(PlanError, match="spread the blocks apart"):
        plan(Design.preset("one"), ws, loose)


def test_plan_from_loose_blocks_needs_enough():
    from robotic_experiments.blocks.planner import Loose

    with pytest.raises(PlanError, match="only 1 pickable blocks were found"):
        plan(Design.preset("line"), WS, [Loose("red", 6.0, 1.0)])


def test_twisted_block_is_grabbed_with_the_wrist_turned_at_travel_height():
    from robotic_experiments.blocks.planner import Loose, wrist_theta

    ws = dataclasses.replace(WS, theta_deg=-91.0)
    p = plan(Design.preset("one"), ws, [Loose("red", 6.0, 1.0, rotation_deg=25.0)])
    turned = [s for s in p.steps if s.target and s.target.theta != -91.0]
    assert wrist_theta(-91.0, 25.0) == -116.0
    assert {s.target.theta for s in turned} == {-116.0}
    labels = [s.label for s in p.steps]
    i_turn = next(i for i, l in enumerate(labels) if l.startswith("turn wrist +25"))
    i_back = next(i for i, l in enumerate(labels) if l.startswith("turn wrist back"))
    assert p.steps[i_turn].target.z == p.steps[i_back].target.z == ws.travel_z
    assert labels[i_turn - 1].startswith("travel above") and labels[i_back - 1] == "lift to travel height"
    # a square-enough block is grabbed without turning
    q = plan(Design.preset("one"), ws, [Loose("red", 6.0, 1.0, rotation_deg=1.0)])
    assert not any(s.label.startswith("turn wrist") for s in q.steps)


def test_turned_fingers_change_which_neighbours_are_in_the_way():
    ws = dataclasses.replace(WS, theta_deg=-91.0)
    # neighbour one square away along the files: beside the usual fingers, in front of fingers turned 90
    assert ws.finger_conflict_at((6.0, 1.0), (5.0, 1.0), 100.0, "n") is None
    assert ws.finger_conflict_at((6.0, 1.0), (5.0, 1.0), 100.0, "n", turn_deg=90.0)


def test_pickup_order_avoids_neighbours_that_are_already_gone():
    from robotic_experiments.blocks.planner import Loose

    ws = dataclasses.replace(WS, theta_deg=-91.0)
    # diagonal neighbours: square a is clear of b, but b is twisted so its fingers open toward a.
    # b is listed first; picking a first makes b clear.
    a, b = Loose("a", 6.0, 1.0), Loose("b", 6.9, 1.6, rotation_deg=-35.0)
    assert ws.finger_conflict_at((6.9, 1.6), (6.0, 1.0), ws.full_open_span_mm, "a", -35.0)
    assert not ws.finger_conflict_at((6.0, 1.0), (6.9, 1.6), ws.full_open_span_mm, "b", 0.0)
    p = plan(Design.parse("...|...|##."), ws, [b, a])
    assert [x.pickup_square for x in p.assignments] == ["a", "b"]
    # mutual conflict: neither order works
    with pytest.raises(PlanError, match="spread the blocks apart"):
        plan(Design.parse("...|...|##."), ws, [Loose("a", 6.0, 1.0), Loose("b", 6.0, 1.9)])
