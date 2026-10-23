import pytest

from robotic_experiments.blocks.structure import StructureError, parse_at, solve

SQ = 57.0


def test_positions_squares_and_seams():
    assert parse_at("a3") == (0.0, 2.0)
    assert parse_at("a3|b3") == (0.5, 2.0)
    assert parse_at("c4 | c5") == (2.0, 3.5)
    with pytest.raises(StructureError, match="share a side"):
        parse_at("a3|b4")


def test_three_cube_tower():
    p = solve([{"at": "a3"}, {"at": "a3"}, {"at": "a3", "level": 3}], SQ)
    assert [b.level for b in p] == [1, 2, 3]
    assert [b.base_mm for b in p] == [0.0, 30.0, 60.0]
    assert p[2].supports == (1,)


def test_brick_bridges_two_cubes_across_a_seam():
    # cubes 57 mm apart; the 60 mm brick overlaps each by 16.5 mm
    p = solve([{"at": "a3"}, {"at": "b3"}, {"shape": "brick", "at": "a3|b3", "along": "files"}], SQ)
    assert p[2].level == 2 and set(p[2].supports) == {0, 1}


def test_cube_on_a_seam_over_two_cubes_barely_touches_them():
    # adjacent cubes leave a 27 mm gap; a 30 mm cube over the seam overlaps each by only 1.5 mm
    with pytest.raises(StructureError, match="barely touch"):
        solve([{"at": "a3"}, {"at": "b3"}, {"at": "a3|b3"}], SQ)


def test_cube_on_a_seam_on_top_of_a_bridging_brick():
    p = solve([{"at": "a3"}, {"at": "b3"}, {"shape": "brick", "at": "a3|b3"}, {"at": "a3|b3", "level": 3}], SQ)
    assert p[3].supports == (2,) and p[3].base_mm == 60.0


def test_brick_resting_on_one_end_tips_over():
    # 16.5 mm of contact on a3 is plenty, but the brick's center hangs over the empty b3
    with pytest.raises(StructureError, match="tip over"):
        solve([{"at": "a3"}, {"shape": "brick", "at": "a3|b3"}], SQ)


def test_seam_on_the_mat():
    p = solve([{"shape": "cylinder", "at": "c3|c4"}], SQ)
    assert p[0].level == 1 and p[0].base_mm == 0.0 and (p[0].f, p[0].r) == (2.0, 2.5)


def test_bridge_needs_supports_at_one_height():
    # brick across a 1-cube and a 2-cube stack rests on the taller one only -> its center is not supported
    with pytest.raises(StructureError, match="tip over"):
        solve([{"at": "a3"}, {"at": "b3"}, {"at": "b3"}, {"shape": "brick", "at": "a3|b3"}], SQ)


def test_level_assertion_catches_wrong_order():
    with pytest.raises(StructureError, match="list blocks bottom first"):
        solve([{"at": "a3", "level": 2}], SQ)


def test_brick_along_ranks_footprint():
    p = solve([{"shape": "brick", "at": "a3|a4", "along": "ranks"}], SQ)
    fp = p[0].footprint(SQ)
    assert (fp.u1 - fp.u0, fp.v1 - fp.v0) == (30.0, 60.0)


def test_unknown_shape():
    with pytest.raises(StructureError, match="unknown shape"):
        solve([{"shape": "pyramid", "at": "a3"}], SQ)


def _ws():
    import dataclasses

    from robotic_experiments.blocks import workspace

    return dataclasses.replace(workspace.load("tests/workspace_fixture.toml"), theta_deg=-91.0,
                               bounds_min=(-1e4, -1e4, -1e4), bounds_max=(1e4, 1e4, 1e4))


def test_tower_releases_one_cube_higher_each_level():
    from robotic_experiments.blocks.planner import plan_structure

    ws = _ws()
    placed = solve([{"at": "a3"}, {"at": "a3"}, {"at": "a3"}], ws.square_size_mm)
    p = plan_structure(ws, placed, [("f1", (5.0, 0.0)), ("g1", (6.0, 0.0)), ("h1", (7.0, 0.0))])
    lows = [s.target.z for s in p.steps if s.label.startswith("lower block")]
    assert [round(z - lows[0], 6) for z in lows] == [0.0, 30.0, 60.0]
    assert all(s.position == ws.release_position(30.0) for s in p.steps if s.kind == "release")


def test_brick_along_ranks_is_released_wider():
    from robotic_experiments.blocks.planner import plan_structure

    ws = _ws()
    placed = solve([{"shape": "brick", "at": "a3|a4", "along": "ranks"}], ws.square_size_mm)
    p = plan_structure(ws, placed, [("f1|f2", (5.0, 0.5))])
    (rel,) = [s for s in p.steps if s.kind == "release"]
    assert rel.position == ws.release_position(60.0)


def test_tall_cylinder_is_gripped_higher():
    from robotic_experiments.blocks.planner import plan_structure

    ws = _ws()
    placed = solve([{"shape": "cylinder", "at": "c3"}], ws.square_size_mm)
    p = plan_structure(ws, placed, [("f1", (5.0, 0.0))])
    grip = next(s for s in p.steps if s.label.startswith("descend"))
    assert grip.target.z == pytest.approx(ws.grip_z + 60.0 - ws.finger_depth_mm)


def test_drop_next_to_a_taller_neighbour_along_the_finger_opening_is_refused():
    from robotic_experiments.blocks.planner import PlanError, plan_structure

    ws = _ws()
    # a 2-cube stack on a4, then a cube on a3 right next to it along the ranks (the finger opening)
    placed = solve([{"at": "a4"}, {"at": "a4"}, {"at": "a3"}], ws.square_size_mm)
    with pytest.raises(PlanError, match="drop on a3 would touch the block on a4"):
        plan_structure(ws, placed, [("f1", (5.0, 0.0)), ("g1", (6.0, 0.0)), ("h1", (7.0, 0.0))])


def test_travel_rises_over_tall_stacks():
    from robotic_experiments.blocks.planner import TRAVEL_CLEARANCE_MM, plan_structure

    ws = _ws()
    placed = solve([{"at": "a3"}, {"at": "a3"}, {"at": "a3"}, {"shape": "cylinder", "at": "a3"}], ws.square_size_mm)
    p = plan_structure(ws, placed, [(s, (f, 0.0)) for s, f in (("e1", 4.0), ("f1", 5.0), ("g1", 6.0), ("h1", 7.0))])
    last_travel = max(s.target.z for s in p.steps if s.block == 3 and s.label.startswith("travel above"))
    lift = ws.grip_lift_mm(60.0)
    assert last_travel >= ws.mat_top_z + 90 + TRAVEL_CLEARANCE_MM + 5 + lift + ws.tcp_above_fingertips_mm - 1e-6
