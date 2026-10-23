from robotic_experiments.blocks import scene
from robotic_experiments.blocks.check_scene import Seen, compare, updated
from robotic_experiments.blocks.structure import solve

SQ = 57.0


def blocks(spec):
    return [scene.from_placed(p) for p in solve(spec, SQ)]


def test_only_tops_are_visible():
    rec = blocks([{"at": "a3"}, {"at": "a3"}, {"at": "b3"}])
    tops = scene.visible_tops(rec, SQ)
    assert [(b.at, b.top_mm) for b in tops] == [("a3", 60.0), ("b3", 30.0)]


def test_compare_matches_missing_and_unexpected():
    rec = blocks([{"at": "a3"}, {"at": "a3"}, {"at": "b3"}])
    seen = [Seen("red", 0.05, 2.0, 61.0),  # top of the a3 stack, 3 mm off
            Seen("blue", 4.0, 5.0, 30.0)]  # something on e6 that is not in the record
    rep = compare(rec, seen, SQ)
    assert [(t.at, s.color) for t, s in rep.matched] == [("a3", "red")]
    assert [t.at for t in rep.missing] == ["b3"]
    assert [s.color for s in rep.unexpected] == ["blue"]
    assert not rep.ok


def test_wrong_height_is_not_a_match():
    rec = blocks([{"at": "a3"}, {"at": "a3"}])
    rep = compare(rec, [Seen("red", 0.0, 2.0, 30.0)], SQ)  # only one cube there
    assert rep.missing and rep.unexpected


def test_update_moves_matched_tops_to_measured_positions():
    rec = blocks([{"at": "a3"}])
    rep = compare(rec, [Seen("red", 0.04, 2.02, 30.5)], SQ)
    (b,) = updated(rec, rep)
    assert (b.f, b.r, b.measured, b.color) == (0.04, 2.02, True, "red")


def test_scene_file_roundtrip(tmp_path):
    path = tmp_path / "scene.json"
    rec = blocks([{"at": "a3"}, {"shape": "brick", "at": "a4|b4"}])
    scene.save(rec, path)
    assert scene.load(path) == rec
    scene.add(rec[0], path)
    assert len(scene.load(path)) == 3
