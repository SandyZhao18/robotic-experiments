import numpy as np
import pytest

from robotic_experiments.blocks import workspace
from robotic_experiments.blocks.fit_correction import FitError, fit

WS = workspace.load("tests/workspace_fixture.toml")


def trial(src, dst, err, start_files=0.0):
    """A logged trial whose result follows the true error field `err(square) -> (files, ranks)`."""
    ef_dst, er_dst = err(dst)
    ef_src, _ = err(src)
    return {
        "from": src, "square": dst,
        "along_files_mm": start_files + ef_dst - ef_src, "along_ranks_mm": er_dst,
        "start_at": {"along_files_mm": start_files},
    }


def test_recovers_linear_error_field():
    e0 = np.array([1.5, -4.0])
    E = np.array([[0.01, -0.02], [-0.04, 0.005]])
    err = lambda sq: e0 + E @ np.array(WS.square_uv(sq))
    trials = [trial("a1", "a3", err, 2.0), trial("c1", "c3", err, -1.0), trial("a1", "e3", err),
              trial("c1", "a7", err, 3.0), trial("a1", "e7", err), trial("c1", "c5", err)]
    e0_r, E_fit = fit(trials, WS)
    assert e0_r == pytest.approx(e0[1])
    assert E_fit == pytest.approx(E)


def test_refuses_without_spread_along_ranks():
    err = lambda sq: np.array([0.0, -4.0])
    with pytest.raises(FitError, match="spread"):
        fit([trial("a1", "a3", err), trial("c1", "c3", err)], WS)


def test_offset_only_shifts_ranks_and_keeps_gradient(tmp_path, monkeypatch, capsys):
    import json
    import sys

    from robotic_experiments.blocks import check_placement, fit_correction

    log = tmp_path / "log.jsonl"
    rows = [{"trial": 1, "square": sq, "from": "a1", "color": "red", "along_files_mm": 0.0, "along_ranks_mm": r,
             "placement_offset_in_use": list(WS.placement_offset_mm), "placement_gradient_in_use": [list(g) for g in WS.placement_gradient]}
            for sq, r in (("a3", 6.0), ("a5", 7.0))]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(check_placement, "LOG_PATH", log)
    monkeypatch.setattr(fit_correction, "LOG_PATH", log)
    monkeypatch.setattr(sys, "argv", ["fit-correction", "--offset-only", "--workspace", "tests/workspace_fixture.toml"])
    fit_correction.main()
    out = capsys.readouterr().out
    assert f"placement_offset_mm = [{WS.placement_offset_mm[0]:.2f}, {WS.placement_offset_mm[1] - 6.5:.2f}]" in out
    assert "placement_gradient = [[0.00000, 0.00000], [0.00000, 0.00000]]" in out
