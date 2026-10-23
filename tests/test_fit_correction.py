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
