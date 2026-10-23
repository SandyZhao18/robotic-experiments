"""Fit the placement correction across the mat from logged trials. Offline: reads the log only.

    uv run fit-correction --since-trial 7 --exclude 8
    uv run fit-correction --since-trial 18 --offset-only   # shift only; keep the fitted gradient

Model: when sent to a square, the fingers end up off by an error that varies linearly across
the mat, err(u, v) = e0 + E @ (u, v), with (u, v) the square's position in mm from a1 along
(files, ranks). Each trial gives:

- ranks: the fingers re-center the block along ranks, so the result's ranks offset is
  err_ranks at the target square;
- files: the block keeps its offset from pickup along files, so (result - start) along files
  is err_files(target) - err_files(pickup). That fixes E's files row but not e0's files part,
  which cancels for any pick-and-place and is left unchanged.

Trials used must share the correction that is configured now. Prints the new
placement_offset_mm / placement_gradient and each trial's predicted residual.
"""

import argparse
import json
import sys

import numpy as np

from robotic_experiments.blocks import workspace
from robotic_experiments.blocks.check_placement import LOG_PATH, is_trial

DEFAULT_WORKSPACE = "config/workspace.toml"


def pair_starts(recs: list[dict]) -> list[dict]:
    """Trial results, each with the latest start of the same block on its pickup square."""
    pending, out = {}, []
    for r in recs:
        key = (r.get("color", "red"), r["square"] if r.get("start") else r.get("from"))
        if r.get("start"):
            pending[key] = r
        elif is_trial(r):
            out.append({**r, "start_at": pending.pop(key, None)})
    return out


class FitError(ValueError):
    pass


def _lstsq(rows, obs, what: str) -> np.ndarray:
    A, b = np.array(rows, dtype=float), np.array(obs, dtype=float)
    if len(b) < A.shape[1] or np.linalg.matrix_rank(A) < A.shape[1]:
        raise FitError(f"not enough spread for the {what}: need trials on squares that differ along both files and ranks")
    return np.linalg.lstsq(A, b, rcond=None)[0]


def fit(trials: list[dict], ws: workspace.Workspace) -> tuple[float, np.ndarray]:
    """Least-squares error model: returns (ranks offset e0_ranks, 2x2 gradient E in mm per mm)."""
    rank_rows, rank_obs, file_rows, file_obs = [], [], [], []
    for t in trials:
        u, v = ws.square_uv(t["square"])
        rank_rows.append([1.0, u, v])
        rank_obs.append(t["along_ranks_mm"])
        if t.get("start_at") and t.get("from"):
            pu, pv = ws.square_uv(t["from"])
            file_rows.append([u - pu, v - pv])
            file_obs.append(t["along_files_mm"] - t["start_at"]["along_files_mm"])
    er = _lstsq(rank_rows, rank_obs, "ranks error")
    ef = _lstsq(file_rows, file_obs, "files error (trials need start checks)")
    return float(er[0]), np.array([[ef[0], ef[1]], [er[1], er[2]]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since-trial", type=int, default=1, help="ignore earlier trials (e.g. other measurement method)")
    parser.add_argument("--offset-only", action="store_true",
                        help="only re-fit the ranks offset (trials cover too small an area for gradients)")
    parser.add_argument("--exclude", type=int, nargs="*", default=[], help="trial numbers to leave out (known bad runs)")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, help="workspace TOML (default: %(default)s)")
    args = parser.parse_args()

    ws = workspace.load(args.workspace)
    recs = [json.loads(line) for line in LOG_PATH.read_text().splitlines() if line.strip()]
    current = (list(ws.placement_offset_mm), [list(r) for r in ws.placement_gradient])
    trials = [
        t for t in pair_starts(recs)
        if t.get("trial", 0) >= args.since_trial and t.get("trial") not in args.exclude
        and (t.get("placement_offset_in_use"), t.get("placement_gradient_in_use", [[0.0, 0.0], [0.0, 0.0]])) == current
    ]
    if not trials:
        sys.exit("No trials with the current correction since that trial.")

    if args.offset_only:
        e0_r, E = float(np.mean([t["along_ranks_mm"] for t in trials])), np.zeros((2, 2))
    else:
        try:
            e0_r, E = fit(trials, ws)
        except FitError as e:
            sys.exit(f"Cannot fit yet: {e}.")
    G = np.array(ws.placement_gradient)
    new_offset = (ws.placement_offset_mm[0], ws.placement_offset_mm[1] - e0_r)
    new_gradient = G - E

    print(f"Using {len(trials)} trials: " + ", ".join(f"{t['trial']} ({t.get('from')}->{t['square']})" for t in trials))
    if args.exclude:
        print(f"Excluded trials: {', '.join(map(str, args.exclude))}")
    print(f"Error model: ranks offset {e0_r:+.2f} mm; per metre along files/ranks: "
          f"files [{E[0, 0] * 1000:+.1f}, {E[0, 1] * 1000:+.1f}], ranks [{E[1, 0] * 1000:+.1f}, {E[1, 1] * 1000:+.1f}] mm")
    print("\nPredicted remaining error per trial (files change, ranks), mm:")
    for t in trials:
        u, v = ws.square_uv(t["square"])
        rank_res = t["along_ranks_mm"] - (e0_r + E[1] @ (u, v))
        files = "     -"
        if t.get("start_at"):
            pu, pv = ws.square_uv(t["from"])
            files = f"{(t['along_files_mm'] - t['start_at']['along_files_mm']) - E[0] @ (u - pu, v - pv):+6.1f}"
        print(f"  trial {t['trial']:2d} {t.get('from')}->{t['square']}: files {files}, ranks {rank_res:+6.1f}")
    print("\nNew correction for [mat] in the workspace:")
    print(f"placement_offset_mm = [{new_offset[0]:.2f}, {new_offset[1]:.2f}]")
    print(f"placement_gradient = [[{new_gradient[0, 0]:.5f}, {new_gradient[0, 1]:.5f}], [{new_gradient[1, 0]:.5f}, {new_gradient[1, 1]:.5f}]]")


if __name__ == "__main__":
    main()
