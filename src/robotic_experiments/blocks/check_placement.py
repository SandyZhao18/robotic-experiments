"""Measure where the red block sits relative to a square's center. Read-only: never moves the robot.

    uv run check-placement a3

The block and the chessboard corners are measured in the same camera frame, so errors in the
camera's mounting offset cancel out: the result is the block's physical offset on the mat.
Squares in view are identified by matching against the camera-measured mat in the workspace,
so a1 does not need to be visible. Each result is appended to config/placement_log.jsonl
together with the placement correction configured *now*, so run it right after a trial.
"""

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from robotic_experiments.blocks import workspace
from robotic_experiments.blocks.locate_mat import LocateError, capture
from robotic_experiments.connection import connect

DEFAULT_WORKSPACE = "config/workspace.toml"
LOG_PATH = Path("config/placement_log.jsonl")
MAX_INDEX_RESIDUAL = 0.3  # squares; how far a corner may sit from where the config predicts one


@dataclass
class Placement:
    square: str
    along_files_mm: float  # + toward the h-file
    along_ranks_mm: float  # + toward rank 8
    distance_mm: float
    view_shift_mm: float  # how far this view's mat sits from the configured one (camera consistency)
    corners_used: int


def measure(world_lattice: np.ndarray, block_xy, ws: workspace.Workspace, square: str) -> Placement:
    a1 = np.array(ws.a1)
    m_cfg = np.column_stack([ws.file_step, ws.rank_step])
    pts = world_lattice[..., :2].reshape(-1, 2)

    # Each inner corner sits at a1 + (F + 0.5) * file_step + (R + 0.5) * rank_step.
    coords = np.linalg.solve(m_cfg, (pts - a1).T).T - 0.5
    idx = np.rint(coords)
    worst = float(np.abs(coords - idx).max())
    if worst > MAX_INDEX_RESIDUAL:
        raise LocateError(
            f"corners in view are {worst:.2f} squares from where the configured mat puts them; "
            "has the mat moved? rerun locate-mat"
        )

    A = np.column_stack([np.ones(len(idx)), idx[:, 0] + 0.5, idx[:, 1] + 0.5])
    (a1_view, fs, rs), *_ = np.linalg.lstsq(A, pts, rcond=None)

    f, r = workspace.parse_square(square)
    center = a1_view + f * fs + r * rs
    fu, ru = fs / np.linalg.norm(fs), rs / np.linalg.norm(rs)
    along_f, along_r = np.linalg.solve(np.column_stack([fu, ru]), np.asarray(block_xy) - center)
    return Placement(
        square=square,
        along_files_mm=float(along_f),
        along_ranks_mm=float(along_r),
        distance_mm=float(np.hypot(along_f, along_r)),
        view_shift_mm=float(np.linalg.norm(a1_view - a1)),
        corners_used=len(idx),
    )


async def run(ws: workspace.Workspace, square: str) -> Placement:
    async with await connect() as machine:
        c = await capture(machine)
    return measure(c.world_lattice, c.block_world[:2], ws, square)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("square", help="the square the block should be on, e.g. a3")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, help="workspace TOML (default: %(default)s)")
    args = parser.parse_args()
    ws = workspace.load(args.workspace)
    try:
        workspace.parse_square(args.square)
        p = asyncio.run(run(ws, args.square))
    except (LocateError, ValueError) as e:
        raise SystemExit(f"Could not check placement: {e}")

    with LOG_PATH.open("a") as f:
        record = {**asdict(p), "placement_offset_in_use": list(ws.placement_offset_mm),
                  "recorded_at": datetime.now().isoformat(timespec="seconds")}
        f.write(json.dumps(record) + "\n")

    of, orank = ws.placement_offset_mm
    print(f"Block is {p.distance_mm:.1f} mm from the center of {p.square}:")
    print(f"  {p.along_files_mm:+.1f} mm along files (+ = toward h)")
    print(f"  {p.along_ranks_mm:+.1f} mm along ranks (+ = toward rank 8)")
    print(f"  ({p.corners_used} corners; this view's mat is {p.view_shift_mm:.1f} mm from the configured one)")
    print("Assuming the block was placed with the current config:")
    print(f"  if this error is systematic, placement_offset_mm would become "
          f"[{of - p.along_files_mm:.1f}, {orank - p.along_ranks_mm:.1f}] (now [{of:.1f}, {orank:.1f}]).")
    print(f"Logged to {LOG_PATH}.")


if __name__ == "__main__":
    main()
