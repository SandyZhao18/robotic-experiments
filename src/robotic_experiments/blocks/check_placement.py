"""Measure where the red block sits relative to a square's center. Read-only: never moves the robot.

    uv run check-placement a3 --from a1

The block and the chessboard corners are measured in the same camera frame, so errors in the
camera's mounting offset cancel out: the result is the block's physical offset on the mat.
Squares in view are identified by matching against the camera-measured mat in the workspace,
so a1 does not need to be visible. Each result is appended to config/placement_log.jsonl
together with the placement correction configured *now*, so run it right after a trial.
An annotated photo of the square is saved to runs/placements/ for every check, named like
trial-03_a1-to-a3_1.8mm_2026-10-23_1351.jpg.
"""

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from robotic_experiments.blocks import workspace
from robotic_experiments.blocks.locate_mat import Capture, LocateError, capture
from robotic_experiments.connection import connect

DEFAULT_WORKSPACE = "config/workspace.toml"
LOG_PATH = Path("config/placement_log.jsonl")
PHOTO_DIR = Path("runs/placements")
MAX_INDEX_RESIDUAL = 0.3  # squares; how far a corner may sit from where the config predicts one


@dataclass
class Placement:
    square: str
    along_files_mm: float  # + toward the h-file
    along_ranks_mm: float  # + toward rank 8
    distance_mm: float
    view_shift_mm: float  # how far this view's mat sits from the configured one (camera consistency)
    corners_used: int


def mat_coords(world_lattice: np.ndarray, ws: workspace.Workspace) -> np.ndarray:
    """(N, 2) mat coordinates of each detected corner, where square (f, r) has its center at (f, r).

    Each inner corner sits at a1 + (F + 0.5) * file_step + (R + 0.5) * rank_step for whole F, R;
    corners are snapped to those positions using the configured mat.
    """
    pts = world_lattice[..., :2].reshape(-1, 2)
    coords = np.linalg.solve(np.column_stack([ws.file_step, ws.rank_step]), (pts - np.array(ws.a1)).T).T - 0.5
    idx = np.rint(coords)
    worst = float(np.abs(coords - idx).max())
    if worst > MAX_INDEX_RESIDUAL:
        raise LocateError(
            f"corners in view are {worst:.2f} squares from where the configured mat puts them; "
            "has the mat moved? rerun locate-mat"
        )
    return idx + 0.5


def measure(world_lattice: np.ndarray, block_xy, ws: workspace.Workspace, square: str) -> Placement:
    a1 = np.array(ws.a1)
    pts = world_lattice[..., :2].reshape(-1, 2)
    uv = mat_coords(world_lattice, ws)

    A = np.column_stack([np.ones(len(uv)), uv])
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
        corners_used=len(uv),
    )


def trial_number(recheck: bool) -> int:
    """Next trial number from the log; a recheck reuses the latest trial's number."""
    trials = 0
    if LOG_PATH.exists():
        trials = sum(1 for line in LOG_PATH.read_text().splitlines() if line.strip() and not json.loads(line).get("recheck"))
    return max(trials, 1) if recheck else trials + 1


def photo_name(trial: int, recheck: bool, src: str | None, p: Placement, when: datetime) -> str:
    move = f"{src}-to-{p.square}" if src else f"on-{p.square}"
    tag = f"trial-{trial:02d}{'-recheck' if recheck else ''}"
    return f"{tag}_{move}_{p.distance_mm:.1f}mm_{when:%Y-%m-%d_%H%M%S}.jpg"


def annotate(c: Capture, ws: workspace.Workspace, p: Placement) -> tuple[np.ndarray, list[int]]:
    """Outline the target square, mark its center and the block, and print the offset.

    The block is marked where it was measured on the mat, not at its blob in the image: the
    camera sees the 30 mm-tall block at an angle, so its image centroid is shifted by parallax.
    Returns the image and a crop box [x0, y0, x1, y1] around the square for side-by-side reports.
    """
    uv = mat_coords(c.world_lattice, ws)
    H, _ = cv2.findHomography(uv.astype(np.float32), c.corners_px.reshape(-1, 2).astype(np.float32))

    def to_px(u, v):
        x, y, w = H @ np.array([u, v, 1.0])
        return int(round(x / w)), int(round(y / w))

    f, r = workspace.parse_square(p.square)
    out = c.bgr.copy()
    outline = np.array([to_px(f + du, r + dv) for du, dv in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5))])
    cv2.polylines(out, [outline], True, (0, 255, 0), 2)
    cv2.drawMarker(out, to_px(f, r), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
    sq = ws.square_size_mm
    cv2.drawMarker(out, to_px(f + p.along_files_mm / sq, r + p.along_ranks_mm / sq), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 24, 2)
    lines = [
        f"{p.square}: {p.distance_mm:.1f} mm off center",
        f"files {p.along_files_mm:+.1f} mm, ranks {p.along_ranks_mm:+.1f} mm",
        "green = square center, magenta = block",
    ]
    for i, text in enumerate(lines):
        y = 30 + 30 * i
        cv2.putText(out, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(out, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    ring = np.array([to_px(f + du, r + dv) for du in (-1, 1) for dv in (-1, 1)])  # one square of margin
    h, w = out.shape[:2]
    x0, y0 = np.clip(ring.min(axis=0), 0, [w, h])
    x1, y1 = np.clip(ring.max(axis=0), 0, [w, h])
    return out, [int(x0), int(y0), int(x1), int(y1)]


async def run(ws: workspace.Workspace, square: str) -> tuple[Placement, np.ndarray, list[int]]:
    async with await connect() as machine:
        c = await capture(machine)
    p = measure(c.world_lattice, c.block_world[:2], ws, square)
    return p, *annotate(c, ws, p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("square", help="the square the block should be on, e.g. a3")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, help="workspace TOML (default: %(default)s)")
    parser.add_argument("--from", dest="src", metavar="SQUARE", help="where the block was picked up (for naming)")
    parser.add_argument("--recheck", action="store_true", help="re-measuring a block already logged (not a new trial)")
    args = parser.parse_args()
    ws = workspace.load(args.workspace)
    try:
        workspace.parse_square(args.square)
        p, photo, crop = asyncio.run(run(ws, args.square))
    except (LocateError, ValueError) as e:
        raise SystemExit(f"Could not check placement: {e}")

    now = datetime.now()
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    trial = trial_number(args.recheck)
    photo_path = PHOTO_DIR / photo_name(trial, args.recheck, args.src, p, now)
    cv2.imwrite(str(photo_path), photo)

    with LOG_PATH.open("a") as f:
        record = {**asdict(p), "placement_offset_in_use": list(ws.placement_offset_mm),
                  "trial": trial, "from": args.src,
                  "photo": str(photo_path), "crop_px": crop, "recheck": args.recheck,
                  "recorded_at": now.isoformat(timespec="seconds")}
        f.write(json.dumps(record) + "\n")

    of, orank = ws.placement_offset_mm
    print(f"Trial {trial}{' (recheck)' if args.recheck else ''}: block is {p.distance_mm:.1f} mm from the center of {p.square}:")
    print(f"  {p.along_files_mm:+.1f} mm along files (+ = toward h)")
    print(f"  {p.along_ranks_mm:+.1f} mm along ranks (+ = toward rank 8)")
    print(f"  ({p.corners_used} corners; this view's mat is {p.view_shift_mm:.1f} mm from the configured one)")
    print("Assuming the block was placed with the current config:")
    print(f"  if this error is systematic, placement_offset_mm would become "
          f"[{of - p.along_files_mm:.1f}, {orank - p.along_ranks_mm:.1f}] (now [{of:.1f}, {orank:.1f}]).")
    print(f"Logged to {LOG_PATH}; photo {photo_path}")


if __name__ == "__main__":
    main()
