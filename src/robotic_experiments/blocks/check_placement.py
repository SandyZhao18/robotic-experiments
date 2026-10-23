"""Measure where the red block sits relative to a square's center. Read-only: never moves the robot.

    uv run check-placement a1 --start      # where the block starts, before a trial
    uv run check-placement a3 --from a1    # where it landed, after the trial
    uv run check-placement c3 --from c1 --color yellow   # only consider yellow blocks

The block measured is the one nearest the square (any known color unless --color is given).

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
from robotic_experiments.blocks.locate_mat import BLOCK_COLORS, Capture, LocateError, capture
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


def fit_view(world_lattice: np.ndarray, ws: workspace.Workspace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The mat as seen in this frame: a1's center, file step and rank step, in `world` x/y."""
    pts = world_lattice[..., :2].reshape(-1, 2)
    uv = mat_coords(world_lattice, ws)
    A = np.column_stack([np.ones(len(uv)), uv])
    (a1_view, fs, rs), *_ = np.linalg.lstsq(A, pts, rcond=None)
    return a1_view, fs, rs


def measure(world_lattice: np.ndarray, block_xy, ws: workspace.Workspace, square: str) -> Placement:
    a1 = np.array(ws.a1)
    uv = mat_coords(world_lattice, ws)
    a1_view, fs, rs = fit_view(world_lattice, ws)

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


def is_trial(record: dict) -> bool:
    return not record.get("recheck") and not record.get("start")


def trial_number(kind: str) -> int:
    """Number of the trial this check belongs to.

    A result starts a new trial; a recheck belongs to the latest trial; start measurements
    belong to the trials about to run, in order (two starts before a two-block run -> N+1, N+2).
    """
    recs = [json.loads(line) for line in LOG_PATH.read_text().splitlines() if line.strip()] if LOG_PATH.exists() else []
    trials = sum(1 for r in recs if is_trial(r))
    if kind == "recheck":
        return max(trials, 1)
    if kind == "start":
        pending = 0
        for r in reversed(recs):
            if is_trial(r):
                break
            pending += bool(r.get("start"))
        return trials + 1 + pending
    return trials + 1


def photo_name(trial: int, kind: str, color: str, src: str | None, p: Placement, when: datetime) -> str:
    move = f"{src}-to-{p.square}" if src else f"on-{p.square}"
    tag = f"trial-{trial:02d}{'-' + kind if kind else ''}"
    return f"{tag}_{color}_{move}_{p.distance_mm:.1f}mm_{when:%Y-%m-%d_%H%M%S}.jpg"


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


async def run(ws: workspace.Workspace, square: str, color: str) -> tuple[Placement, str, np.ndarray, list[int]]:
    async with await connect() as machine:
        c = await capture(machine, color, near=ws.measured_square_xy(square))
    p = measure(c.world_lattice, c.block_world[:2], ws, square)
    return p, c.block_color, *annotate(c, ws, p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("square", help="the square the block should be on, e.g. a3")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, help="workspace TOML (default: %(default)s)")
    parser.add_argument("--from", dest="src", metavar="SQUARE", help="where the block was picked up (for naming)")
    parser.add_argument("--color", choices=["any", *sorted(BLOCK_COLORS)], default="any",
                        help="only consider blocks of this color (default: any)")
    kind = parser.add_mutually_exclusive_group()
    kind.add_argument("--recheck", action="store_true", help="re-measuring a block already logged (not a new trial)")
    kind.add_argument("--start", action="store_true", help="where the block starts, before the next trial")
    args = parser.parse_args()
    ws = workspace.load(args.workspace)
    try:
        workspace.parse_square(args.square)
        p, color, photo, crop = asyncio.run(run(ws, args.square, args.color))
    except (LocateError, ValueError) as e:
        raise SystemExit(f"Could not check placement: {e}")

    now = datetime.now()
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    tag = "recheck" if args.recheck else "start" if args.start else ""
    trial = trial_number(tag)
    photo_path = PHOTO_DIR / photo_name(trial, tag, color, args.src, p, now)
    cv2.imwrite(str(photo_path), photo)

    with LOG_PATH.open("a") as f:
        record = {**asdict(p), "placement_offset_in_use": list(ws.placement_offset_mm),
                  "placement_gradient_in_use": [list(row) for row in ws.placement_gradient],
                  "trial": trial, "from": args.src, "color": color,
                  "photo": str(photo_path), "crop_px": crop, "recheck": args.recheck, "start": args.start,
                  "recorded_at": now.isoformat(timespec="seconds")}
        f.write(json.dumps(record) + "\n")

    print(f"Trial {trial}{f' ({tag})' if tag else ''}: {color} block is {p.distance_mm:.1f} mm from the center of {p.square}:")
    print(f"  {p.along_files_mm:+.1f} mm along files (+ = toward h)")
    print(f"  {p.along_ranks_mm:+.1f} mm along ranks (+ = toward rank 8)")
    print(f"  ({p.corners_used} corners; this view's mat is {p.view_shift_mm:.1f} mm from the configured one)")
    if args.start:
        print(f"Logged as trial {trial}'s starting position; photo {photo_path}")
        return
    cf, cr = ws.correction_at(p.square)
    print(f"Correction in use at {p.square}: [{cf:.1f}, {cr:.1f}] mm. To refit across the mat: uv run fit-correction")
    print(f"Logged to {LOG_PATH}; photo {photo_path}")


if __name__ == "__main__":
    main()
