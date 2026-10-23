"""Compare the scene record (runs/scene.json) with what the camera sees. Read-only unless asked.

    uv run check-scene            # report: matched, missing, unexpected
    uv run check-scene --update   # also store the measured positions of matched blocks
    uv run check-scene --sync     # replace the record with what the camera sees

Only the build files (a-e by default) are checked, and only blocks the camera can see from above:
the top of each stack. A recorded top matches a detection if it is within MATCH_XY_MM across the
mat and MATCH_HEIGHT_MM in height. Any hand or other object at block height in the build area
counts as unexpected, so a run will not start with something in the way.
"""

import argparse
import asyncio
import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from robotic_experiments.blocks import scene, workspace
from robotic_experiments.blocks.check_placement import fit_view
from robotic_experiments.blocks.locate_mat import STRUCTURE_HEIGHT_RANGE_MM, Frame, LocateError, capture_frame
from robotic_experiments.connection import connect

DEFAULT_WORKSPACE = "config/workspace.toml"
BUILD_FILES = "abcde"
MATCH_XY_MM = 12.0
MATCH_HEIGHT_MM = 12.0


@dataclass
class Seen:
    color: str
    f: float
    r: float
    top_mm: float


@dataclass
class Report:
    matched: list[tuple[scene.SceneBlock, Seen]]
    missing: list[scene.SceneBlock]
    unexpected: list[Seen]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.unexpected


def seen_in_build_area(frame: Frame, ws: workspace.Workspace, files: str = BUILD_FILES) -> list[Seen]:
    a1_view, fs, rs = fit_view(frame.world_lattice, ws)
    to_mat = np.linalg.inv(np.column_stack([fs, rs]))
    lo, hi = STRUCTURE_HEIGHT_RANGE_MM
    out = []
    for b in frame.blocks:
        if not lo <= b.height_mm <= hi:
            continue
        f, r = to_mat @ (b.world[:2] - a1_view)
        if -0.5 <= f <= len(files) - 0.5:
            out.append(Seen(b.color, float(f), float(r), b.height_mm))
    return out


def compare(recorded: list[scene.SceneBlock], seen: list[Seen], square_mm: float) -> Report:
    tops = scene.visible_tops(recorded, square_mm)
    pool = list(seen)
    matched, missing = [], []
    for t in sorted(tops, key=lambda b: -b.top_mm):
        best = min(pool, key=lambda s: math.dist((s.f, s.r), (t.f, t.r)), default=None)
        if (best is not None and math.dist((best.f, best.r), (t.f, t.r)) * square_mm <= MATCH_XY_MM
                and abs(best.top_mm - t.top_mm) <= MATCH_HEIGHT_MM):
            matched.append((t, best))
            pool.remove(best)
        else:
            missing.append(t)
    return Report(matched, missing, pool)


def updated(recorded: list[scene.SceneBlock], report: Report) -> list[scene.SceneBlock]:
    """The record with matched tops moved to where the camera measured them."""
    moved = {id(t): s for t, s in report.matched}
    return [dataclasses.replace(b, f=moved[id(b)].f, r=moved[id(b)].r, measured=True, color=moved[id(b)].color)
            if id(b) in moved else b for b in recorded]


def synced(seen: list[Seen], ws: workspace.Workspace) -> list[scene.SceneBlock]:
    """A record built from the camera alone: each top becomes a cube-sized column from the mat up
    to its height (the camera cannot see what is underneath, so assume it is solid)."""
    out = []
    for s in seen:
        level = max(1, round(s.top_mm / ws.block_size_mm))
        out.append(scene.SceneBlock("cube", "", "files", s.f, s.r, 0.0, s.top_mm, level, measured=True, color=s.color))
    return out


def describe(report: Report, square_mm: float) -> str:
    def where(f, r):
        sq = workspace.square_name(round(f), round(r)) if 0 <= round(f) < 8 and 0 <= round(r) < 8 else "?"
        return f"{sq} ({(f - round(f)) * square_mm:+.0f},{(r - round(r)) * square_mm:+.0f} mm)"

    lines = []
    for t, s in report.matched:
        off = math.dist((t.f, t.r), (s.f, s.r)) * square_mm
        lines.append(f"  ok        {t.shape:8s} top {t.top_mm:3.0f} mm at {where(s.f, s.r)}  ({off:.1f} mm from the record)")
    for t in report.missing:
        lines.append(f"  MISSING   {t.shape:8s} top {t.top_mm:3.0f} mm expected at {where(t.f, t.r)}")
    for s in report.unexpected:
        lines.append(f"  UNEXPECTED {s.color} top {s.top_mm:3.0f} mm at {where(s.f, s.r)}")
    return "\n".join(lines) or "  (nothing recorded, nothing seen)"


async def check(ws: workspace.Workspace) -> tuple[Report, list[Seen]]:
    async with await connect() as machine:
        frame = await capture_frame(machine)
    seen = seen_in_build_area(frame, ws)
    return compare(scene.load(), seen, ws.square_size_mm), seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--update", action="store_true", help="store measured positions of matched blocks")
    mode.add_argument("--sync", action="store_true", help="replace the record with what the camera sees")
    args = parser.parse_args()
    ws = workspace.load(args.workspace)
    try:
        report, seen = asyncio.run(check(ws))
    except LocateError as e:
        raise SystemExit(f"Could not check the scene: {e}")

    print(f"Scene record: {len(scene.load())} block(s); camera sees {len(seen)} top(s) in files {BUILD_FILES}:")
    print(describe(report, ws.square_size_mm))
    if args.sync:
        scene.save(synced(seen, ws))
        print(f"Record replaced with {len(seen)} block(s) from the camera; it matches what the camera sees now.")
        return
    elif args.update and report.matched:
        scene.save(updated(scene.load(), report))
        print(f"Stored measured positions for {len(report.matched)} block(s).")
    print("Scene matches." if report.ok else "Scene does NOT match the record.")


if __name__ == "__main__":
    main()
