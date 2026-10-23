"""Offline preview of a construction plan. Never connects to the robot.

    uv run blocks-dry-run --preset L
    uv run blocks-dry-run --grid "#..|#..|###"
    uv run blocks-dry-run --transfer a1 a3
    uv run blocks-dry-run --transfer a1 e3 c1 a7     # several moves, in order
    uv run blocks-dry-run --hover a1
"""

import argparse
import asyncio
import dataclasses
import sys

from robotic_experiments.blocks import workspace
from robotic_experiments.blocks.design import PRESETS, Design
from robotic_experiments.blocks.planner import (
    Loose, Plan, PlanError, plan, plan_goto, plan_hover, plan_open_gripper, plan_transfer,
)

DEFAULT_WORKSPACE = "config/workspace.toml"
DEFAULT_HOVER_Z = 150.0


def add_design_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preset", choices=sorted(PRESETS), help="built-in pattern")
    group.add_argument("--grid", help='pattern rows separated by "|", e.g. "#..|#..|###"')
    group.add_argument("--transfer", nargs="+", metavar="SQUARE", help="FROM TO pairs, e.g. a1 a3 or a1 e3 c1 a7")
    group.add_argument("--hover", metavar="SQUARE", help="go above a square and stop, for measuring")
    group.add_argument("--open-gripper", action="store_true", help="only open the fingers to the configured position")
    group.add_argument("--goto", nargs=3, type=float, metavar=("X", "Y", "Z"), help="move the gripper to a point, e.g. to aim the camera")
    parser.add_argument("--grid-top-left", metavar="SQUARE", help="override the build grid's top-left square")
    parser.add_argument("--grid-stride", type=int, help="override squares between design cells (1 = adjacent)")
    parser.add_argument("--find", action="store_true",
                        help="pick up blocks found by the camera in the supply files (arm must be at the find pose)")
    parser.add_argument("--pickups", nargs="+", metavar="SQUARE", help="override the pickup squares, in order")
    parser.add_argument("--hover-z", type=float, default=DEFAULT_HOVER_Z, help="gripper z for --hover (default: %(default)s)")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, help="workspace TOML (default: %(default)s)")


def plan_from_args(args: argparse.Namespace, ws: workspace.Workspace) -> Plan:
    if args.grid_top_left or args.grid_stride or args.pickups:
        for sq in args.pickups or []:
            workspace.parse_square(sq)
        ws = dataclasses.replace(
            ws,
            grid_top_left=args.grid_top_left or ws.grid_top_left,
            grid_stride=args.grid_stride or ws.grid_stride,
            pickup_squares=tuple(args.pickups) if args.pickups else ws.pickup_squares,
        )
    if args.transfer:
        if len(args.transfer) % 2:
            raise ValueError("--transfer takes FROM TO pairs")
        for sq in args.transfer:
            workspace.parse_square(sq)
        return plan_transfer(ws, list(zip(args.transfer[::2], args.transfer[1::2])))
    if args.goto:
        return plan_goto(ws, *args.goto)
    if args.open_gripper:
        return plan_open_gripper()
    if args.hover:
        workspace.parse_square(args.hover)
        return plan_hover(ws, args.hover, args.hover_z)
    design = Design.preset(args.preset) if args.preset else Design.parse(args.grid)
    if not args.find:
        return plan(design, ws)
    return plan(design, ws, find_loose(ws))


def find_loose(ws: workspace.Workspace) -> list[Loose]:
    """Find blocks with the camera (read-only) and list them for the planner, pickable ones first."""
    from robotic_experiments.blocks import find_cubes  # needs the camera stack; only load when used

    try:
        found, photo = asyncio.run(find_cubes.find(ws, require_find_pose=True))
    except find_cubes.LocateError as e:
        raise ValueError(f"camera: {e}") from e
    print(f"Camera found {len(found)} block(s):")
    print(find_cubes.table(found, ws))
    print(f"Photo: {find_cubes.save_photo(photo, len(found))}\n")
    return [Loose(f"{c.color} {c.square}", c.f, c.r, c.usable(ws) is None) for c in found]


def describe(p: Plan, ws: workspace.Workspace) -> str:
    lines = []
    if not ws.calibrated:
        lines += ["!! Workspace is NOT marked calibrated: only a high --hover can run live.", ""]
    if p.design is not None:
        lines += ["Design:", p.design.render(), ""]
        lines.append(f"Blocks: {len(p.assignments)}   (pickup spots available: {len(ws.pickups)})")
    for a in p.assignments:
        cell = "" if a.cell is None else f" = cell (row {a.cell[0]}, col {a.cell[1]})"
        lines.append(
            f"  block {a.block}: pickup {a.pickup_square} ({a.pickup_xy[0]:.1f}, {a.pickup_xy[1]:.1f})"
            f"  ->  {a.cell_square}{cell} ({a.cell_xy[0]:.1f}, {a.cell_xy[1]:.1f})"
        )
    lines += ["", f"Steps (gripper-frame targets in '{ws.frame}', mm):"]
    for i, s in enumerate(p.steps):
        tag = f"[b{s.block}]" if s.block is not None else "    "
        target = f"  {s.target}" if s.target else ""
        lines.append(f"  {i:3d} {tag} {s.kind:7s} {s.label:38s}{target}")
    lines += ["", "Dry run only checks inputs and bounds; it does not prove the motion is reachable or collision-free."]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_design_args(parser)
    args = parser.parse_args()
    try:
        ws = workspace.load(args.workspace)
        p = plan_from_args(args, ws)
    except (ValueError, PlanError) as e:
        sys.exit(f"Invalid: {e}")
    print(describe(p, ws))


if __name__ == "__main__":
    main()
