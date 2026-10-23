"""Offline preview of a construction plan. Never connects to the robot.

    uv run blocks-dry-run --preset L
    uv run blocks-dry-run --grid "#..|#..|###"
    uv run blocks-dry-run --transfer a1 a3
    uv run blocks-dry-run --hover a1
"""

import argparse
import sys

from robotic_experiments.blocks import workspace
from robotic_experiments.blocks.design import PRESETS, Design
from robotic_experiments.blocks.planner import Plan, PlanError, plan, plan_hover, plan_transfer

DEFAULT_WORKSPACE = "config/workspace.toml"
DEFAULT_HOVER_Z = 150.0


def add_design_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preset", choices=sorted(PRESETS), help="built-in pattern")
    group.add_argument("--grid", help='pattern rows separated by "|", e.g. "#..|#..|###"')
    group.add_argument("--transfer", nargs=2, metavar=("FROM", "TO"), help="move one block between squares, e.g. a1 a3")
    group.add_argument("--hover", metavar="SQUARE", help="go above a square and stop, for measuring")
    parser.add_argument("--hover-z", type=float, default=DEFAULT_HOVER_Z, help="gripper z for --hover (default: %(default)s)")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, help="workspace TOML (default: %(default)s)")


def plan_from_args(args: argparse.Namespace, ws: workspace.Workspace) -> Plan:
    if args.transfer:
        src, dst = args.transfer
        workspace.parse_square(src), workspace.parse_square(dst)
        return plan_transfer(ws, src, dst)
    if args.hover:
        workspace.parse_square(args.hover)
        return plan_hover(ws, args.hover, args.hover_z)
    return plan(Design.preset(args.preset) if args.preset else Design.parse(args.grid), ws)


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
