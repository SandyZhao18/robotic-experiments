"""Offline preview of a construction plan. Never connects to the robot.

    uv run blocks-dry-run --preset L
    uv run blocks-dry-run --grid "#..|#..|###"
"""

import argparse
import sys

from robotic_experiments.blocks import workspace
from robotic_experiments.blocks.design import PRESETS, Design
from robotic_experiments.blocks.planner import Plan, PlanError, plan

DEFAULT_WORKSPACE = "config/workspace.toml"


def add_design_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preset", choices=sorted(PRESETS), help="built-in pattern")
    group.add_argument("--grid", help='pattern rows separated by "|", e.g. "#..|#..|###"')
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, help="workspace TOML (default: %(default)s)")


def design_from_args(args: argparse.Namespace) -> Design:
    return Design.preset(args.preset) if args.preset else Design.parse(args.grid)


def describe(p: Plan, ws: workspace.Workspace) -> str:
    lines = []
    if not ws.calibrated:
        lines += ["!! Workspace is NOT calibrated: positions are placeholders. Live execution is disabled.", ""]
    lines += ["Design:", p.design.render(), ""]
    lines.append(f"Blocks: {len(p.assignments)}   (pickup spots available: {len(ws.pickups)})")
    for a in p.assignments:
        lines.append(
            f"  block {a.block}: pickup {a.pickup_square} ({a.pickup_xy[0]:.1f}, {a.pickup_xy[1]:.1f})"
            f"  ->  {a.cell_square} = cell (row {a.cell[0]}, col {a.cell[1]}) ({a.cell_xy[0]:.1f}, {a.cell_xy[1]:.1f})"
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
        p = plan(design_from_args(args), ws)
    except (ValueError, PlanError) as e:
        sys.exit(f"Invalid: {e}")
    print(describe(p, ws))


if __name__ == "__main__":
    main()
