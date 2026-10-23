"""Print (and optionally save) the gripper frame's pose in `world`. Read-only: never moves the robot.

Use it while positioning the arm by hand in manual mode:
    uv run record-pose                    # print current pose
    uv run record-pose grid_top_left      # print and save under a name
    uv run record-pose --list             # show saved poses
"""

import argparse
import asyncio
import json
import math
from datetime import datetime
from pathlib import Path

from viam.components.arm import Arm
from viam.proto.common import Pose, PoseInFrame

from robotic_experiments.connection import connect

ARM = "arm"
GRIPPER = "gripper"
FRAME = "world"
SAVE_PATH = Path("config/recorded_poses.json")
SETTLE_SECONDS = 1.0
MAX_DRIFT_MM = 0.5


async def gripper_pose(machine) -> Pose:
    return (await machine.transform_pose(PoseInFrame(reference_frame=GRIPPER, pose=Pose(o_z=1)), FRAME)).pose


def fmt(p: dict) -> str:
    return (f"x={p['x']:7.1f}  y={p['y']:7.1f}  z={p['z']:7.1f}   "
            f"dir=({p['o_x']:.2f}, {p['o_y']:.2f}, {p['o_z']:.2f})  theta={p['theta']:6.1f}")


def load() -> dict:
    return json.loads(SAVE_PATH.read_text()) if SAVE_PATH.exists() else {}


async def capture() -> dict:
    async with await connect() as machine:
        if await Arm.from_robot(machine, ARM).is_moving():
            raise SystemExit("Arm is moving. Wait for it to stop, then try again.")
        first = await gripper_pose(machine)
        await asyncio.sleep(SETTLE_SECONDS)
        second = await gripper_pose(machine)
    drift = math.dist((first.x, first.y, first.z), (second.x, second.y, second.z))
    if drift > MAX_DRIFT_MM:
        raise SystemExit(f"Gripper moved {drift:.1f} mm during the check. Hold still and try again.")
    return {k: round(getattr(second, k), 2) for k in ("x", "y", "z", "o_x", "o_y", "o_z", "theta")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", nargs="?", help="save under this name, e.g. grid_top_left, pickup_0, grip")
    parser.add_argument("--force", action="store_true", help="overwrite an existing name")
    parser.add_argument("--list", action="store_true", help="list saved poses and exit")
    args = parser.parse_args()

    saved = load()
    if args.list:
        for name, p in sorted(saved.items()):
            print(f"  {name:16s} {fmt(p)}   ({p['recorded_at']})")
        return
    if args.name and args.name in saved and not args.force:
        raise SystemExit(f"{args.name!r} already saved. Use --force to overwrite.")

    pose = asyncio.run(capture())
    print(f"gripper in {FRAME}: {fmt(pose)}")
    if pose["o_z"] > -0.99:
        print("note: tool is not pointing straight down (o_z should be about -1.00)")
    if args.name:
        saved[args.name] = {**pose, "recorded_at": datetime.now().isoformat(timespec="seconds")}
        SAVE_PATH.write_text(json.dumps(saved, indent=2) + "\n")
        print(f"saved as {args.name!r} in {SAVE_PATH}")


if __name__ == "__main__":
    main()
