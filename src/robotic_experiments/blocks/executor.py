"""Guarded live execution of a construction plan. This is the only module that moves hardware.

    uv run blocks-run --hover a1                      # go 150 mm above a1 and stop, to measure
    uv run blocks-run --goto 314 -392 600             # move the gripper to a point (e.g. aim the camera)
    uv run blocks-run --transfer a1 a3                # move one block from a1 to a3
    uv run blocks-run --transfer a1 e3 c1 a7 --max-blocks 2   # several moves, in order
    uv run blocks-run --preset one                    # build a design
    uv run blocks-run --preset one --stop-after 2     # stop after step 2 (e.g. to check the approach)

Safety rules:
- Refuses to run unless the workspace is marked calibrated. The exceptions are --hover and
  --goto, which must stay at least MIN_UNCALIBRATED_HOVER_MM above the mat surface.
- Refuses plans with more blocks than --max-blocks (default 1, the first milestone).
- Checks the machine before moving: components present and healthy, obstacles configured,
  arm still, gripper empty.
- Asks for a typed "yes" before the first movement (--approved skips the prompt; use it only
  when a person at the arm has approved this exact run).
- Every call has a timeout. On any failure, timeout or Ctrl-C it stops the arm and gripper
  and exits. It never retries, never opens the gripper, never makes a recovery move.
"""

import argparse
import asyncio
import math
import sys

from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.proto.service.motion import Constraints, LinearConstraint
from viam.services.motion import MotionClient

from robotic_experiments.blocks import workspace
from robotic_experiments.blocks.dry_run import add_design_args, describe, plan_from_args
from robotic_experiments.blocks.planner import Plan, PlanError, Step, Target
from robotic_experiments.connection import connect

ARM = "arm"
GRIPPER = "gripper"
MOTION = "builtin"
MIN_UNCALIBRATED_HOVER_MM = 150.0  # gripper frame above the mat surface
VERTICAL_XY_TOLERANCE_MM = 3.0  # moves that change x/y less than this are straight up/down

STRAIGHT_LINE = Constraints(linear_constraint=[LinearConstraint(line_tolerance_mm=2.0, orientation_tolerance_degs=2.0)])


class Abort(Exception):
    pass


def target_pose(t: Target) -> Pose:
    return Pose(x=t.x, y=t.y, z=t.z, o_x=0, o_y=0, o_z=-1, theta=t.theta)


async def bounded(coro, seconds: float, what: str):
    try:
        return await asyncio.wait_for(coro, seconds)
    except asyncio.TimeoutError:
        raise Abort(f"{what} timed out after {seconds:.0f}s")


async def gripper_pose(machine, frame: str) -> Pose:
    return (await machine.transform_pose(PoseInFrame(reference_frame=GRIPPER, pose=Pose(o_z=1)), frame)).pose


async def preflight(machine, ws: workspace.Workspace, allow_held_block: bool = False) -> None:
    names = {(rn.subtype, rn.name) for rn in machine.resource_names}
    missing = [n for n in [("arm", ARM), ("gripper", GRIPPER), ("motion", MOTION)] if n not in names]
    missing += [("obstacle", o) for o in ws.required_obstacles if not any(name == o for _, name in names)]
    if missing:
        raise Abort(f"missing on machine: {missing}")

    status = await machine.get_machine_status()
    watched = {ARM, GRIPPER, MOTION, *ws.required_obstacles}
    broken = [f"{s.name.name}: {s.error}" for s in status.resources if s.name.name in watched and s.error]
    if broken:
        raise Abort("unhealthy resources: " + "; ".join(broken))

    if await Arm.from_robot(machine, ARM).is_moving():
        raise Abort("arm is moving")
    holding = await Gripper.from_robot(machine, GRIPPER).is_holding_something()
    if holding.is_holding_something and not parked_open(holding, ws) and not allow_held_block:
        raise Abort("gripper reports it is already holding something")

    p = await gripper_pose(machine, ws.frame)
    print(f"preflight ok. gripper now at x={p.x:.1f} y={p.y:.1f} z={p.z:.1f} (o_z={p.o_z:.2f})")


def parked_open(holding, ws) -> bool:
    """True when "holding" only reflects fingers resting at the configured partial opening.

    In the module's G1 mode, holding means the position is between fully closed (10) and fully
    open (830), so a partial opening always reads as holding. A held 30 mm cube sits near 220,
    well away from open_position.
    """
    pos = (holding.meta or {}).get("position")
    if pos is None:
        return False
    # release openings in use: the workspace default, and one per block size gripped (30 / 60 mm)
    parked = [p for p in (ws.open_position, ws.release_position(30.0), ws.release_position(60.0)) if p is not None]
    return any(abs(pos - p) <= 2 * GRIPPER_POSITION_TOLERANCE + 5 for p in parked)


async def run_step(machine, ws, s: Step, prev: Target | None, move_timeout: float, grip_timeout: float) -> None:
    gripper = Gripper.from_robot(machine, GRIPPER)
    if s.kind == "move":
        t = s.target
        # Straight up/down moves keep a straight line; a pure wrist turn (same point, new theta) can't.
        vertical = (prev is not None and math.dist((prev.x, prev.y), (t.x, t.y)) <= VERTICAL_XY_TOLERANCE_MM
                    and abs(prev.z - t.z) > 1.0 and abs((prev.theta - t.theta + 180) % 360 - 180) < 1.0)
        motion = MotionClient.from_robot(machine, MOTION)
        ok = await bounded(
            motion.move(
                component_name=GRIPPER,
                destination=PoseInFrame(reference_frame=ws.frame, pose=target_pose(t)),
                constraints=STRAIGHT_LINE if vertical else None,
                timeout=move_timeout,
            ),
            move_timeout + 5,
            s.label,
        )
        if not ok:
            raise Abort(f"motion service reported failure for: {s.label}")
        p = await bounded(gripper_pose(machine, ws.frame), 10, "reading gripper pose")
        err = math.dist((p.x, p.y, p.z), (t.x, t.y, t.z))
        if err > ws.arrival_tolerance_mm:
            raise Abort(f"{s.label}: ended {err:.1f} mm from target (tolerance {ws.arrival_tolerance_mm} mm)")
    elif s.kind == "open":  # before a pickup: fully open, so an off-center block still fits
        await bounded(gripper.open(timeout=grip_timeout), grip_timeout + 5, s.label)
    elif s.kind == "release":
        if s.position is not None:
            await bounded(open_partly(gripper, s.position), grip_timeout + 5, s.label)
        elif ws.open_position is None:
            await bounded(gripper.open(timeout=grip_timeout), grip_timeout + 5, s.label)
        else:
            await bounded(open_partly(gripper, ws.open_position), grip_timeout + 5, s.label)
    elif s.kind == "check_released":
        # In G1 mode "holding" only reflects finger position, so test physically: well above the
        # placed block, close the fingers. They close fully on air; they stop on a stuck block.
        await bounded(gripper.grab(timeout=grip_timeout), grip_timeout + 5, s.label)
        pos = (await bounded(gripper.do_command({"get": True}), 10, "reading gripper position"))["pos"]
        if pos > STUCK_BLOCK_MIN_POSITION:
            raise Abort(f"block still stuck in the gripper after release (fingers stopped at {pos:.0f})")
    elif s.kind == "grab":
        grabbed = await bounded(gripper.grab(timeout=grip_timeout), grip_timeout + 5, s.label)
        holding = await bounded(gripper.is_holding_something(), 10, "checking grasp")
        if not (grabbed and holding.is_holding_something):
            raise Abort(f"grasp failed (grab={grabbed}, holding={holding.is_holding_something}, meta={holding.meta})")
    else:
        raise Abort(f"unknown step kind {s.kind!r}")


GRIPPER_POSITION_TOLERANCE = 20.0  # gripper units (~2 mm)
STUCK_BLOCK_MIN_POSITION = 100.0  # fingers stopping above this (~18 mm gap) are on a block, not closed


async def open_partly(gripper, position: float) -> None:
    """Open to `position` with the module's DoCommand and wait until the fingers get there."""
    await gripper.do_command({"set": float(position)})
    while True:
        pos = (await gripper.do_command({"get": True}))["pos"]
        if abs(pos - position) <= GRIPPER_POSITION_TOLERANCE:
            return
        await asyncio.sleep(0.1)


async def emergency_stop(machine) -> None:
    for cls, name in ((Arm, ARM), (Gripper, GRIPPER)):
        try:
            await asyncio.wait_for(cls.from_robot(machine, name).stop(), 5)
            print(f"  {name} stopped")
        except BaseException as e:  # keep trying the others
            print(f"  could not stop {name}: {e!r}")


async def execute(p: Plan, ws, stop_after: int | None, move_timeout: float, grip_timeout: float, approved: bool,
                  allow_held_block: bool = False) -> int:
    async with await connect() as machine:
        try:
            await preflight(machine, ws, allow_held_block)
        except Abort as e:
            print(f"ABORT before moving: {e}")
            return 1

        last = len(p.steps) - 1 if stop_after is None else stop_after
        print(f"\nWill execute steps 0..{last} of {len(p.steps) - 1}. The arm WILL move.")
        print("Make sure someone is at the arm with the e-stop in reach.")
        if approved:
            print("Approved on the command line.")
        elif input('Type "yes" to start: ').strip() != "yes":
            print("Not confirmed. Nothing moved.")
            return 1

        # Start from where the gripper actually is, so a first move straight up or down is
        # also kept to a vertical line.
        here = await gripper_pose(machine, ws.frame)
        prev = Target(here.x, here.y, here.z, here.theta)
        i = -1
        try:
            for i, s in enumerate(p.steps[: last + 1]):
                print(f"  {i:3d} {s.kind:7s} {s.label}", flush=True)
                await run_step(machine, ws, s, prev, move_timeout, grip_timeout)
                prev = s.target or prev
        except (Abort, Exception, asyncio.CancelledError, KeyboardInterrupt) as e:
            reason = "interrupted" if isinstance(e, (asyncio.CancelledError, KeyboardInterrupt)) else str(e) or repr(e)
            print(f"\nSTOPPED at step {i}: {reason}")
            await emergency_stop(machine)
            print("No retry, no release, no recovery move. Check the arm by hand before doing anything else.")
            return 1

        if p.assignments:
            print(f"\nDone through step {last}. Check by eye that the block is where it should be.")
        else:
            print(f"\nDone through step {last}. The arm is holding position.")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_design_args(parser)
    parser.add_argument("--max-blocks", type=int, default=1, help="refuse designs with more blocks (default: 1)")
    parser.add_argument("--stop-after", type=int, help="stop after this step index")
    parser.add_argument("--move-timeout", type=float, default=30.0, help="seconds per arm move (default: 30)")
    parser.add_argument("--grip-timeout", type=float, default=10.0, help="seconds per gripper action (default: 10)")
    parser.add_argument("--approved", action="store_true", help="skip the typed confirmation (a supervisor approved this run)")
    parser.add_argument("--allow-held-block", action="store_true",
                        help="recovery only (--goto/--open-gripper): allow moving while a block is held")
    args = parser.parse_args()

    try:
        ws = workspace.load(args.workspace)
        p = plan_from_args(args, ws)
    except (ValueError, PlanError) as e:
        sys.exit(f"Invalid: {e}")

    print(describe(p, ws))
    print()
    if args.allow_held_block and not (args.goto or args.open_gripper):
        sys.exit("--allow-held-block is only for --goto and --open-gripper recoveries.")
    if args.open_gripper:
        pass  # fingers only; nothing near the mat moves
    elif args.hover or args.goto:
        z = args.goto[2] if args.goto else args.hover_z
        if z < ws.mat_top_z + MIN_UNCALIBRATED_HOVER_MM and not ws.calibrated:
            sys.exit(f"Refusing: uncalibrated moves must stay at z >= {ws.mat_top_z + MIN_UNCALIBRATED_HOVER_MM:.0f}.")
    elif not ws.calibrated:
        sys.exit(f"Refusing to move: {args.workspace} is not marked calibrated.")
    if len(p.assignments) > args.max_blocks:
        sys.exit(f"Refusing: design has {len(p.assignments)} blocks, --max-blocks is {args.max_blocks}.")
    if args.stop_after is not None and not 0 <= args.stop_after < len(p.steps):
        sys.exit(f"--stop-after must be between 0 and {len(p.steps) - 1}")

    sys.exit(asyncio.run(execute(p, ws, args.stop_after, args.move_timeout, args.grip_timeout, args.approved,
                                 args.allow_held_block)))


if __name__ == "__main__":
    main()
