"""Deterministic construction planner: design + workspace -> list of steps. No robot access."""

import math
from dataclasses import dataclass

from robotic_experiments.blocks.design import Design
from robotic_experiments.blocks.workspace import Workspace


@dataclass(frozen=True)
class Target:
    """Gripper-frame pose in the workspace frame, tool pointing straight down."""

    x: float
    y: float
    z: float
    theta: float

    def __str__(self) -> str:
        return f"x={self.x:7.1f} y={self.y:7.1f} z={self.z:7.1f} θ={self.theta:6.1f}"


@dataclass(frozen=True)
class Step:
    kind: str  # "move" | "open" | "grab" | "release" | "check_released"
    label: str
    block: int | None = None  # index of the block this step belongs to
    target: Target | None = None


@dataclass(frozen=True)
class Assignment:
    block: int
    pickup_index: int
    pickup_square: str
    pickup_xy: tuple[float, float]
    cell: tuple[int, int] | None  # design cell, None for a plain square-to-square transfer
    cell_square: str
    cell_xy: tuple[float, float]


@dataclass(frozen=True)
class Plan:
    design: Design | None
    assignments: tuple[Assignment, ...]
    steps: tuple[Step, ...]


class PlanError(ValueError):
    pass


def plan(design: Design, ws: Workspace, recenter: bool = False) -> Plan:
    """Build a 3x3 design: fill cells row by row from pickup squares in order."""
    problems = ws.problems()
    if problems:
        raise PlanError("; ".join(problems))
    filled = design.filled()
    if not filled:
        problems.append("design has no filled cells")
    if len(filled) > len(ws.pickups):
        problems.append(f"design needs {len(filled)} blocks but only {len(ws.pickups)} pickup spots are configured")
    if problems:
        raise PlanError("; ".join(problems))

    assignments = tuple(
        Assignment(i, i, ws.pickup_squares[i], ws.pickups[i], cell, ws.cell_square(*cell), ws.cell_xy(*cell))
        for i, cell in enumerate(filled)
    )
    if recenter:
        _check_recenter_square(ws, list(ws.pickup_squares[: len(filled)]) + [a.cell_square for a in assignments])
    return Plan(design, assignments, _pick_and_place_steps(ws, assignments, recenter))


def plan_transfer(ws: Workspace, moves: list[tuple[str, str]], recenter: bool = False) -> Plan:
    """Move blocks between mat squares, in order, ignoring the build grid."""
    problems = ws.mat_problems()
    for i, (src, dst) in enumerate(moves):
        if src == dst:
            problems.append(f"move {i}: source and destination are the same square")
        if dst in [d for _, d in moves[:i]] or src in [s for s, _ in moves[:i]]:
            problems.append(f"move {i}: {src} or {dst} is already used by an earlier move")
        if dst in [s for s, _ in moves[i + 1:]]:
            problems.append(f"move {i}: {dst} still holds a block that a later move picks up")
    # Every placed block, and every block still waiting, must clear the open fingers: fully open
    # at each pickup, partly open (open_position) at each drop.
    for i, (src, dst) in enumerate(moves):
        neighbours = [d for _, d in moves[:i]] + [s for s, _ in moves[i + 1:]]
        for square, span, what in ((src, ws.full_open_span_mm, "pickup"), (dst, ws.open_finger_span_mm, "drop")):
            for n in neighbours:
                if why := ws.finger_conflict(square, n, span):
                    problems.append(f"move {i}: {what} {square}: block on {why}")
    if problems:
        raise PlanError("; ".join(problems))
    assignments = tuple(
        Assignment(i, i, src, ws.square_xy(src), None, dst, ws.square_xy(dst)) for i, (src, dst) in enumerate(moves)
    )
    if recenter:
        _check_recenter_square(ws, [sq for move in moves for sq in move])
    return Plan(None, assignments, _pick_and_place_steps(ws, assignments, recenter))


def _check_recenter_square(ws: Workspace, used: list[str]) -> None:
    """The wrist turns 90 degrees with open fingers on the re-centering square: keep everything clear."""
    sq = ws.recenter_square
    if sq is None:
        raise PlanError("--recenter needs [pickups] recenter_square in the workspace")
    need = ws.turning_clearance_mm()
    close = [f"{u} ({math.dist(ws.measured_square_xy(sq), ws.measured_square_xy(u)):.0f} mm)" for u in dict.fromkeys(used)
             if math.dist(ws.measured_square_xy(sq), ws.measured_square_xy(u)) < need]
    if close:
        raise PlanError(f"re-centering square {sq} must be {need:.0f} mm from every block; too close: {', '.join(close)}")


def plan_hover(ws: Workspace, square: str, z: float) -> Plan:
    """Go above a square and stop there, for measuring the fingertip gap and finger alignment."""
    problems = ws.mat_problems()
    if problems:
        raise PlanError("; ".join(problems))
    xy = ws.square_xy(square)
    steps = (
        Step("move", "move to safe start", target=Target(*ws.safe_start, ws.theta_deg)),
        Step("move", f"travel above {square}", target=Target(*xy, max(z, ws.travel_z), ws.theta_deg)),
        Step("move", f"hover above {square} at z={z:.0f}", target=Target(*xy, z, ws.theta_deg)),
    )
    _check_bounds(ws, steps)
    return Plan(None, (), steps)


def plan_open_gripper() -> Plan:
    """Only open the fingers (to the configured open_position), e.g. to measure the gap."""
    return Plan(None, (), (Step("release", "open fingers to the configured release position"),))


def plan_goto(ws: Workspace, x: float, y: float, z: float) -> Plan:
    """Move the gripper to a point, tool down at the configured finger angle (e.g. to aim the camera)."""
    steps = (Step("move", f"go to x={x:.0f} y={y:.0f} z={z:.0f}", target=Target(x, y, z, ws.theta_deg)),)
    _check_bounds(ws, steps)
    return Plan(None, (), steps)


def turned(theta: float) -> float:
    """theta + 90 degrees, normalised to (-180, 180]."""
    t = (theta + 90.0) % 360.0
    return t - 360.0 if t > 180.0 else t


def _recenter_steps(ws: Workspace, b: int) -> list[Step]:
    """Let the block go on the re-centering square, grab it once with the wrist turned 90 degrees
    (centering it along the other axis), then grab it again at the normal angle."""
    xy = ws.square_xy(ws.recenter_square)
    sq = ws.recenter_square
    above = ws.grip_z + ws.approach_height_mm
    t0, t1 = ws.theta_deg, turned(ws.theta_deg)

    def at(z, theta):
        return Target(xy[0], xy[1], z, theta)

    return [
        Step("move", f"travel above re-centering square {sq}", b, at(ws.travel_z, t0)),
        Step("move", f"approach {sq}", b, at(above, t0)),
        Step("move", f"lower block onto {sq}", b, at(ws.grip_z + ws.place_release_offset_mm, t0)),
        Step("open", "let go (fully open)", b),
        Step("move", "rise to approach height", b, at(above, t0)),
        Step("check_released", "close fingers to confirm the block was let go", b),
        Step("move", "turn the wrist 90 degrees", b, at(above, t1)),
        Step("open", "open fingers", b),
        Step("move", "descend to grip, turned", b, at(ws.grip_z, t1)),
        Step("grab", "close fingers: centers the block along the other axis", b),
        Step("open", "let go (fully open)", b),
        Step("move", "rise to approach height", b, at(above, t1)),
        Step("check_released", "close fingers to confirm the block was let go", b),
        Step("move", "turn the wrist back", b, at(above, t0)),
        Step("open", "open fingers", b),
        Step("move", f"descend to grip at {sq}", b, at(ws.grip_z, t0)),
        Step("grab", "close fingers and verify grasp", b),
        Step("move", "lift to approach height", b, at(above, t0)),
        Step("move", "lift to travel height", b, at(ws.travel_z, t0)),
    ]


def _pick_and_place_steps(ws: Workspace, assignments, recenter: bool = False) -> tuple[Step, ...]:
    def at(xy, z):
        return Target(xy[0], xy[1], z, ws.theta_deg)

    grip_z = ws.grip_z
    above = grip_z + ws.approach_height_mm
    release_z = grip_z + ws.place_release_offset_mm

    steps = [Step("move", "move to safe start", target=Target(*ws.safe_start, ws.theta_deg))]
    for a in assignments:
        b = a.block
        src = f"pickup {a.pickup_square}"
        dst = a.cell_square if a.cell is None else f"{a.cell_square} (cell {a.cell[0]},{a.cell[1]})"
        steps += [
            Step("move", f"travel above {src}", b, at(a.pickup_xy, ws.travel_z)),
            Step("move", f"approach {src}", b, at(a.pickup_xy, above)),
            Step("open", "open fingers", b),
            Step("move", f"descend to grip at {src}", b, at(a.pickup_xy, grip_z)),
            Step("grab", "close fingers and verify grasp", b),
            Step("move", "lift to approach height", b, at(a.pickup_xy, above)),
            Step("move", "lift to travel height", b, at(a.pickup_xy, ws.travel_z)),
            *(_recenter_steps(ws, b) if recenter else []),
            Step("move", f"travel above {dst}", b, at(a.cell_xy, ws.travel_z)),
            Step("move", f"approach {dst}", b, at(a.cell_xy, above)),
            Step("move", f"lower block onto {dst}", b, at(a.cell_xy, release_z)),
            Step("release", "open fingers to release", b),
            Step("move", "retreat upward", b, at(a.cell_xy, above)),
            Step("check_released", "close fingers to confirm the block was let go", b),
            Step("move", "retreat to travel height", b, at(a.cell_xy, ws.travel_z)),
        ]
    steps.append(Step("move", "return to safe start", target=Target(*ws.safe_start, ws.theta_deg)))
    _check_bounds(ws, steps)
    return tuple(steps)


def _check_bounds(ws: Workspace, steps) -> None:
    out_of_bounds = [
        f"{s.label} ({s.target})"
        for s in steps
        if s.target
        and not all(lo <= v <= hi for v, lo, hi in zip((s.target.x, s.target.y, s.target.z), ws.bounds_min, ws.bounds_max))
    ]
    if out_of_bounds:
        raise PlanError("targets outside configured workspace bounds: " + "; ".join(out_of_bounds))
