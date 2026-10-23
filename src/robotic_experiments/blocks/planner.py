"""Deterministic construction planner: design + workspace -> list of steps. No robot access."""

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
    kind: str  # "move" | "open" | "grab" | "release"
    label: str
    block: int | None = None  # index of the block this step belongs to
    target: Target | None = None


@dataclass(frozen=True)
class Assignment:
    block: int
    pickup_index: int
    pickup_xy: tuple[float, float]
    cell: tuple[int, int]
    cell_xy: tuple[float, float]


@dataclass(frozen=True)
class Plan:
    design: Design
    assignments: tuple[Assignment, ...]
    steps: tuple[Step, ...]


class PlanError(ValueError):
    pass


def plan(design: Design, ws: Workspace) -> Plan:
    problems = ws.problems()
    filled = design.filled()
    if not filled:
        problems.append("design has no filled cells")
    if len(filled) > len(ws.pickups):
        problems.append(f"design needs {len(filled)} blocks but only {len(ws.pickups)} pickup spots are configured")
    if problems:
        raise PlanError("; ".join(problems))

    assignments = tuple(
        Assignment(i, i, ws.pickups[i], cell, ws.cell_xy(*cell)) for i, cell in enumerate(filled)
    )

    def at(xy, z):
        return Target(xy[0], xy[1], z, ws.theta_deg)

    grip_z = ws.grip_z
    above = grip_z + ws.approach_height_mm
    release_z = grip_z + ws.place_release_offset_mm

    steps = [Step("move", "move to safe start", target=Target(*ws.safe_start, ws.theta_deg))]
    for a in assignments:
        b = a.block
        src = f"pickup {a.pickup_index}"
        dst = f"cell (row {a.cell[0]}, col {a.cell[1]})"
        steps += [
            Step("move", f"travel above {src}", b, at(a.pickup_xy, ws.travel_z)),
            Step("move", f"approach {src}", b, at(a.pickup_xy, above)),
            Step("open", "open fingers", b),
            Step("move", f"descend to grip at {src}", b, at(a.pickup_xy, grip_z)),
            Step("grab", "close fingers and verify grasp", b),
            Step("move", "lift to approach height", b, at(a.pickup_xy, above)),
            Step("move", "lift to travel height", b, at(a.pickup_xy, ws.travel_z)),
            Step("move", f"travel above {dst}", b, at(a.cell_xy, ws.travel_z)),
            Step("move", f"approach {dst}", b, at(a.cell_xy, above)),
            Step("move", f"lower block onto {dst}", b, at(a.cell_xy, release_z)),
            Step("release", "open fingers to release", b),
            Step("move", "retreat upward", b, at(a.cell_xy, above)),
            Step("move", "retreat to travel height", b, at(a.cell_xy, ws.travel_z)),
        ]
    steps.append(Step("move", "return to safe start", target=Target(*ws.safe_start, ws.theta_deg)))

    out_of_bounds = [
        f"{s.label} ({s.target})"
        for s in steps
        if s.target
        and not all(lo <= v <= hi for v, lo, hi in zip((s.target.x, s.target.y, s.target.z), ws.bounds_min, ws.bounds_max))
    ]
    if out_of_bounds:
        raise PlanError("targets outside configured workspace bounds: " + "; ".join(out_of_bounds))

    return Plan(design, assignments, tuple(steps))
