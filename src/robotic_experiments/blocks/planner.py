"""Deterministic construction planner: design + workspace -> list of steps. No robot access."""

import math
from dataclasses import dataclass

from robotic_experiments.blocks.design import Design
from robotic_experiments.blocks.workspace import Workspace, parse_square


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
    position: float | None = None  # gripper position for "release" (None: the workspace default)
    record: object | None = None  # scene.SceneBlock to record once this step succeeds
    expect: object | None = None  # scene.SceneBlock the camera must find for "verify_placed"


@dataclass(frozen=True)
class Assignment:
    block: int
    pickup_index: int
    pickup_square: str
    pickup_xy: tuple[float, float]
    cell: tuple[int, int] | None  # design cell, None for a plain square-to-square transfer
    cell_square: str
    cell_xy: tuple[float, float]
    pickup_turn_deg: float = 0.0  # wrist turn to grab a twisted block face-on (counterclockwise, degrees)
    place_base_mm: float = 0.0  # what the block lands on, above the mat (0: the mat itself)
    grip_lift_mm: float = 0.0  # grip this much higher than usual (tall blocks vs the palm)
    release_position: float | None = None  # gripper position to let go (None: the workspace default)
    travel_z: float | None = None  # travel height while carrying (None: the workspace default)
    shape: str = "cube"
    record: object | None = None  # scene.SceneBlock to record once the block is confirmed let go


@dataclass(frozen=True)
class Plan:
    design: Design | None
    assignments: tuple[Assignment, ...]
    steps: tuple[Step, ...]
    structure: tuple | None = None  # placed blocks, for build files


class PlanError(ValueError):
    pass


@dataclass(frozen=True)
class Loose:
    """A block found by the camera: a label and its mat coordinates (square (F, R) at whole numbers)."""

    label: str
    f: float
    r: float
    pickable: bool = True
    rotation_deg: float = 0.0  # twist vs the mat lines, counterclockwise from files toward ranks, in (-45, 45]


def plan(design: Design, ws: Workspace, loose: list[Loose] | None = None) -> Plan:
    """Build a 3x3 design: fill cells row by row, from the pickup squares in order or, with
    `loose`, from blocks found by the camera (the first pickable ones, in the given order)."""
    problems = ws.problems() if loose is None else ws.mat_problems() + ws.grid_problems()
    if problems:
        raise PlanError("; ".join(problems))
    filled = design.filled()
    sources = ([(sq, ws.square_xy(sq), parse_square(sq), 0.0) for sq in ws.pickup_squares] if loose is None
               else [(c.label, ws.xy_at(c.f, c.r), (c.f, c.r), c.rotation_deg) for c in loose if c.pickable])
    if not filled:
        problems.append("design has no filled cells")
    if len(filled) > len(sources):
        what = "pickup spots are configured" if loose is None else "pickable blocks were found"
        problems.append(f"design needs {len(filled)} blocks but only {len(sources)} {what}")
    if problems:
        raise PlanError("; ".join(problems))

    if loose is not None:
        sources = _pickup_order(ws, sources, loose, len(filled), problems)
        if problems:
            raise PlanError("; ".join(problems))

    assignments = tuple(
        Assignment(i, i, label, xy, cell, ws.cell_square(*cell), ws.cell_xy(*cell), turn)
        for i, ((label, xy, _, turn), cell) in enumerate(zip(sources, filled))
    )
    return Plan(design, assignments, _pick_and_place_steps(ws, assignments))


def plan_transfer(ws: Workspace, moves: list[tuple[str, str]]) -> Plan:
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
    return Plan(None, assignments, _pick_and_place_steps(ws, assignments))


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


def _pickup_order(ws: Workspace, sources: list, loose: list[Loose], n: int, problems: list[str]) -> list:
    """Choose n pickable blocks and an order in which each can be picked up with the fully open
    fingers clear of every block still on the supply area (blocks already picked are gone).

    Greedy: repeatedly take the first block (in the given order) that is clear right now.
    """
    remaining = list(sources)
    on_mat = [(c.f, c.r, c.label) for c in loose]
    order = []

    def blockers(src):
        label, _, at, turn = src
        return [why for f, r, other in on_mat
                if (f, r) != at and (why := ws.finger_conflict_at(at, (f, r), ws.full_open_span_mm, other, turn))]

    while len(order) < n:
        nxt = next((s for s in remaining if not blockers(s)), None)
        if nxt is None:
            for s in remaining:
                problems += [f"picking {s[0]}: {why} - spread the blocks apart" for why in blockers(s)]
            return order
        order.append(nxt)
        remaining.remove(nxt)
        on_mat = [m for m in on_mat if (m[0], m[1]) != nxt[2]]
    return order


TRAVEL_CLEARANCE_MM = 20.0  # a carried block's bottom stays this far above everything on the mat


def plan_structure(ws: Workspace, placed: list, pickups: list[tuple[str, tuple[float, float]]],
                   existing: list | None = None, start: int = 0, check_scene: bool = False,
                   others: list | None = None, turns: list[float] | None = None) -> Plan:
    """Build a solved structure (structure.solve) block by block, block i coming from pickups[i]
    (label, mat coordinates). Blocks before `start` are already built; `existing` (scene blocks,
    including those) are obstacles too. Placed blocks and blocks still waiting are obstacles for
    the fingers at every pickup and drop; travel height rises to clear the tallest of them. With
    `check_scene`, the run checks the scene with the camera right after reaching the safe start."""
    from robotic_experiments.blocks import scene
    from robotic_experiments.blocks.structure import Rect, held_mm  # pure; avoids an import cycle

    existing = list(existing or [])
    others = list(others or [])  # (label, footprint Rect, height): objects on the supply area not used
    turns = list(turns or [0.0] * len(placed))  # wrist turn to grab each block face-on
    problems = ws.mat_problems()
    if len(pickups) != len(placed):
        problems.append(f"{len(placed)} blocks but {len(pickups)} pickups")
    if start >= len(placed):
        problems.append("everything in the build file is already built")
    if problems:
        raise PlanError("; ".join(problems))
    sq = ws.square_size_mm
    built_before = existing  # scene blocks: what is on the mat now

    def footprint_at(p, f, r):
        lu, lv = (p.shape.length, p.shape.width) if p.along == "files" else (p.shape.width, p.shape.length)
        u, v = ws.uv_at(f, r)
        return Rect(u - lu / 2, v - lv / 2, u + lu / 2, v + lv / 2)

    for i, ((label, (f, r)), p) in enumerate(zip(pickups, placed)):
        if i < start:
            continue
        here = footprint_at(p, f, r)
        if any(here.overlap(q.footprint(sq)) for q in placed) or any(here.overlap(b.footprint(sq)) for b in built_before):
            problems.append(f"pickup {label} for block {i} lies inside the structure or under a placed block")
        # a new block must not collide with a placed block that is not part of this build
        new_fp = p.footprint(sq)
        for b in [b for b in built_before if not _is_build_block(b, placed[:start])]:
            if new_fp.overlap(b.footprint(sq)) and b.top_mm > p.base_mm and b.base_mm < p.top_mm:
                problems.append(f"block {i} ({p.shape.name} at {p.at}) would overlap a block already on the mat at {b.at or 'unknown'}")

    assignments = []
    for i, ((label, (pf, pr)), p) in enumerate(zip(pickups, placed)):
        if i < start:
            continue
        held = held_mm(p, ws.fingers_open_along_ranks)
        lift = ws.grip_lift_mm(p.shape.height)
        built = ([(b.at or "a placed block", b.footprint(sq), b.top_mm) for b in built_before]
                 + [(q.at, q.footprint(sq), q.top_mm) for q in placed[start:i]])
        waiting = [(lab, footprint_at(q, f, r), q.shape.height) for (lab, (f, r)), q in zip(pickups[i + 1:], placed[i + 1:])] + others

        def clashes(f, r, gap, fingertip_mm, what, at_label):
            for name, fp, top in built + waiting:
                if top <= fingertip_mm:
                    continue  # entirely below the fingers
                if any(fp.overlap(fr) for fr in ws.finger_rects(f, r, gap, ws.finger_clearance_mm)):
                    problems.append(f"block {i} ({p.shape.name}): fingers at the {what} {at_label} would touch the block on {name}")

        full_gap = ws.full_open_span_mm - 2 * ws.finger_thickness_mm
        clashes(pf, pr, full_gap, 5.0 + lift, "pickup", label)
        clashes(p.f, p.r, ws.release_gap_mm(held), p.base_mm + 5.0 + lift + ws.place_release_offset_mm, "drop on", p.at)

        tallest = max([t for _, _, t in built + waiting], default=0.0)
        place_above = ws.grip_z + p.base_mm + lift + ws.approach_height_mm
        travel = max(ws.travel_z, place_above,
                     ws.mat_top_z + tallest + TRAVEL_CLEARANCE_MM + 5.0 + lift + ws.tcp_above_fingertips_mm)
        assignments.append(Assignment(
            i, i, label, ws.xy_at(pf, pr), None, p.at, ws.xy_at(p.f, p.r),
            place_base_mm=p.base_mm, grip_lift_mm=lift, release_position=ws.release_position(held),
            travel_z=travel, shape=p.shape.name, record=scene.from_placed(p), pickup_turn_deg=turns[i],
        ))
    if problems:
        raise PlanError("; ".join(problems))
    assignments = tuple(assignments)
    steps = list(_pick_and_place_steps(ws, assignments))
    if check_scene:
        steps.insert(1, Step("check_scene", "check the build area against the scene record"))
    return Plan(None, assignments, tuple(steps), tuple(placed))


def _is_build_block(b, built_part) -> bool:
    """Is scene block b one of the build file's blocks that are already built?"""
    return any(b.shape == q.shape.name and b.at == q.at and b.level == q.level for q in built_part)


def already_built(placed: list, existing: list) -> int:
    """How many leading blocks of the build file are already in the scene (same shape, place, level)."""
    k = 0
    for q in placed:
        if any(b.shape == q.shape.name and b.at == q.at and b.level == q.level for b in existing):
            k += 1
        else:
            break
    return k


MIN_TURN_DEG = 2.0  # smaller twists are grabbed at the usual wrist angle


def wrist_theta(theta: float, turn_deg: float) -> float:
    """Wrist angle that turns the fingers `turn_deg` counterclockwise. The gripper x-axis points at
    180 - theta, so turning the fingers counterclockwise lowers theta. Normalised to (-180, 180]."""
    t = (theta - turn_deg) % 360.0
    return t - 360.0 if t > 180.0 else t


def _pick_and_place_steps(ws: Workspace, assignments) -> tuple[Step, ...]:
    def at(xy, z, theta=None):
        return Target(xy[0], xy[1], z, ws.theta_deg if theta is None else theta)

    steps = [Step("move", "move to safe start", target=Target(*ws.safe_start, ws.theta_deg))]
    for a in assignments:
        grip_z = ws.grip_z + a.grip_lift_mm  # pickups are on the mat
        above = grip_z + ws.approach_height_mm
        place_z = ws.grip_z + a.place_base_mm + a.grip_lift_mm
        place_above = place_z + ws.approach_height_mm
        stacked = a.place_base_mm > 0
        release_z = place_z + (ws.stack_release_offset_mm if stacked else ws.place_release_offset_mm)
        travel = a.travel_z if a.travel_z is not None else ws.travel_z
        b = a.block
        src = f"pickup {a.pickup_square}"
        dst = a.cell_square if a.cell is None else f"{a.cell_square} (cell {a.cell[0]},{a.cell[1]})"
        turned = abs(a.pickup_turn_deg) >= MIN_TURN_DEG
        tp = wrist_theta(ws.theta_deg, a.pickup_turn_deg) if turned else ws.theta_deg
        steps += [
            Step("move", f"travel above {src}", b, at(a.pickup_xy, travel)),
            *([Step("move", f"turn wrist {a.pickup_turn_deg:+.0f} deg to match the block", b,
                    at(a.pickup_xy, travel, tp))] if turned else []),
            Step("move", f"approach {src}", b, at(a.pickup_xy, above, tp)),
            Step("open", "open fingers", b),
            Step("move", f"descend to grip at {src}", b, at(a.pickup_xy, grip_z, tp)),
            Step("grab", "close fingers and verify grasp", b),
            Step("move", "lift to approach height", b, at(a.pickup_xy, above, tp)),
            Step("move", "lift to travel height", b, at(a.pickup_xy, travel, tp)),
            *([Step("move", "turn wrist back (block is now square to the mat)", b,
                    at(a.pickup_xy, travel))] if turned else []),
            Step("move", f"travel above {dst}", b, at(a.cell_xy, travel)),
            Step("move", f"approach {dst}", b, at(a.cell_xy, place_above)),
            Step("move", f"lower block onto {dst}", b, at(a.cell_xy, release_z)),
            Step("release", "open fingers to release", b, position=a.release_position),
            Step("move", "retreat upward", b, at(a.cell_xy, place_above)),
            Step("check_released", "close fingers to confirm the block was let go", b, record=a.record),
            Step("move", "retreat to travel height", b, at(a.cell_xy, travel)),
            # A block on top of others can still fall after a clean release: look before going on.
            *([Step("move", "go to the safe start to look", b, Target(*ws.safe_start, ws.theta_deg)),
               Step("verify_placed", f"camera: the block on {a.cell_square} stayed on top", b, expect=a.record)]
              if stacked and a.record is not None else []),
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
