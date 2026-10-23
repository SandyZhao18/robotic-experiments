"""Measured workspace, loaded from TOML. All positions are in the `world` frame, in mm.

The table carries an 8x8 chess mat. Squares use chess names seen from white's side:
files a..h run left to right, ranks 1..8 run bottom to top. Three recorded square
centers (a1, h1, a8) locate every square. The 3x3 build grid and the pickup spots
are both just squares on the mat.

Every z value refers to the *gripper frame* (the point record-pose reports and
motion.move targets), not the block. grip_z is measured by closing the gripper
on a block sitting on the mat and recording the pose.
"""

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

Vec2 = tuple[float, float]

MAT_SIZE = 8
FILES = "abcdefgh"


def parse_square(name: str) -> tuple[int, int]:
    """'c6' -> (file index 2, rank index 5)."""
    if len(name) != 2 or name[0] not in FILES or name[1] not in "12345678":
        raise ValueError(f"bad square name {name!r}; expected a1..h8")
    return FILES.index(name[0]), int(name[1]) - 1


def square_name(f: int, r: int) -> str:
    return f"{FILES[f]}{r + 1}"


@dataclass(frozen=True)
class Workspace:
    calibrated: bool
    frame: str

    table_z: float  # table top in world: the reference every height is measured from
    mat_thickness_mm: float

    # Recorded centers of three corner squares.
    a1: Vec2
    h1: Vec2
    a8: Vec2
    square_size_mm: float  # measured with a ruler; cross-checks the recorded corners
    # Correction found by placement trials, applied to where the arm is sent (not to where the
    # camera measured the mat). In (files a->h, ranks 1->8) components, at a square whose measured
    # center is (u, v) mm from a1's along files and ranks:
    #     correction = placement_offset_mm + placement_gradient @ (u, v)
    placement_offset_mm: tuple[float, float]
    placement_gradient: tuple[tuple[float, float], tuple[float, float]]

    grid_top_left: str  # square under design cell (row 0, col 0)
    grid_stride: int  # squares between neighbouring design cells
    pickup_squares: tuple[str, ...]  # used in this order
    supply_files: str  # files where loose cubes are found by the camera, e.g. "fgh"
    find_pose: tuple[float, float, float]  # gripper position from which the camera sees the supply area
    max_reach_mm: float  # pickups farther than this from the arm's base are refused

    block_size_mm: float
    open_finger_span_mm: float  # outer width of the fingers when releasing (at open_position, if set)
    finger_clearance_mm: float  # extra gap required beside neighbouring blocks
    # Gripper position (0 closed .. ~850 fully open) used to RELEASE a block; None = fully open.
    # Picking up always opens fully, so a block that is a few mm off its pickup square still fits.
    open_position: float | None
    full_open_span_mm: float  # outer width of the fully open fingers (used around pickups)
    finger_width_mm: float  # width of the finger assembly across the opening direction

    grip_z: float  # gripper-frame z when the fingers are around a block on the mat
    approach_height_mm: float  # above grip_z, for approach and retreat
    place_release_offset_mm: float  # above grip_z when releasing on the mat
    travel_z: float
    theta_deg: float  # rotation about the downward tool axis that aligns the fingers

    safe_start: tuple[float, float, float]
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]

    required_obstacles: tuple[str, ...]
    arrival_tolerance_mm: float

    @property
    def fingers_open_along_ranks(self) -> bool:
        """At theta = -91 (or 89) the fingers open along the ranks; 90 degrees off, along the files."""
        return abs((self.theta_deg + 91) % 180) < 45 or abs((self.theta_deg + 91) % 180) > 135

    def finger_conflict(self, square: str, neighbour: str, span_mm: float) -> str | None:
        """Why open fingers (outer span `span_mm`) at `square` would touch a block on `neighbour`, or None.

        The fingers reach span/2 along their opening direction but only finger_width/2 across it,
        so blocks beside the fingers can be much closer than blocks in front of them.
        """
        return self.finger_conflict_at(parse_square(square), parse_square(neighbour), span_mm, neighbour)

    def finger_conflict_at(self, at: Vec2, other: Vec2, span_mm: float, label: str, turn_deg: float = 0.0) -> str | None:
        """finger_conflict for mat coordinates (f, r) instead of square names. `turn_deg` turns the
        fingers counterclockwise (files toward ranks) from their usual direction, e.g. to match a
        twisted block."""
        (u1, v1), (u2, v2) = self.uv_at(*at), self.uv_at(*other)
        base = 90.0 if self.fingers_open_along_ranks else 0.0  # opening direction, degrees from the files axis
        a = math.radians(base + turn_deg)
        du, dv = u2 - u1, v2 - v1
        d_open, d_across = abs(du * math.cos(a) + dv * math.sin(a)), abs(-du * math.sin(a) + dv * math.cos(a))
        half, margin = self.block_size_mm / 2, self.finger_clearance_mm
        need_open, need_across = span_mm / 2 + half + margin, self.finger_width_mm / 2 + half + margin
        if d_open < need_open and d_across < need_across:
            return (f"{label} is {d_open:.0f} mm away along the finger opening (need {need_open:.0f}) "
                    f"and {d_across:.0f} mm across it (need {need_across:.0f})")
        return None

    @property
    def mat_top_z(self) -> float:
        return self.table_z + self.mat_thickness_mm

    @property
    def file_step(self) -> Vec2:
        return ((self.h1[0] - self.a1[0]) / 7, (self.h1[1] - self.a1[1]) / 7)

    @property
    def rank_step(self) -> Vec2:
        return ((self.a8[0] - self.a1[0]) / 7, (self.a8[1] - self.a1[1]) / 7)

    @property
    def file_unit(self) -> Vec2:
        n = math.hypot(*self.file_step)
        return (self.file_step[0] / n, self.file_step[1] / n)

    @property
    def rank_unit(self) -> Vec2:
        n = math.hypot(*self.rank_step)
        return (self.rank_step[0] / n, self.rank_step[1] / n)

    def measured_xy_at(self, f: float, r: float) -> Vec2:
        """Mat coordinates (square (F, R) centered at whole numbers, a1 = (0, 0)) to `world` x/y."""
        fs, rs = self.file_step, self.rank_step
        return (self.a1[0] + f * fs[0] + r * rs[0], self.a1[1] + f * fs[1] + r * rs[1])

    def measured_square_xy(self, name: str) -> Vec2:
        """Square center as the camera measured it."""
        return self.measured_xy_at(*parse_square(name))

    def uv_at(self, f: float, r: float) -> Vec2:
        """Mat coordinates to mm from a1's center, along (files, ranks)."""
        return (f * math.hypot(*self.file_step), r * math.hypot(*self.rank_step))

    def square_uv(self, name: str) -> Vec2:
        """Measured center of a square in mm from a1's center, along (files, ranks)."""
        return self.uv_at(*parse_square(name))

    def correction_at(self, name: str) -> Vec2:
        """Placement correction at a square, in mm along (files, ranks)."""
        return self.correction_at_fr(*parse_square(name))

    def correction_at_fr(self, f: float, r: float) -> Vec2:
        """Placement correction at mat coordinates (f, r), in mm along (files, ranks)."""
        u, v = self.uv_at(f, r)
        (gff, gfr), (grf, grr) = self.placement_gradient
        of, orank = self.placement_offset_mm
        return (of + gff * u + gfr * v, orank + grf * u + grr * v)

    def square_xy(self, name: str) -> Vec2:
        """Where to send the gripper for this square: measured center plus the placement correction."""
        return self.xy_at(*parse_square(name))

    def xy_at(self, f: float, r: float) -> Vec2:
        """Where to send the gripper for mat coordinates (f, r), correction included."""
        x, y = self.measured_xy_at(f, r)
        cf, cr = self.correction_at_fr(f, r)
        fu, ru = self.file_unit, self.rank_unit
        return (x + cf * fu[0] + cr * ru[0], y + cf * fu[1] + cr * ru[1])

    def cell_square(self, row: int, col: int) -> str:
        """Design rows go down the display, i.e. toward rank 1."""
        f, r = parse_square(self.grid_top_left)
        f, r = f + col * self.grid_stride, r - row * self.grid_stride
        if not (0 <= f < MAT_SIZE and 0 <= r < MAT_SIZE):
            raise ValueError(f"design cell ({row}, {col}) falls off the mat")
        return square_name(f, r)

    def cell_xy(self, row: int, col: int) -> Vec2:
        return self.square_xy(self.cell_square(row, col))

    @property
    def pickups(self) -> tuple[Vec2, ...]:
        return tuple(self.square_xy(s) for s in self.pickup_squares)

    def problems(self) -> list[str]:
        """All static checks: the mat and heights, plus the build grid and pickup layout."""
        return self.mat_problems() + self.layout_problems()

    def grid_problems(self) -> list[str]:
        """layout_problems without the fixed pickup squares (for cubes found by the camera)."""
        return [p for p in self.layout_problems() if not p.startswith("pickup")]

    def mat_problems(self) -> list[str]:
        """Checks needed for any move on the mat: recorded corners and heights."""
        out = []
        fp, rp = math.hypot(*self.file_step), math.hypot(*self.rank_step)
        if fp == 0 or rp == 0:
            return ["a1, h1 and a8 must be distinct"]
        for label, p in (("a1→h1", fp), ("a1→a8", rp)):
            if abs(p - self.square_size_mm) > 2:
                out.append(f"{label} gives {p:.1f} mm squares but square_size_mm is {self.square_size_mm} — recheck the recorded corners")
        cos = (self.file_step[0] * self.rank_step[0] + self.file_step[1] * self.rank_step[1]) / (fp * rp)
        if abs(cos) > math.sin(math.radians(2)):
            out.append(f"files and ranks are {math.degrees(math.acos(cos)):.1f}° apart, expected 90° — are h1 and a8 swapped?")
        if self.grip_z <= self.mat_top_z:
            out.append(f"grip_z {self.grip_z} is at or below the mat surface (z = {self.mat_top_z})")
        if self.travel_z < self.grip_z + self.approach_height_mm:
            out.append("travel_z must be at or above grip_z + approach_height_mm")
        if self.approach_height_mm < self.block_size_mm:
            out.append("approach_height_mm should be at least one block height so a held block clears its neighbours")
        return out

    def layout_problems(self) -> list[str]:
        """Checks on the 3x3 build grid and pickup squares."""
        out = []
        fp, rp = math.hypot(*self.file_step), math.hypot(*self.rank_step)
        try:
            cells = {self.cell_square(r, c) for r in range(3) for c in range(3)}
        except ValueError as e:
            return out + [f"build grid from {self.grid_top_left} with stride {self.grid_stride}: {e}"]

        # Releasing on a cell (partly open) next to another cell; picking up (fully open) next to
        # any cell or other pickup.
        for a in sorted(cells):
            for b in sorted(cells):
                if a < b and (why := self.finger_conflict(a, b, self.open_finger_span_mm)):
                    out.append(f"build grid too tight for the open fingers: dropping on {a}, {why}")
        for i, s in enumerate(self.pickup_squares):
            if s in cells:
                out.append(f"pickup {i} ({s}) is inside the build grid")
                continue
            for other in sorted(cells) + list(self.pickup_squares[:i]):
                if why := self.finger_conflict(s, other, self.full_open_span_mm):
                    out.append(f"pickup {i} ({s}): {why}")
        return out


def load(path: str | Path) -> Workspace:
    with open(path, "rb") as f:
        d = tomllib.load(f)
    m, g, grip, lim = d["mat"], d["grid"], d["gripper"], d["limits"]
    pickups = tuple(d["pickups"]["squares"])
    for s in (g["top_left"], *pickups):
        parse_square(s)
    return Workspace(
        calibrated=bool(d.get("calibrated", False)),
        frame=d.get("frame", "world"),
        table_z=float(d["table"]["z"]),
        mat_thickness_mm=float(m["thickness_mm"]),
        a1=tuple(m["a1"]),
        h1=tuple(m["h1"]),
        a8=tuple(m["a8"]),
        square_size_mm=float(m["square_size_mm"]),
        placement_offset_mm=tuple(m.get("placement_offset_mm", (0.0, 0.0))),
        placement_gradient=tuple(tuple(row) for row in m.get("placement_gradient", ((0.0, 0.0), (0.0, 0.0)))),
        grid_top_left=g["top_left"],
        grid_stride=int(g["stride"]),
        pickup_squares=pickups,
        supply_files=d.get("find", {}).get("supply_files", "fgh"),
        find_pose=tuple(d.get("find", {}).get("pose", (314.0, -392.0, 600.0))),
        max_reach_mm=float(d.get("find", {}).get("max_reach_mm", 700.0)),
        block_size_mm=float(d["block"]["size_mm"]),
        open_finger_span_mm=float(grip["open_finger_span_mm"]),
        finger_clearance_mm=float(grip["finger_clearance_mm"]),
        open_position=float(grip["open_position"]) if "open_position" in grip else None,
        full_open_span_mm=float(grip.get("full_open_span_mm", grip["open_finger_span_mm"])),
        finger_width_mm=float(grip.get("finger_width_mm", 45.0)),
        grip_z=float(grip["grip_z"]),
        approach_height_mm=float(grip["approach_height_mm"]),
        place_release_offset_mm=float(grip["place_release_offset_mm"]),
        travel_z=float(grip["travel_z"]),
        theta_deg=float(grip["theta_deg"]),
        safe_start=tuple(d["safe_start"]["xyz"]),
        bounds_min=tuple(lim["min_xyz"]),
        bounds_max=tuple(lim["max_xyz"]),
        required_obstacles=tuple(lim["required_obstacles"]),
        arrival_tolerance_mm=float(lim["arrival_tolerance_mm"]),
    )
