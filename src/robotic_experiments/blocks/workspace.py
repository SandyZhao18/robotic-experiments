"""Measured workspace, loaded from TOML. All positions are in the `world` frame, in mm.

Every z value refers to the *gripper frame* (the point record-pose reports and
motion.move targets), not the block. grip_z is measured by closing the gripper
on a block sitting on the table and recording the pose.
"""

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

Vec2 = tuple[float, float]


@dataclass(frozen=True)
class Workspace:
    calibrated: bool
    frame: str

    # Grid: centers of three corner cells as the design is displayed.
    top_left: Vec2
    top_right: Vec2
    bottom_left: Vec2

    pickups: tuple[Vec2, ...]  # used in this order

    block_size_mm: float
    open_finger_span_mm: float  # outer width of the open fingers
    finger_clearance_mm: float  # extra gap required beside neighbouring blocks

    grip_z: float  # gripper-frame z when the fingers are around a block on the table
    approach_height_mm: float  # above grip_z, for approach and retreat
    place_release_offset_mm: float  # above grip_z when releasing on the table
    travel_z: float
    theta_deg: float  # rotation about the downward tool axis that aligns the fingers

    safe_start: tuple[float, float, float]
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]

    required_obstacles: tuple[str, ...]
    arrival_tolerance_mm: float

    @property
    def col_step(self) -> Vec2:
        return ((self.top_right[0] - self.top_left[0]) / 2, (self.top_right[1] - self.top_left[1]) / 2)

    @property
    def row_step(self) -> Vec2:
        return ((self.bottom_left[0] - self.top_left[0]) / 2, (self.bottom_left[1] - self.top_left[1]) / 2)

    def cell_xy(self, row: int, col: int) -> Vec2:
        rs, cs = self.row_step, self.col_step
        return (
            self.top_left[0] + row * rs[0] + col * cs[0],
            self.top_left[1] + row * rs[1] + col * cs[1],
        )

    def problems(self) -> list[str]:
        """Static checks on the measurements themselves (independent of any design)."""
        out = []
        rp, cp = math.hypot(*self.row_step), math.hypot(*self.col_step)
        if rp == 0 or cp == 0:
            return ["grid corners must be distinct"]
        if abs(rp - cp) > 3:
            out.append(f"row pitch {rp:.1f} mm and column pitch {cp:.1f} mm differ by more than 3 mm")
        cos = (self.row_step[0] * self.col_step[0] + self.row_step[1] * self.col_step[1]) / (rp * cp)
        if abs(cos) > math.sin(math.radians(3)):
            out.append(f"grid rows and columns are {math.degrees(math.acos(cos)):.1f}° apart, expected 90°")
        needed = (self.block_size_mm + self.open_finger_span_mm) / 2 + self.finger_clearance_mm
        if min(rp, cp) < needed:
            out.append(
                f"cell pitch {min(rp, cp):.1f} mm is too tight: open fingers beside a placed block need {needed:.1f} mm"
            )
        cells = [self.cell_xy(r, c) for r in range(3) for c in range(3)]
        for i, p in enumerate(self.pickups):
            for j, q in enumerate(self.pickups[:i]):
                if math.dist(p, q) < needed:
                    out.append(f"pickup {i} is {math.dist(p, q):.1f} mm from pickup {j}, need {needed:.1f} mm")
            for k, q in enumerate(cells):
                if math.dist(p, q) < needed:
                    out.append(f"pickup {i} is {math.dist(p, q):.1f} mm from grid cell {divmod(k, 3)}, need {needed:.1f} mm")
        if self.travel_z < self.grip_z + self.approach_height_mm:
            out.append("travel_z must be at or above grip_z + approach_height_mm")
        if self.approach_height_mm < self.block_size_mm:
            out.append("approach_height_mm should be at least one block height so a held block clears its neighbours")
        return out


def load(path: str | Path) -> Workspace:
    with open(path, "rb") as f:
        d = tomllib.load(f)
    g, grip, lim = d["grid"], d["gripper"], d["limits"]
    return Workspace(
        calibrated=bool(d.get("calibrated", False)),
        frame=d.get("frame", "world"),
        top_left=tuple(g["top_left"]),
        top_right=tuple(g["top_right"]),
        bottom_left=tuple(g["bottom_left"]),
        pickups=tuple(tuple(p) for p in d["pickups"]["spots"]),
        block_size_mm=float(d["block"]["size_mm"]),
        open_finger_span_mm=float(grip["open_finger_span_mm"]),
        finger_clearance_mm=float(grip["finger_clearance_mm"]),
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
