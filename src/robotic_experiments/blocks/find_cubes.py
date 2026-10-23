"""Find every cube on the mat with the camera. Read-only: never moves the robot.

    uv run blocks-run --goto 314 -392 600 --approved   # the [find] pose: camera sees the whole board
    uv run find-cubes

Each cube is located against the chessboard squares in the same photo (like check-placement),
so errors in the camera's mounting offset cancel out. Positions are reported in mat
coordinates: (f, r) with square (file F, rank R) centered at whole numbers, a1 = (0, 0). The
cube's rotation relative to the mat lines comes from its top face. An annotated photo is saved
to runs/finds/.
"""

import argparse
import asyncio
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from robotic_experiments.blocks import workspace
from robotic_experiments.blocks.check_placement import fit_view, mat_coords
from robotic_experiments.blocks.locate_mat import Frame, LocateError, capture_frame
from robotic_experiments.connection import connect

DEFAULT_WORKSPACE = "config/workspace.toml"
PHOTO_DIR = Path("runs/finds")
MAX_ROTATION_DEG = 20.0  # parallel fingers square up a cube twisted this much as they close


@dataclass
class Found:
    color: str
    f: float  # mat coordinates: square (F, R) is centered at (F, R); a1 = (0, 0)
    r: float
    rotation_deg: float  # top face vs the mat lines, in (-45, 45]
    top_size_mm: tuple[float, float]  # top face sides, as a sanity check on the detection
    top_px: tuple[float, float]

    def usable(self, ws: workspace.Workspace) -> str | None:
        """None if the arm can pick this cube up, otherwise why not."""
        sq = self.square
        if sq is None:
            return "off the mat"
        if sq[0] not in ws.supply_files:
            return f"not in the supply files ({ws.supply_files})"
        if math.hypot(*ws.xy_at(self.f, self.r)) > ws.max_reach_mm:
            return f"out of reach (> {ws.max_reach_mm:.0f} mm)"
        if abs(self.rotation_deg) > MAX_ROTATION_DEG:
            return f"twisted {self.rotation_deg:+.0f} deg (straighten to within {MAX_ROTATION_DEG:.0f})"
        return None

    @property
    def square(self) -> str | None:
        F, R = round(self.f), round(self.r)
        return workspace.square_name(F, R) if 0 <= F < 8 and 0 <= R < 8 else None

    def offset_mm(self, square_mm: float) -> tuple[float, float]:
        """mm from the nearest square's center, along (files, ranks)."""
        return ((self.f - round(self.f)) * square_mm, (self.r - round(self.r)) * square_mm)


def square_rotation(points_mm: np.ndarray) -> tuple[float, tuple[float, float]]:
    """Rotation in (-45, 45] and side lengths of the rectangle best fitting a square's outline."""
    (_, _), (w, h), angle = cv2.minAreaRect(points_mm.astype(np.float32))
    angle = (angle + 45.0) % 90.0 - 45.0
    return (45.0 if angle == -45.0 else angle), (w, h)


def locate(frame: Frame, ws: workspace.Workspace) -> list[Found]:
    """Mat coordinates and rotation of every block in the frame."""
    a1_view, fs, rs = fit_view(frame.world_lattice, ws)
    to_mat = np.linalg.inv(np.column_stack([fs, rs]))
    # Pixel -> mat coordinates on the mat plane, for the top-face outline (a plane parallel to
    # the mat maps by a similarity, so angles survive even though the face sits 30 mm higher).
    H, _ = cv2.findHomography(frame.corners_px.reshape(-1, 2).astype(np.float32),
                              mat_coords(frame.world_lattice, ws).astype(np.float32))
    sq = ws.square_size_mm
    out = []
    for b in frame.blocks:
        f, r = to_mat @ (b.world[:2] - a1_view)
        contours, _ = cv2.findContours(b.top_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        outline = max(contours, key=cv2.contourArea).reshape(-1, 1, 2).astype(np.float32)
        on_mat = cv2.perspectiveTransform(outline, H).reshape(-1, 2) * sq
        rot, size = square_rotation(on_mat)
        out.append(Found(b.color, float(f), float(r), rot, size, b.top_px))
    return sorted(out, key=lambda c: (round(c.r), round(c.f)))


def annotate(frame: Frame, found: list[Found], ws: workspace.Workspace) -> np.ndarray:
    img = frame.bgr.copy()
    for c in found:
        u, v = int(c.top_px[0]), int(c.top_px[1])
        cv2.drawMarker(img, (u, v), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 20, 2)
        df, dr = c.offset_mm(ws.square_size_mm)
        text = f"{c.color} {c.square or '?'} {df:+.0f},{dr:+.0f} {c.rotation_deg:+.0f}deg"
        cv2.putText(img, text, (u - 60, v - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
        cv2.putText(img, text, (u - 60, v - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


FIND_POSE_TOLERANCE_MM = 20.0


async def find(ws: workspace.Workspace, require_find_pose: bool = False) -> tuple[list[Found], np.ndarray]:
    async with await connect() as machine:
        if require_find_pose:
            from robotic_experiments.blocks.executor import gripper_pose  # avoid a module cycle

            p = await gripper_pose(machine, ws.frame)
            off = math.dist((p.x, p.y, p.z), ws.find_pose)
            if off > FIND_POSE_TOLERANCE_MM:
                x, y, z = ws.find_pose
                raise LocateError(f"the gripper is {off:.0f} mm from the find pose; first run: "
                                  f"uv run blocks-run --goto {x:.0f} {y:.0f} {z:.0f}")
        frame = await capture_frame(machine)
    found = locate(frame, ws)
    return found, annotate(frame, found, ws)


def save_photo(photo: np.ndarray, n: int) -> Path:
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    path = PHOTO_DIR / f"find_{datetime.now():%Y-%m-%d_%H%M%S}_{n}-cubes.jpg"
    cv2.imwrite(str(path), photo)
    return path


def table(found: list[Found], ws: workspace.Workspace) -> str:
    lines = [f"  {'color':7s} {'square':6s} {'files mm':>8s} {'ranks mm':>8s} {'rotation':>8s}  {'top face':>9s}  pickable?"]
    for c in found:
        df, dr = c.offset_mm(ws.square_size_mm)
        w, h = c.top_size_mm
        why = c.usable(ws)
        lines.append(f"  {c.color:7s} {c.square or 'off mat':6s} {df:+8.1f} {dr:+8.1f} {c.rotation_deg:+7.1f}°  "
                     f"{w:3.0f} x {h:3.0f}  {'yes' if why is None else 'no: ' + why}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, help="workspace TOML (default: %(default)s)")
    args = parser.parse_args()
    ws = workspace.load(args.workspace)
    try:
        found, photo = asyncio.run(find(ws))
    except LocateError as e:
        raise SystemExit(f"Could not find cubes: {e}")

    path = save_photo(photo, len(found))
    print(f"{len(found)} cube(s) in view:")
    print(table(found, ws))
    print(f"Photo: {path}")


if __name__ == "__main__":
    main()
