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
MIN_TOP_FRACTION = 0.6  # a detection whose top face is narrower than this x block size is a fragment


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


def edge_rotation(gray: np.ndarray, top_mask: np.ndarray, to_mat: np.ndarray) -> float:
    """Rotation in (-45, 45] of a cube's top face vs the mat lines, from its edges in the image.

    Brightness gradients along the top face's outline point across its four edges; folded modulo
    90 degrees (angles x 4) they agree, so a magnitude-weighted circular mean is robust even when
    the outline itself is ragged. `to_mat` (2x2) maps image directions to mat directions near the
    cube, so the angle is measured against the files and ranks, not the image axes.
    """
    m = top_mask.astype(np.uint8)
    band = cv2.dilate(m, np.ones((5, 5), np.uint8)) & (1 - cv2.erode(m, np.ones((5, 5), np.uint8)))
    g = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    sel = band.astype(bool)
    vecs = to_mat @ np.vstack([gx[sel], gy[sel]])  # gradient directions in mat coordinates
    weight = np.hypot(*vecs)
    theta = np.arctan2(vecs[1], vecs[0])
    phi = math.degrees(math.atan2(np.sum(weight * np.sin(4 * theta)), np.sum(weight * np.cos(4 * theta)))) / 4
    return 45.0 if phi == -45.0 else phi


def locate(frame: Frame, ws: workspace.Workspace) -> list[Found]:
    """Mat coordinates and rotation of every block in the frame."""
    a1_view, fs, rs = fit_view(frame.world_lattice, ws)
    to_mat = np.linalg.inv(np.column_stack([fs, rs]))
    # Pixel -> mat coordinates on the mat plane, for the top-face outline (a plane parallel to
    # the mat maps by a similarity, so angles survive even though the face sits 30 mm higher).
    H, _ = cv2.findHomography(frame.corners_px.reshape(-1, 2).astype(np.float32),
                              mat_coords(frame.world_lattice, ws).astype(np.float32))
    sq = ws.square_size_mm
    gray = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2GRAY)
    out = []
    for b in frame.blocks:
        f, r = to_mat @ (b.world[:2] - a1_view)
        contours, _ = cv2.findContours(b.top_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        outline = max(contours, key=cv2.contourArea).reshape(-1, 1, 2).astype(np.float32)
        on_mat = cv2.perspectiveTransform(outline, H).reshape(-1, 2) * sq
        _, size = square_rotation(on_mat)  # outline: only for the size sanity check
        rot = edge_rotation(gray, b.top_mask, local_jacobian(H, b.top_px))
        out.append(Found(b.color, float(f), float(r), rot, size, b.top_px))
    return sorted(merge_fragments(out, ws), key=lambda c: (round(c.r), round(c.f)))


def local_jacobian(H: np.ndarray, px: tuple[float, float]) -> np.ndarray:
    """2x2 derivative of the pixel -> mat homography at a pixel (maps image directions to mat ones)."""
    def apply(u, v):
        x, y, w = H @ np.array([u, v, 1.0])
        return np.array([x / w, y / w])
    u, v = px
    return np.column_stack([apply(u + 1, v) - apply(u - 1, v), apply(u, v + 1) - apply(u, v - 1)]) / 2


def merge_fragments(found: list[Found], ws: workspace.Workspace) -> list[Found]:
    """Drop detections that are not whole cubes: a top face much smaller than a cube (e.g. a
    shaded side face picked up in another color), or one sitting on a bigger detection."""
    min_side = MIN_TOP_FRACTION * ws.block_size_mm
    whole = sorted((c for c in found if min(c.top_size_mm) >= min_side), key=lambda c: -c.top_size_mm[0] * c.top_size_mm[1])
    kept: list[Found] = []
    for c in whole:
        if all(math.dist((c.f, c.r), (k.f, k.r)) * ws.square_size_mm >= ws.block_size_mm for k in kept):
            kept.append(c)
    return kept


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
