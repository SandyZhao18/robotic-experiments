"""Locate the chess mat with the wrist camera. Read-only: never moves the robot.

Put the red block on a1, point the camera down at the mat (it does not need to see the
whole board, but must see the a1 corner), then:

    uv run locate-mat

It finds the chessboard's inner corners in the color image, measures them in 3D with depth
(projected onto a plane fitted to the mat to average out noise), converts them to `world`,
and fits the board. Only x/y are taken from the camera: heights come from the configured
table. The fitted square size must match the ruler-measured one, which checks the depth scale. The red block marks
a1; standard board handedness (files left->right, ranks bottom->top from white's side)
fixes which direction is files. Results go to config/mat_calibration.json together with
an annotated image to check by eye against the printed labels.
"""

import argparse
import asyncio
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.proto.common import Pose, PoseInFrame

from robotic_experiments.blocks import workspace
from robotic_experiments.connection import connect

ARM = "arm"
CAMERA = "cam"
FRAME = "world"
OUT_JSON = Path("config/mat_calibration.json")
OUT_IMAGE = Path("config/mat_calibration.jpg")
# Inner-corner grids to look for, largest first. Narrow ones (7x3) cover close-up views of a strip of the mat.
_SHAPES = [(7, 7), (7, 6), (6, 6), (7, 5), (6, 5), (7, 4), (5, 5), (6, 4), (7, 3), (5, 4), (6, 3), (4, 4)]
PATTERNS = [p for a, b in _SHAPES for p in ((a, b), (b, a)) if a != b or p == (a, b)]
PATTERNS = list(dict.fromkeys(sorted(PATTERNS, key=lambda p: -p[0] * p[1])))
DEPTH_WINDOW = 3  # half-size of the window used to sample depth around a pixel
TOP_FACE_TOLERANCE_MM = 6.0  # block pixels this close to the nearest block surface count as the top face

# HSV ranges per block color, measured on the real cubes under the lab light. The mat's wood
# squares and skin are reddish too but duller (S <= ~90 lit, dark squares V <= ~60).
# Red (H 0-2) and orange (H 4-7) are too close for per-pixel ranges, so both are found with
# one "warm" range and each blob is then named by its median hue.
WARM = [((0, 100, 90), (12, 255, 255)), ((170, 100, 90), (180, 255, 255))]
RED_MAX_MEDIAN_HUE = 3  # a warm blob with median hue <= 3 (or >= 170) is red, otherwise orange
BLOCK_COLORS = {
    "red": WARM,
    "orange": WARM,
    # Yellow: top H 28 S 131-207. Mat/border H 0-13 or ~96, tape and table H 97-99.
    "yellow": [((20, 80, 110), (38, 255, 255))],
    # Green: H 84-86 S 222-255 V 93-141. The blue tape is H 98-99, so stay below ~92.
    "green": [((70, 120, 60), (92, 255, 255))],
}


def _warm_name(hsv: np.ndarray, mask: np.ndarray) -> str:
    h = np.median(hsv[..., 0][mask])
    return "red" if h <= RED_MAX_MEDIAN_HUE or h >= 170 else "orange"
DEPTH_DISAGREEMENT_WARN_MM = 10.0
SQUARE_SIZE_TOLERANCE_MM = 1.5
DEFAULT_WORKSPACE = "config/workspace.toml"


class LocateError(Exception):
    pass


# ---------- image processing (pure) ----------


def decode_depth(raw: bytes) -> np.ndarray:
    """Viam depth format: b'DEPTHMAP', width, height (big-endian uint64), then big-endian uint16 mm."""
    if raw[:8] != b"DEPTHMAP":
        raise LocateError("unexpected depth image format")
    w, h = int.from_bytes(raw[8:16], "big"), int.from_bytes(raw[16:24], "big")
    return np.frombuffer(raw[24:], dtype=">u2").reshape(h, w).astype(np.float64)


def find_corners(gray: np.ndarray) -> np.ndarray:
    """Largest chessboard inner-corner grid found, shape (rows, cols, 2) in pixels."""
    for cols, rows in PATTERNS:
        ok, c = cv2.findChessboardCornersSB(gray, (cols, rows), flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if ok:
            return c.reshape(rows, cols, 2)
    raise LocateError("no chessboard corners found; is the mat in view and well lit?")


MIN_BLOB_PX = 300


def block_blobs(bgr: np.ndarray, colors) -> list[tuple[str, np.ndarray]]:
    """(color, mask) for every blob of the given block colors, largest first."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    wanted = set(colors)
    families = {"warm" if c in ("red", "orange") else c for c in wanted}
    out = []
    for family in families:
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in WARM if family == "warm" else BLOCK_COLORS[family]:
            mask |= cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < MIN_BLOB_PX:
                continue
            blob = labels == i
            color = _warm_name(hsv, blob) if family == "warm" else family
            if color in wanted:
                out.append((color, blob, stats[i, cv2.CC_STAT_AREA]))
    return [(c, m) for c, m, _ in sorted(out, key=lambda b: -b[2])]


def block_mask(bgr: np.ndarray, color: str = "red") -> np.ndarray:
    """Boolean mask of the largest blob of the block's color."""
    blobs = block_blobs(bgr, [color])
    if not blobs:
        raise LocateError(f"no {color} block found in view")
    return blobs[0][1]


def find_red_block(bgr: np.ndarray) -> tuple[float, float]:
    """Pixel centroid of the largest red blob (top face plus any visible side faces)."""
    v, u = np.nonzero(block_mask(bgr, "red"))
    return float(u.mean()), float(v.mean())


def find_block_top(bgr: np.ndarray, depth: np.ndarray, color: str = "red") -> tuple[tuple[float, float], float]:
    """Pixel centroid and depth of the block's top face.

    Seen at an angle, the color blob includes side faces, which pull its centroid sideways.
    The top face is the block surface nearest the camera, so keep only block pixels within
    TOP_FACE_TOLERANCE_MM of the nearest block depth.
    """
    return top_face(block_mask(bgr, color), depth)


def top_face(mask: np.ndarray, depth: np.ndarray) -> tuple[tuple[float, float], float]:
    """Pixel centroid and depth of the part of a block's blob nearest the camera."""
    top, d = top_face_mask(mask, depth)
    v, u = np.nonzero(top)
    return (float(u.mean()), float(v.mean())), float(np.median(d[top]))


def top_face_mask(mask: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Boolean mask of a block's top face (its pixels nearest the camera), and the masked depth."""
    mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)  # drop mixed edge pixels
    d = np.where(mask & (depth > 0), depth, np.nan)
    if np.isnan(d).all():
        raise LocateError("no depth on the block")
    nearest = np.nanpercentile(d, 5)
    top = d <= nearest + TOP_FACE_TOLERANCE_MM
    if top.sum() < 100:
        raise LocateError("too few top-face pixels on the block")
    return top, d


def sample_depth(depth: np.ndarray, u: float, v: float) -> float:
    r = DEPTH_WINDOW
    patch = depth[int(v) - r : int(v) + r + 1, int(u) - r : int(u) + r + 1]
    patch = patch[patch > 0]
    if patch.size == 0:
        raise LocateError(f"no depth at pixel ({u:.0f}, {v:.0f})")
    return float(np.median(patch))


def rays(pixels: np.ndarray, fx, fy, cx, cy) -> np.ndarray:
    """Unit-depth rays (z = 1) in the camera frame for an (N, 2) array of pixels."""
    return np.column_stack([(pixels[:, 0] - cx) / fx, (pixels[:, 1] - cy) / fy, np.ones(len(pixels))])


def project_onto_plane(pixels: np.ndarray, depths: np.ndarray, intr) -> np.ndarray:
    """Fit a plane to the depth-deprojected points, then intersect each pixel's ray with it.

    Using the fitted plane rather than raw per-pixel depth averages out depth noise.
    """
    d = rays(pixels, *intr)
    pts = d * depths[:, None]
    centroid = pts.mean(axis=0)
    normal = np.linalg.svd(pts - centroid)[2][-1]
    t = (centroid @ normal) / (d @ normal)
    return d * t[:, None]


# ---------- board fitting (pure, in world coordinates) ----------


@dataclass
class MatFit:
    a1: tuple[float, float]
    h1: tuple[float, float]
    a8: tuple[float, float]
    square_size_mm: float
    file_yaw_deg: float  # direction a1 -> h1, measured from world +x toward +y
    block_xy: tuple[float, float]
    block_offset_from_a1_mm: float
    residual_rms_mm: float
    corners_used: int


def orient_lattice(world: np.ndarray, block_xy: np.ndarray) -> np.ndarray:
    """Flip/transpose a (rows, cols, 3) corner lattice so [0, 0] is the a1 inner corner,
    axis 1 runs along files (a->h) and axis 0 along ranks (1->8)."""
    rows, cols = world.shape[:2]
    extremes = {(0, 0): world[0, 0], (0, cols - 1): world[0, -1], (rows - 1, 0): world[-1, 0], (rows - 1, cols - 1): world[-1, -1]}
    (r0, c0) = min(extremes, key=lambda k: np.linalg.norm(extremes[k][:2] - block_xy))
    if r0:
        world = world[::-1]
    if c0:
        world = world[:, ::-1]
    along_cols = world[0, -1, :2] - world[0, 0, :2]
    along_rows = world[-1, 0, :2] - world[0, 0, :2]
    # From white's side files go right and ranks go up: files x ranks must point up (+z).
    if along_cols[0] * along_rows[1] - along_cols[1] * along_rows[0] < 0:
        world = world.transpose(1, 0, 2)
    return world


def fit_board(world: np.ndarray, block_world: np.ndarray) -> MatFit:
    """world: (rows, cols, 3) inner corners in `world`; block_world: block top center. Only x/y are used."""
    block_xy = block_world[:2]
    lat = orient_lattice(world, block_xy)
    ranks, files = lat.shape[:2]
    # inner corner (i, j) sits between files i, i+1 and ranks j, j+1: P = c00 + i*fs + j*rs
    j, i = np.mgrid[0:ranks, 0:files]
    A = np.column_stack([np.ones(i.size), i.ravel(), j.ravel()])
    xy = lat[..., :2].reshape(-1, 2)
    coef, *_ = np.linalg.lstsq(A, xy, rcond=None)
    c00, fs, rs = coef
    residual = math.sqrt(float(np.mean(np.sum((A @ coef - xy) ** 2, axis=1))))

    def center(f, r):  # square center, 0-based file/rank
        return c00 + (f - 0.5) * fs + (r - 0.5) * rs

    a1 = center(0, 0)
    size_f, size_r = np.linalg.norm(fs), np.linalg.norm(rs)
    problems = []
    if abs(size_f - size_r) > 1.5:
        problems.append(f"squares measure {size_f:.1f} x {size_r:.1f} mm, expected square")
    cos = fs @ rs / (size_f * size_r)
    if abs(cos) > math.sin(math.radians(2)):
        problems.append(f"files and ranks are {math.degrees(math.acos(cos)):.1f}° apart, expected 90°")
    if residual > 2.0:
        problems.append(f"corner fit residual {residual:.1f} mm is too large")
    offset = float(np.linalg.norm(block_xy - a1))
    if offset > 0.35 * size_f:
        problems.append(
            f"red block is {offset:.0f} mm from the a1 center the camera computed; the a1 corner may be out of view "
            "or the block is not on a1"
        )
    if problems:
        raise LocateError("; ".join(problems))

    return MatFit(
        a1=tuple(map(float, a1)),
        h1=tuple(map(float, center(7, 0))),
        a8=tuple(map(float, center(0, 7))),
        square_size_mm=float((size_f + size_r) / 2),
        file_yaw_deg=math.degrees(math.atan2(fs[1], fs[0])),
        block_xy=tuple(map(float, block_xy)),
        block_offset_from_a1_mm=offset,
        residual_rms_mm=residual,
        corners_used=int(lat.shape[0] * lat.shape[1]),
    )


def annotate(bgr: np.ndarray, pixel_lattice: np.ndarray, world_lattice: np.ndarray, block_px, block_xy) -> np.ndarray:
    """Draw the detected corners and the square names the fit assigned, for checking by eye."""
    lat_px = orient_lattice_pixels(pixel_lattice, world_lattice, block_xy)
    ranks, files = lat_px.shape[:2]
    j, i = np.mgrid[0:ranks, 0:files]
    H, _ = cv2.findHomography(np.column_stack([i.ravel(), j.ravel()]).astype(np.float32), lat_px.reshape(-1, 2))
    out = bgr.copy()
    for p in lat_px.reshape(-1, 2):
        cv2.circle(out, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
    h, w = out.shape[:2]
    for f in range(8):
        for r in range(8):
            u, v, s = H @ np.array([f - 0.5, r - 0.5, 1.0])
            u, v = u / s, v / s
            if 0 <= u < w and 0 <= v < h:
                label = f"{'abcdefgh'[f]}{r + 1}"
                cv2.putText(out, label, (int(u) - 14, int(v) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(out, label, (int(u) - 14, int(v) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.drawMarker(out, (int(block_px[0]), int(block_px[1])), (255, 0, 255), cv2.MARKER_CROSS, 30, 3)
    return out


def orient_lattice_pixels(pixel_lattice, world_lattice, block_xy):
    """Apply the same flips/transpose to the pixel lattice that orient_lattice applies to world."""
    rows, cols = world_lattice.shape[:2]
    idx = np.stack(np.mgrid[0:rows, 0:cols], axis=-1).astype(float)
    oriented = orient_lattice(np.concatenate([world_lattice, idx], axis=-1), block_xy)  # indices ride along
    return pixel_lattice[oriented[..., 3].astype(int), oriented[..., 4].astype(int)]


# ---------- robot I/O (read-only) ----------


async def to_world(machine, pts_cam: np.ndarray) -> np.ndarray:
    out = []
    for x, y, z in pts_cam:
        p = await machine.transform_pose(PoseInFrame(reference_frame=CAMERA, pose=Pose(x=x, y=y, z=z, o_z=1)), FRAME)
        out.append((p.pose.x, p.pose.y, p.pose.z))
    return np.array(out)


@dataclass
class DepthCheck:
    """Where the camera's frame puts the mat and block heights, compared with the configured table."""

    mat_z: float
    disagreement_mm: float  # depth-measured mat z minus configured mat z
    block_height_mm: float


@dataclass
class Capture:
    """One camera frame, measured: chessboard inner corners and the red block, in `world`."""

    bgr: np.ndarray
    corners_px: np.ndarray  # (rows, cols, 2)
    world_lattice: np.ndarray  # (rows, cols, 3)
    block_px: tuple[float, float]
    block_world: np.ndarray  # (3,) block top center
    block_color: str


@dataclass
class Block:
    color: str
    top_px: tuple[float, float]  # top-face centroid in the image
    top_mask: np.ndarray  # top-face pixels
    world: np.ndarray  # (3,) top-face center in `world`


@dataclass
class Frame:
    """One camera frame: the chessboard corners and every block in view, measured in `world`."""

    bgr: np.ndarray
    corners_px: np.ndarray  # (rows, cols, 2)
    world_lattice: np.ndarray  # (rows, cols, 3)
    blocks: list[Block]


async def capture_frame(machine, colors=None) -> Frame:
    """Measure the mat and every block of the given colors (default: all known colors)."""
    if await Arm.from_robot(machine, ARM).is_moving():
        raise LocateError("arm is moving; wait for it to stop")
    cam = Camera.from_robot(machine, CAMERA)
    k = (await cam.get_properties()).intrinsic_parameters
    intr = (k.focal_x_px, k.focal_y_px, k.center_x_px, k.center_y_px)
    images, _ = await cam.get_images()
    by_name = {im.name: im for im in images}
    if "color" not in by_name or "depth" not in by_name:
        raise LocateError(f"camera returned {sorted(by_name)}, expected color and depth")
    bgr = cv2.imdecode(np.frombuffer(by_name["color"].data, np.uint8), cv2.IMREAD_COLOR)
    depth = decode_depth(by_name["depth"].data)
    if depth.shape != bgr.shape[:2]:
        raise LocateError(f"depth {depth.shape} and color {bgr.shape[:2]} are not aligned")

    corners = find_corners(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    rows, cols = corners.shape[:2]
    px = corners.reshape(-1, 2)
    blobs = block_blobs(bgr, list(colors or BLOCK_COLORS))
    tops = []
    for color, mask in blobs:
        try:
            top, d = top_face_mask(mask, depth)
        except LocateError:
            continue  # e.g. a blob at the image edge with no depth
        v, u = np.nonzero(top)
        tops.append((color, (float(u.mean()), float(v.mean())), top, float(np.median(d[top]))))
    block_cam = np.array([rays(np.array([p]), *intr)[0] * d for _, p, _, d in tops]).reshape(-1, 3)

    depths = np.array([sample_depth(depth, u, v) for u, v in px])
    on_plane = project_onto_plane(px, depths, intr)
    world = await to_world(machine, np.vstack([on_plane, block_cam]))
    lattice, blocks_w = world[: len(px)].reshape(rows, cols, 3), world[len(px):]
    blocks = [Block(c, p, m, w) for (c, p, m, _), w in zip(tops, blocks_w)]
    return Frame(bgr, corners, lattice, blocks)


async def capture(machine, color: str = "red", near: tuple[float, float] | None = None) -> Capture:
    """Measure the mat and one block. color "any" accepts every known block color; with `near`
    (world x, y) the block nearest that point is chosen, otherwise the largest blob."""
    f = await capture_frame(machine, None if color == "any" else [color])
    if not f.blocks:
        raise LocateError(f"no {'' if color == 'any' else color + ' '}block found in view")
    if near is None:
        b = max(f.blocks, key=lambda b: b.top_mask.sum())
    else:
        b = min(f.blocks, key=lambda b: np.linalg.norm(b.world[:2] - np.asarray(near)))
    return Capture(f.bgr, f.corners_px, f.world_lattice, b.top_px, b.world, b.color)


async def capture_and_fit(ws: workspace.Workspace) -> tuple[MatFit, DepthCheck, np.ndarray]:
    async with await connect() as machine:
        c = await capture(machine)

    depth_mat_z, depth_block_z = float(c.world_lattice[..., 2].mean()), float(c.block_world[2])
    fit = fit_board(c.world_lattice, c.block_world)
    if abs(fit.square_size_mm - ws.square_size_mm) > SQUARE_SIZE_TOLERANCE_MM:
        raise LocateError(
            f"camera measures {fit.square_size_mm:.1f} mm squares but the ruler says {ws.square_size_mm} mm; "
            "depth scale or intrinsics are off"
        )
    check = DepthCheck(
        mat_z=float(depth_mat_z),
        disagreement_mm=float(depth_mat_z - ws.mat_top_z),
        block_height_mm=float(depth_block_z - depth_mat_z),
    )
    image = annotate(c.bgr, c.corners_px, c.world_lattice, c.block_px, c.block_world[:2])
    return fit, check, image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, help="workspace TOML (default: %(default)s)")
    args = parser.parse_args()
    ws = workspace.load(args.workspace)
    try:
        fit, check, image = asyncio.run(capture_and_fit(ws))
    except LocateError as e:
        raise SystemExit(f"Could not locate the mat: {e}")

    OUT_JSON.write_text(json.dumps({
        **asdict(fit),
        "mat_top_z": ws.mat_top_z,
        "depth_check": asdict(check),
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=2) + "\n")
    cv2.imwrite(str(OUT_IMAGE), image)
    print(f"Mat located from {fit.corners_used} corners (fit residual {fit.residual_rms_mm:.2f} mm).")
    print(f"Heights from config: mat surface z = {ws.mat_top_z:.1f} (table {ws.table_z:.1f} + mat {ws.mat_thickness_mm:.1f}).")
    print(f"  a1 = [{fit.a1[0]:.1f}, {fit.a1[1]:.1f}]")
    print(f"  h1 = [{fit.h1[0]:.1f}, {fit.h1[1]:.1f}]")
    print(f"  a8 = [{fit.a8[0]:.1f}, {fit.a8[1]:.1f}]")
    print(f"  square size {fit.square_size_mm:.1f} mm (ruler {ws.square_size_mm}), files point {fit.file_yaw_deg:.1f}° from world +x")
    print(f"  block is {fit.block_offset_from_a1_mm:.1f} mm from the computed a1 center")
    print(f"Height check: the camera's frame puts the mat at z = {check.mat_z:.1f} ({check.disagreement_mm:+.1f} mm vs table config), "
          f"block height {check.block_height_mm:.1f} mm")
    if abs(check.disagreement_mm) > DEPTH_DISAGREEMENT_WARN_MM:
        print(f"  NOTE: the camera frame sits about {abs(check.disagreement_mm):.0f} mm "
              f"{'too high' if check.disagreement_mm > 0 else 'too low'} relative to the table. With the camera pointing "
              "down this barely affects x/y, but the camera's mounting offset in the Viam config should be corrected.")
    print(f"Saved {OUT_JSON} and {OUT_IMAGE} — check the square labels in the image match the mat.")


if __name__ == "__main__":
    main()
