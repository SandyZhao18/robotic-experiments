"""3D structures: shapes, build files, support and stability. Pure Python, no robot.

A build file lists blocks in build order, bottom first:

    [[block]]
    shape = "cube"          # cube | brick | cylinder
    at = "a3"               # a square, or a seam between two neighbouring squares: "a3|b3"
    along = "files"         # bricks only: long axis along the files or the ranks
    level = 2               # optional: assert which level the block ends up on (1 = on the mat)

Each block rests on whatever is already under its footprint: its base is the highest top it
overlaps, and it must be stably supported there (its center over the contact area). Positions
are in mat coordinates (f, r): square (F, R) is centered at whole numbers, a1 = (0, 0); sizes are
in mm, with u along the files and v along the ranks.
"""

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from robotic_experiments.blocks.workspace import parse_square


@dataclass(frozen=True)
class Shape:
    name: str
    length: float  # along its long axis (files unless placed along ranks)
    width: float
    height: float
    round: bool = False  # cylinders: no twist to match, round footprint


SHAPES = {
    "cube": Shape("cube", 30.0, 30.0, 30.0),
    "brick": Shape("brick", 60.0, 30.0, 30.0),
    "cylinder": Shape("cylinder", 30.0, 30.0, 60.0, round=True),
}

HEIGHT_TOLERANCE_MM = 1.0  # supports this close in height count as one flat base
STABILITY_MARGIN_MM = 3.0  # a block's center must sit this far inside its contact area
MIN_CONTACT_MM = 8.0  # each support must overlap the block by this much both ways (placement error is a few mm)


class StructureError(ValueError):
    pass


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle in mat mm: u along files, v along ranks."""

    u0: float
    v0: float
    u1: float
    v1: float

    def overlap(self, other: "Rect") -> "Rect | None":
        u0, v0 = max(self.u0, other.u0), max(self.v0, other.v0)
        u1, v1 = min(self.u1, other.u1), min(self.v1, other.v1)
        return Rect(u0, v0, u1, v1) if u1 - u0 > 1e-6 and v1 - v0 > 1e-6 else None

    def contains(self, u: float, v: float, margin: float = 0.0) -> bool:
        return self.u0 + margin <= u <= self.u1 - margin and self.v0 + margin <= v <= self.v1 - margin


@dataclass(frozen=True)
class Placed:
    """A block in the structure, with its solved position and height."""

    index: int
    shape: Shape
    at: str
    along: str  # "files" | "ranks"
    f: float
    r: float
    base_mm: float  # bottom above the mat
    supports: tuple[int, ...]  # indices of the blocks it rests on (empty: the mat)
    level: int  # 1 = on the mat

    @property
    def top_mm(self) -> float:
        return self.base_mm + self.shape.height

    def footprint(self, square_mm: float) -> Rect:
        u, v = self.f * square_mm, self.r * square_mm
        lu, lv = ((self.shape.length, self.shape.width) if self.along == "files"
                  else (self.shape.width, self.shape.length))
        return Rect(u - lu / 2, v - lv / 2, u + lu / 2, v + lv / 2)


def parse_at(at: str) -> tuple[float, float]:
    """'a3' -> (0, 2); 'a3|b3' -> (0.5, 2) (the seam between two neighbouring squares)."""
    parts = [p.strip() for p in at.split("|")]
    if len(parts) == 1:
        f, r = parse_square(parts[0])
        return float(f), float(r)
    if len(parts) != 2:
        raise StructureError(f"bad position {at!r}: use a square (a3) or a seam (a3|b3)")
    (f1, r1), (f2, r2) = parse_square(parts[0]), parse_square(parts[1])
    if abs(f1 - f2) + abs(r1 - r2) != 1:
        raise StructureError(f"{at!r}: a seam joins two squares that share a side")
    return (f1 + f2) / 2, (r1 + r2) / 2


def solve(blocks: list[dict], square_mm: float) -> list[Placed]:
    """Place blocks in order: each rests on the highest tops under its footprint, stably."""
    placed: list[Placed] = []
    for i, b in enumerate(blocks):
        name = b.get("shape", "cube")
        if name not in SHAPES:
            raise StructureError(f"block {i}: unknown shape {name!r} (use {', '.join(SHAPES)})")
        shape = SHAPES[name]
        along = b.get("along", "files")
        if along not in ("files", "ranks"):
            raise StructureError(f"block {i}: along must be 'files' or 'ranks'")
        f, r = parse_at(b["at"])
        probe = Placed(i, shape, b["at"], along, f, r, 0.0, (), 1)
        fp = probe.footprint(square_mm)

        under = [(p, ov) for p in placed if (ov := fp.overlap(p.footprint(square_mm)))]
        if not under:
            base, supports, contacts = 0.0, (), [fp]
        else:
            base = max(p.top_mm for p, _ in under)
            top_layer = [(p, ov) for p, ov in under if abs(p.top_mm - base) <= HEIGHT_TOLERANCE_MM]
            supports = tuple(p.index for p, _ in top_layer)
            contacts = [ov for _, ov in top_layer]
        thin = [placed[sid].at for sid, c in zip(supports, contacts) if min(c.u1 - c.u0, c.v1 - c.v0) < MIN_CONTACT_MM]
        if thin:
            raise StructureError(
                f"block {i} ({name} at {b['at']}) would barely touch {', '.join(thin)} "
                f"(less than {MIN_CONTACT_MM:.0f} mm of overlap): it would slip off with a few mm of placement error")
        hull = Rect(min(c.u0 for c in contacts), min(c.v0 for c in contacts),
                    max(c.u1 for c in contacts), max(c.v1 for c in contacts))
        cu, cv = f * square_mm, r * square_mm
        if supports and not hull.contains(cu, cv, STABILITY_MARGIN_MM):
            raise StructureError(
                f"block {i} ({name} at {b['at']}) would tip over: its center is not over what it rests on "
                f"({', '.join(placed[s].at for s in supports)})")
        level = 1 + max((placed[s].level for s in supports), default=0)
        if "level" in b and int(b["level"]) != level:
            raise StructureError(
                f"block {i} ({name} at {b['at']}) ends up on level {level}, not {b['level']}: "
                "list blocks bottom first")
        placed.append(Placed(i, shape, b["at"], along, f, r, base, supports, level))
    return placed


def load(path: str | Path, square_mm: float) -> list[Placed]:
    with open(path, "rb") as fh:
        d = tomllib.load(fh)
    blocks = d.get("block", [])
    if not blocks:
        raise StructureError(f"{path}: no [[block]] entries")
    return solve(blocks, square_mm)
