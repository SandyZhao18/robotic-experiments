"""What has been built on the mat: a record of placed blocks, kept in runs/scene.json.

The executor adds each block right after its release is confirmed, so the record stays true
even when a run stops part way. check-scene compares it with what the camera sees; recorded
blocks are obstacles when planning the next run.
"""

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from robotic_experiments.blocks.structure import SHAPES, Placed, Rect

SCENE_PATH = Path("runs/scene.json")


@dataclass(frozen=True)
class SceneBlock:
    shape: str
    at: str  # where it was meant to go: a square or a seam
    along: str
    f: float  # mat coordinates of its center (measured if `measured`, else planned)
    r: float
    base_mm: float
    top_mm: float
    level: int
    measured: bool = False
    color: str | None = None
    placed_at: str = ""

    def footprint(self, square_mm: float) -> Rect:
        s = SHAPES[self.shape]
        lu, lv = (s.length, s.width) if self.along == "files" else (s.width, s.length)
        u, v = self.f * square_mm, self.r * square_mm
        return Rect(u - lu / 2, v - lv / 2, u + lu / 2, v + lv / 2)


def from_placed(p: Placed) -> SceneBlock:
    return SceneBlock(p.shape.name, p.at, p.along, p.f, p.r, p.base_mm, p.top_mm, p.level,
                      placed_at=datetime.now().isoformat(timespec="seconds"))


def load(path: Path = SCENE_PATH) -> list[SceneBlock]:
    if not path.exists():
        return []
    return [SceneBlock(**b) for b in json.loads(path.read_text()).get("blocks", [])]


def save(blocks: list[SceneBlock], path: Path = SCENE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"blocks": [asdict(b) for b in blocks]}, indent=2) + "\n")


def add(block: SceneBlock, path: Path = SCENE_PATH) -> None:
    save(load(path) + [block], path)


OBSTACLE_MARGIN_MM = 5.0


def obstacle_boxes(blocks: list[SceneBlock], ws) -> list[dict]:
    """Each block as a box in `world` for the motion planner: center, size and yaw (x along its
    long axis), grown by OBSTACLE_MARGIN_MM. Positions use the arm's corrected mapping, i.e. where
    the arm put the block, so they line up with the arm's own coordinates."""
    files_yaw = math.degrees(math.atan2(ws.file_unit[1], ws.file_unit[0]))
    out = []
    for b in blocks:
        s = SHAPES[b.shape]
        x, y = ws.xy_at(b.f, b.r)
        out.append({
            "label": f"{b.shape} {b.at or ''} L{b.level}".strip(),
            "center": (x, y, ws.mat_top_z + (b.base_mm + b.top_mm) / 2),
            "size": (s.length + 2 * OBSTACLE_MARGIN_MM, s.width + 2 * OBSTACLE_MARGIN_MM,
                     (b.top_mm - b.base_mm) + 2 * OBSTACLE_MARGIN_MM),
            "yaw_deg": files_yaw if b.along == "files" else files_yaw + 90.0,
        })
    return out


def visible_tops(blocks: list[SceneBlock], square_mm: float) -> list[SceneBlock]:
    """Blocks the camera can see from above: those with nothing resting on them."""
    out = []
    for b in blocks:
        fp = b.footprint(square_mm)
        covered = any(o is not b and o.base_mm >= b.top_mm - 1.0 and fp.overlap(o.footprint(square_mm)) for o in blocks)
        if not covered:
            out.append(b)
    return out
