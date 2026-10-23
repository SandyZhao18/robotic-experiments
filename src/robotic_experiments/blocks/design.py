"""A 3x3 design: which cells get a block.

Convention: row 0 is the top row as displayed, column 0 is the left column.
How displayed rows/columns map onto the table is decided by the workspace
calibration (top_left / top_right / bottom_left cell centers), not here.
"""

from dataclasses import dataclass

SIZE = 3
FILLED_CHARS = set("#XxOo■1*")
EMPTY_CHARS = set(".-_□0 ")

PRESETS = {
    "one": "...|.#.|...",
    "line": "...|###|...",
    "L": "#..|#..|###",
    "cross": ".#.|###|.#.",
}


@dataclass(frozen=True)
class Design:
    cells: tuple[tuple[bool, ...], ...]

    def __post_init__(self) -> None:
        if len(self.cells) != SIZE or any(len(row) != SIZE for row in self.cells):
            raise ValueError(f"design must be {SIZE}x{SIZE}")

    @classmethod
    def parse(cls, text: str) -> "Design":
        """Parse rows separated by '|' or newlines, e.g. "#..|#..|###" or "■□□\\n■□□\\n■■■"."""
        rows = [r for r in text.replace("|", "\n").splitlines() if r.strip()]
        if len(rows) != SIZE:
            raise ValueError(f"expected {SIZE} rows, got {len(rows)}: {text!r}")
        cells = []
        for r, row in enumerate(rows):
            chars = [ch for ch in row.strip() if ch != " "]
            if len(chars) != SIZE:
                raise ValueError(f"row {r} must have {SIZE} cells, got {len(chars)}: {row!r}")
            parsed = []
            for ch in chars:
                if ch in FILLED_CHARS:
                    parsed.append(True)
                elif ch in EMPTY_CHARS:
                    parsed.append(False)
                else:
                    raise ValueError(f"unknown cell character {ch!r} in row {r}")
            cells.append(tuple(parsed))
        return cls(tuple(cells))

    @classmethod
    def preset(cls, name: str) -> "Design":
        if name not in PRESETS:
            raise ValueError(f"unknown preset {name!r}; choose from {', '.join(PRESETS)}")
        return cls.parse(PRESETS[name])

    def filled(self) -> list[tuple[int, int]]:
        """Filled cells in build order: row by row, left to right."""
        return [(r, c) for r in range(SIZE) for c in range(SIZE) if self.cells[r][c]]

    def render(self) -> str:
        return "\n".join(" ".join("■" if v else "□" for v in row) for row in self.cells)
