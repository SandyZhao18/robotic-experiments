"""Compare all logged placement trials in one image. Offline: reads the log and photos only.

    uv run placement-report            # all trials
    uv run placement-report --square a3

Left: every trial's offset plotted inside the target square (files to the right, ranks up, as
seen from white's side), numbered by trial. Right: a crop of each trial's photo, same numbering.
Writes runs/placements/report_<square>_trials-<first>-<last>.jpg and prints a table with the spread.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from robotic_experiments.blocks.check_placement import LOG_PATH, PHOTO_DIR, is_trial

PLOT = 520  # px, scatter plot size
TILE = 260  # px, photo crop size
COLS = 3


def load(square: str | None) -> list[dict]:
    if not LOG_PATH.exists():
        raise SystemExit(f"No trials logged yet ({LOG_PATH}).")
    recs = [json.loads(line) for line in LOG_PATH.read_text().splitlines() if line.strip()]
    starts = {r["trial"]: r for r in recs if r.get("start")}
    recs = [r for r in recs if is_trial(r) and (square is None or r["square"] == square)]
    for r in recs:
        if r.get("trial") in starts:
            r["start_at"] = starts[r["trial"]]
    if not recs:
        raise SystemExit("No matching trials.")
    return recs


def color(i: int) -> tuple[int, int, int]:
    hue = int(180 * i / 7) % 180
    b, g, r = cv2.cvtColor(np.uint8([[[hue, 200, 220]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def label(img, text, org, scale=0.6, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick + 3)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (30, 30, 30), thick)


def scatter(recs: list[dict], square_mm: float = 57.0, block_mm: float = 30.0) -> np.ndarray:
    img = np.full((PLOT, PLOT, 3), 255, np.uint8)
    half = square_mm / 2 + 6
    s = (PLOT - 80) / (2 * half)  # px per mm
    c = PLOT // 2

    def px(f_mm, r_mm):
        return int(c + f_mm * s), int(c - r_mm * s)  # ranks up

    cv2.rectangle(img, px(-square_mm / 2, square_mm / 2), px(square_mm / 2, -square_mm / 2), (60, 160, 60), 2)
    cv2.rectangle(img, px(-block_mm / 2, block_mm / 2), px(block_mm / 2, -block_mm / 2), (200, 200, 200), 1)
    for ring in (2, 5, 10):
        cv2.circle(img, (c, c), int(ring * s), (225, 225, 225), 1)
        label(img, f"{ring}mm", (c + int(ring * s * 0.71) + 2, c - int(ring * s * 0.71)), 0.4)
    cv2.drawMarker(img, (c, c), (60, 160, 60), cv2.MARKER_CROSS, 16, 2)
    for i, r in enumerate(recs, 1):
        f, rk = r["along_files_mm"], r["along_ranks_mm"]
        clipped = abs(f) > half or abs(rk) > half
        f, rk = max(-half, min(half, f)), max(-half, min(half, rk))
        cv2.circle(img, px(f, rk), 7, color(i), -1)
        label(img, f"{i}{'*' if clipped else ''}", (px(f, rk)[0] + 9, px(f, rk)[1] - 6), 0.55, 2)
    label(img, "files ->", (PLOT - 95, PLOT - 12), 0.5)
    label(img, "ranks ^", (8, 20), 0.5)
    label(img, "green = square, grey = block size, * = off chart", (8, PLOT - 12), 0.42)
    return img


def tiles(recs: list[dict]) -> np.ndarray:
    rows = math.ceil(len(recs) / COLS)
    sheet = np.full((rows * TILE, COLS * TILE, 3), 245, np.uint8)
    for i, r in enumerate(recs):
        y, x = divmod(i, COLS)
        tile = np.full((TILE, TILE, 3), 220, np.uint8)
        photo = cv2.imread(r["photo"]) if r.get("photo") and Path(r["photo"]).exists() else None
        if photo is not None and r.get("crop_px"):
            x0, y0, x1, y1 = r["crop_px"]
            tile = cv2.resize(photo[y0:y1, x0:x1], (TILE, TILE))
        else:
            label(tile, "no photo", (TILE // 2 - 45, TILE // 2), 0.6)
        cv2.rectangle(tile, (0, 0), (TILE - 1, TILE - 1), color(i + 1), 4)
        move = f"{r['from']}->{r['square']}" if r.get("from") else r["square"]
        label(tile, f"{i + 1}: {move} {r['distance_mm']:.1f} mm", (8, 24), 0.6, 2)
        sheet[y * TILE:(y + 1) * TILE, x * TILE:(x + 1) * TILE] = tile
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--square", help="only trials placed on this square")
    args = parser.parse_args()
    recs = load(args.square)

    left, right = scatter(recs), tiles(recs)
    h = max(left.shape[0], right.shape[0])
    pad = lambda im: cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    squares = "-".join(sorted({r["square"] for r in recs}))
    out = PHOTO_DIR / f"report_{squares}_trials-{recs[0].get('trial', 1)}-{recs[-1].get('trial', len(recs))}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.hstack([pad(left), pad(right)]))

    print(f"{'#':>2}  {'square':6} {'files':>7} {'ranks':>7} {'dist':>6}  {'start (files, ranks)':>21}  correction   when")
    for i, r in enumerate(recs, 1):
        i = r.get("trial", i)
        corr = r.get("placement_offset_in_use", [0, 0])
        st = r.get("start_at")
        start = f"{st['square']} {st['along_files_mm']:+5.1f}, {st['along_ranks_mm']:+5.1f}" if st else "-"
        print(f"{i:>2}  {r['square']:6} {r['along_files_mm']:+7.1f} {r['along_ranks_mm']:+7.1f} {r['distance_mm']:6.1f}"
              f"  {start:>21}  [{corr[0]:.1f}, {corr[1]:.1f}]  {r['recorded_at']}")
    current = [r for r in recs if r.get("placement_offset_in_use") == recs[-1].get("placement_offset_in_use")]
    if len(current) >= 2:
        f = np.array([r["along_files_mm"] for r in current])
        rk = np.array([r["along_ranks_mm"] for r in current])
        d = np.array([r["distance_mm"] for r in current])
        print(f"\nWith the current correction ({len(current)} trials): mean offset files {f.mean():+.1f}, ranks {rk.mean():+.1f} mm; "
              f"spread (std) files {f.std(ddof=1):.1f}, ranks {rk.std(ddof=1):.1f} mm; worst {d.max():.1f} mm")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
