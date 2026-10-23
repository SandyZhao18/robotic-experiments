# robotic-experiments
Experiments with robotic arms at Viam

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
cp .env.example .env   # then fill in the API key, key ID, and machine address
```

Credentials come from the machine's **CONNECT** tab in the Viam app. `.env` is gitignored — never commit keys.

## Scripts

Read-only (never move the robot):
- `uv run inspect-machine` — check every known resource on the machine.
- `uv run record-pose [name]` — print (and optionally save) the gripper's pose in `world`.
- `uv run locate-mat` — find the chess mat with the wrist camera (see Calibration).
- `uv run check-placement SQUARE [--from SQUARE] [--start]` — measure where a block sits on a square.

Offline (no robot connection):
- `uv run blocks-dry-run ...` — preview a construction plan.
- `uv run placement-report [--square a3]` — compare all logged trials in one image.
- `uv run fit-correction --since-trial N [--exclude N ...]` — fit the placement correction from trials.
- `uv run pytest` — tests.

Moves the robot (guarded, see Live runs):
- `uv run blocks-run ...`

## Sketch-to-Structure (3×3 block builder)

Pick 30 mm cubes from fixed pickup squares on a chess mat and build a 3×3 pattern with a UFactory
850 arm and its two-finger gripper. Code is in `src/robotic_experiments/blocks/`: the design,
workspace and planner are pure Python; only `executor.py` moves hardware.

### Layout

The table carries an 8×8 chess mat. Squares are named from white's side (files a–h left to right,
ranks 1–8 bottom to top), and a design displays exactly as the board looks from there.

```
 build area: files a-d               supply area: files e-h
 default 3x3 grid (adjacent):        fixed pickups (in order): e1 e3 e5 g1 g3
 a5  b5  c5                          loose blocks for --find: anywhere reachable in e-h
 a4  b4  c4                          (e1-e7, f1-f4, g1-g4, h1-h2)
 a3  b3  c3
```

All of this is in `config/workspace.toml`: `[grid]` (top-left square + stride), `[pickups]`, block
size, gripper heights and openings, limits. Files f–h beyond rank 3 are out of comfortable reach.

### Designs and plans

```sh
uv run blocks-dry-run --preset L                     # one | line | L | cross
uv run blocks-dry-run --grid "#..|#..|###"           # rows top to bottom; ■/□ also work
uv run blocks-dry-run --transfer a1 a3 c1 c5         # FROM TO pairs, in order
uv run blocks-dry-run --preset L --grid-top-left a5 --grid-stride 1   # adjacent squares
```

Blocks fill the design row by row and take pickups in order. For each block: travel above the
pickup, approach, open fully, descend, grab and verify, lift, travel, lower, release, retreat,
close the fingers to confirm the block was let go, retreat. The planner refuses targets outside
the workspace limits and blocks too close together for the open fingers (fully open around
pickups, `open_position` around drops).

### Calibration

Everything is in the `world` frame, in mm; heights are for the gripper frame.

1. **Table and mat:** `[table] z` is the table top (−38); `[mat] thickness_mm` (3); measure one
   square with a ruler (`square_size_mm`, 57).
2. **Locate the mat:** put the red cube on a1, raise the camera so it sees the whole board
   (`uv run blocks-run --goto 314 -392 600 --approved`), then `uv run locate-mat`. It fits all 49
   inner corners, identifies a1 from the cube, checks the square size against the ruler, and saves
   `config/mat_calibration.json` plus an annotated photo — check that the labels match the mat.
   Copy a1/h1/a8 into `[mat]`.
3. **Grip height:** `uv run blocks-run --hover a1 --approved` stops 150 mm above a1; measure the
   fingertip gap to the mat. `grip_z = hover z − gap + 5` puts the fingertips 5 mm above the mat.
4. **Finger angle:** the gripper x-axis points at 180° − θ; pick θ so it lines up with the files.
5. **Placement correction:** run transfers to squares spread over the grid, `check-placement` each
   (with `--start` before), then `fit-correction` and copy the result into `[mat]`. Refit whenever
   the mat is re-located. After a gripper change, trials over a small area can refit just the
   offset with `--offset-only`, keeping the gradient from the spread-out fit.

`check-placement` measures a block against the chessboard squares **in the same photo**, so errors
in the camera's mounting offset cancel out. It finds the block of any known color (red, orange,
yellow, green) nearest the square, by its top face. Results go to `config/placement_log.jsonl`,
photos to `runs/placements/trial-NN_<color>_<from>-to-<to>_<mm>_<time>.jpg`.

### Live runs

`blocks-run` refuses uncalibrated workspaces and more than `--max-blocks` (default 1) blocks,
checks the machine first (components healthy, obstacles configured, arm still, gripper empty) and
needs a typed `yes` (`--approved` skips it when a supervisor approved the run). Every call has a
timeout; every move is checked against its target; vertical moves are kept to a straight line.
Any failure, timeout or Ctrl-C stops the arm and gripper with **no retry, no release and no
recovery move**. `--stop-after N` runs only up to a step. Recovery moves while a block is held need
`--allow-held-block` (only with `--goto` / `--open-gripper`).

A typical trial:

```sh
uv run check-placement a1 --start                      # where the cube starts
uv run blocks-run --transfer a1 a3 --approved          # someone at the e-stop
uv run check-placement a3 --from a1                    # where it landed
uv run placement-report
```

### Results so far (2026-10-23)

| Stage | Result |
|---|---|
| One block, a1 → a3, repeated (trials 2–6) | 1.8–6.4 mm off center; the arm adds < 1 mm, the rest is hand placement |
| Mat moved closer, re-located from a whole-board view | 49 corners, 0.6 mm fit residual |
| Placement correction fitted across the grid (trials 7–12) | ~0.8° rotation + offset; residuals < 1 mm |
| Verification on unseen squares (13–14) | 1.3 and 3.2 mm |
| Two blocks, then a 3-block line (15–17) | all placed; two landed ~12 mm off (finger pad, below) |
| Adjacent squares (44 mm release opening) | cube stuck to the finger; caught by the release check |
| Finger rubber taped, adjacent pair a3–a4 (18–19) | released cleanly; neighbour moved 0.2 mm |
| **Adjacent L**, one cube per run (20–24) | all five placed, none nudged; 5–9 mm off, consistently +6.4 mm along ranks |
| Offset refit for the taped finger (`--offset-only`) | predicted residuals within ±1.6 mm along ranks |
| **Adjacent L in a single run**, pickups a1–e1 (25–29) | 72 steps, no stops; ranks −1.3 to +2.8 mm, worst 6.6 mm (files offset carried from hand placement) |
| Camera-found twisted cube (31): red found −24° on f1, wrist turned to match | placed on c5 2.8 mm off, square within 3° |
| **Adjacent L from camera-found cubes** (32–36), one run | 1.3–5.4 mm off center, mean ranks 0.0 mm |
| 2-cube stack (tower2): first try, released 2 mm above (measured 5 mm gap) | second cube fell off |
| tower2 again, stacked blocks released almost touching + camera check | top 63 mm, 3.6 mm from target |
| **tower3** resumed from the scene record (third cube only) | top 94 mm, 0.8 mm from target |
| **Bridge** from camera-found blocks: cubes a3, b3; blue brick across a3\|b3; cube on top | brick top 63 mm / 2.7 mm, top cube 94 mm / 0.7 mm |
| **Cylinder arch** from camera-found blocks: cylinders a3, b3 (gripped 20 mm higher); brick across; cube on top | brick top 92 mm / 1.9 mm, top cube 122 mm / 3.9 mm |
| **5-level arch tower** from camera-found blocks (cylinders, brick, cube, brick balanced on the cube, cube), one run | tops 92 / 127 / 160 / 188 mm; 3.9 → 8.5 mm off target, growing with height |

### Building from cubes the camera finds

Scatter blocks in the **supply files e–h** (reachable: e1–e7, f1–f4, g1–g4, h1–h2), move the camera up and
build:

```sh
uv run blocks-run --goto 314 -392 600 --approved       # the [find] pose: camera sees the whole board
uv run find-cubes                                      # read-only: what the camera sees
uv run blocks-dry-run --preset L --grid-top-left a5 --grid-stride 1 --find
uv run blocks-run --preset L --grid-top-left a5 --grid-stride 1 --find --max-blocks 5 --approved
```

Cubes are located against the chessboard in the same photo (immune to the camera's mounting
error), filtered to cube height and size (not mat squares or hands), and their twist is measured
from the top face's edges (repeats within ~1.5°). Twisted cubes are grabbed with the wrist turned
to match (turned at travel height) and placed square. The plan refuses cubes out of reach and
cubes whose open fingers (at their turned angle) would touch a cube still waiting, and picks in an
order that avoids that where possible. The run re-finds the cubes before moving: keep hands off.

### Structures: stacking, seams, bricks

Build files (`designs/*.toml`) list blocks bottom first, on squares or seams (`a3|b3`), with
shapes `cube`, `brick` (`along = "files"|"ranks"`) and `cylinder`. The planner rests each block on
what is under it and refuses blocks that would tip or barely touch their support.

```sh
uv run blocks-dry-run --build designs/bridge.toml --find     # blocks found by the camera
uv run blocks-run --build designs/tower3.toml --approved     # or `from = "f1"` per block
uv run check-scene [--update | --sync]                       # scene record vs camera
```

`runs/scene.json` records every placed block. A build run checks the build area against it right
after the safe start (and stops on any difference), resumes after blocks already built, treats
recorded blocks as obstacles (finger clearance, travel height, motion planner for sideways moves),
releases stacked blocks almost touching (`stack_release_offset_mm`), and has the camera confirm
each stacked block stayed on top before going on.

### Known issues

- **Gripper finger missing its pad.** Grasps can be asymmetric (~10 mm along the closing axis), and
  the cube can stick to the finger on release. The release check stops a run when the fingers
  close on a stuck block, but in this gripper's mode a stuck cube is invisible to the "holding"
  status when the fingers are fully open — check the fingers by eye before each run.
- **Camera mounting offset** in the Viam config is ~30 mm off in height and slightly rotated, so
  positions from `locate-mat` depend on the pose it ran from; the fitted placement correction
  absorbs this for now. Fixing the camera frame is the next calibration step.
- **The table obstacle** (top at −38) disagrees with the camera by ~30 mm; heights here follow the
  table.
