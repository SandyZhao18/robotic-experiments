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
 build grid (every other square)     pickups (used in this order)
 a7  c7  e7                          a1  c1  e1  g1  g3
 a5  c5  e5
 a3  c3  e3
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
   the mat is re-located.

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
