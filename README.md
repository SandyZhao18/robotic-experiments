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

- `uv run inspect-machine` — connects and runs a read-only check on every known resource (nothing actuates).
- `uv run record-pose [name]` — read-only; prints (and optionally saves) the gripper's pose in `world` for calibration.
- `uv run blocks-dry-run --preset L` — offline construction plan preview; never connects.
- `uv run blocks-run --preset one` — guarded live execution (see below).
- `uv run pytest` — offline tests.

## Sketch-to-Structure (3×3 block builder)

Code lives in `src/robotic_experiments/blocks/`. The design, workspace, and planner are pure Python;
only `executor.py` talks to hardware.

**Design convention:** row 0 is the top row as displayed, column 0 the left column. Patterns are
given as `--preset one|line|L|cross` or `--grid "#..|#..|###"` (`■`/`□` also work). Blocks are
filled row by row and take pickup squares in listed order.

**Chess mat:** the table carries an 8×8 chess mat. Squares are named from white's side (files a–h
left to right, ranks 1–8 bottom to top), and the design displays exactly as the board looks from
there. The 3×3 build grid (`[grid]`: top-left square + stride) and the pickup spots
(`[pickups]`) are squares on the mat, so three recorded corners locate everything.

**Calibration** (`config/workspace.toml`, all in `world`, mm, values are the gripper frame):
1. Tape the mat down — it must not move after calibration. Note which corner is a1.
2. Measure one square edge (`square_size_mm`) and the block (`size_mm`).
3. For each of a1, h1, a8: put a block centered on the square, guide the gripper over it in
   manual mode (tool straight down, fingers parallel to the mat lines), close the fingers, exit
   manual mode, then `uv run record-pose mat_a1` (`mat_h1`, `mat_a8`).
4. Copy the x/y values into `[mat]`; the z is `grip_z` and the theta is `theta_deg`.
5. `uv run blocks-dry-run --preset one` must pass; then set `calibrated = true`.

**Live runs:** `blocks-run` refuses uncalibrated workspaces and more than `--max-blocks` (default 1)
blocks, checks the machine first, and needs a typed `yes`. Use `--stop-after N` to run only up to a
step (e.g. stop above the pickup to check alignment). Any failure, timeout or Ctrl-C stops the arm and
gripper with no retry, no release and no recovery move.
