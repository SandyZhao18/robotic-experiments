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
filled row by row and take pickup spots in listed order.

**Calibration** (`config/workspace.toml`, all in `world`, mm, values are the gripper frame):
1. Mark the 3×3 grid and the pickup spots on the table; decide which side is the displayed "top".
2. Put the arm in manual mode, tool pointing straight down, fingers aligned with the grid.
3. With the gripper closed around a block at each spot, run `uv run record-pose <name>` for:
   the top-left, top-right and bottom-left grid cells, and every pickup spot.
   The z of those readings is `grip_z`; the theta is `theta_deg`.
4. Measure `size_mm` and the open finger span. Choose `travel_z`, `safe_start` and `limits`.
5. `uv run blocks-dry-run --preset one` must pass with no warnings; then set `calibrated = true`.

**Live runs:** `blocks-run` refuses uncalibrated workspaces and more than `--max-blocks` (default 1)
blocks, checks the machine first, and needs a typed `yes`. Use `--stop-after N` to run only up to a
step (e.g. stop above the pickup to check alignment). Any failure, timeout or Ctrl-C stops the arm and
gripper with no retry, no release and no recovery move.
