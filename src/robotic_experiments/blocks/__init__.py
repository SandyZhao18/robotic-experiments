"""Sketch-to-Structure: build a 3x3 block pattern with the arm.

design    - the 3x3 pattern (pure data, no robot)
workspace - measured positions, loaded from a TOML file (pure data, no robot)
planner   - design + workspace -> deterministic step list (pure, no robot)
dry_run   - CLI: preview a plan offline
executor  - the only module that moves hardware; guarded, stops on any failure
record_pose - CLI: read-only, prints the gripper pose for calibration
"""
