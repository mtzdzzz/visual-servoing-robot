"""Compatibility entry for the historical Stage 20 experiment.

The implementation now lives in :mod:`obstacle_motion_runtime` so demos and
later integrations depend on a core runtime rather than an experiment file.
"""
from obstacle_motion_runtime import Demo, UserExit, LOG_ROOT, run_stage20_demo

__all__ = ["Demo", "UserExit", "LOG_ROOT", "run_stage20_demo"]
