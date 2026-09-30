"""Display-only configuration for the three Stage 21 integration demos.

Controller, perception, camera, planner and motor constants deliberately stay
in their frozen owning modules.  This file contains labels and deterministic
demo selection only.
"""

DEMO_TRACKING_TITLE = "FINAL DEMO 1 - PREDICTIVE VISUAL TRACKING"
DEMO_MULTITARGET_TITLE = "FINAL DEMO 2 - MULTI-TARGET SELECTION"
DEMO_OBSTACLE_TITLE = "FINAL DEMO 3 - OBSTACLE-AWARE MOTION"

DEFAULT_SELECTED_TARGET_ID = "target_1"
OBSTACLE_DEMO_RANDOM_SEED = 1
OBSTACLE_DEMO_ACTIVE_START = (-0.4316, 0.3433, -1.1167)
OBSTACLE_DEMO_POSITION = (0.10, -0.20, 0.80)
