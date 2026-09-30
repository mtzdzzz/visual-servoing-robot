"""Stage 21 Demo 3: deterministic Stage 20 obstacle-aware execution."""

from demo_config import DEMO_OBSTACLE_TITLE, OBSTACLE_DEMO_RANDOM_SEED
from obstacle_motion_runtime import run_stage20_demo


def run_demo_obstacle(*, gui: bool = True, realtime: bool = True, keep_open: bool = True):
    return run_stage20_demo(
        gui=gui, realtime=realtime, keep_open=keep_open,
        seed=OBSTACLE_DEMO_RANDOM_SEED, mode_title=DEMO_OBSTACLE_TITLE,
        window_title=DEMO_OBSTACLE_TITLE,
    )
