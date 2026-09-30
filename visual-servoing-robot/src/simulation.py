"""Command-line entry point for the visual-servoing robot project.

All robot, perception, control, experiment and demo implementations live in
dedicated modules.  This file owns only argument parsing and mode dispatch.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PyBullet Panda visual-servoing experiments and final demos."
    )
    parser.add_argument("--inspect", action="store_true", help="Print Panda structure in DIRECT mode.")
    parser.add_argument("--stage7", action="store_true", help="Run five-trial static robustness validation.")
    parser.add_argument("--stage7-trial", type=int, choices=range(1, 6), metavar="N",
                        help="Run one numbered Stage 7 trial.")
    parser.add_argument("--stage8-manual", action="store_true",
                        help="Run mouse-drag RED-target visual tracking.")
    parser.add_argument("--stage9", action="store_true", help="Run dynamic tracking evaluation.")
    parser.add_argument("--stage10", action="store_true", help="Run robustness evaluation.")
    parser.add_argument("--stage12-estimation", action="store_true",
                        help="Run image-plane motion estimation evaluation.")
    parser.add_argument("--stage13", action="store_true",
                        help="Run predictive visual-servo A/B evaluation.")
    parser.add_argument("--stage14-rgbd", action="store_true",
                        help="Run RGB-D target localization evaluation.")
    parser.add_argument("--stage15-multitarget", action="store_true",
                        help="Run RED/GREEN/BLUE perception evaluation.")
    parser.add_argument("--stage16-selection", action="store_true",
                        help="Run indefinite manual multi-target selection.")
    parser.add_argument("--stage16-eval", action="store_true",
                        help="Run scheduled Stage 16 switching evaluation.")
    parser.add_argument("--stage17-obstacle", action="store_true",
                        help="Run RGB-D obstacle perception evaluation.")
    parser.add_argument("--stage18-collision", action="store_true",
                        help="Run collision-checking evaluation.")
    parser.add_argument("--stage19-rrt-connect", action="store_true",
                        help="Run RRT-Connect planning evaluation without execution.")
    parser.add_argument("--stage20-execute", action="store_true",
                        help="Align selected targets with collision-checked Stage 20 execution.")
    parser.add_argument("--demo-tracking", action="store_true",
                        help="Predictive visual tracking demo with mouse-draggable RED target.")
    parser.add_argument("--demo-multitarget", action="store_true",
                        help="Manual RGB-D multi-target selection demo.")
    parser.add_argument("--demo-obstacle", action="store_true",
                        help="Deterministic obstacle-aware RRT execution demo.")
    return parser


def dispatch_mode(arguments: argparse.Namespace) -> None:
    """Lazy imports keep every historical experiment independently runnable."""
    action: Callable[[], object]
    if arguments.inspect:
        from robotics_core import inspect_panda_structure
        action = inspect_panda_structure
    elif arguments.demo_tracking:
        from demo_tracking import run_demo_tracking
        action = run_demo_tracking
    elif arguments.demo_multitarget:
        from demo_multitarget import run_demo_multitarget
        action = run_demo_multitarget
    elif arguments.demo_obstacle:
        from demo_obstacle import run_demo_obstacle
        action = run_demo_obstacle
    elif arguments.stage16_selection:
        from stage16_target_switching import run_stage16_target_selection
        action = run_stage16_target_selection
    elif arguments.stage16_eval:
        from stage16_target_switching import run_stage16_evaluation
        action = run_stage16_evaluation
    elif arguments.stage17_obstacle:
        from stage17_obstacle_evaluation import run_stage17_obstacle_perception
        action = run_stage17_obstacle_perception
    elif arguments.stage18_collision:
        from stage18_collision_evaluation import run_stage18_collision_checking
        action = run_stage18_collision_checking
    elif arguments.stage20_execute:
        from stage20_target_alignment import run_stage20_selection
        action = run_stage20_selection
    elif arguments.stage19_rrt_connect:
        from stage19_rrt_connect_evaluation import run_stage19_rrt_connect_planning
        action = run_stage19_rrt_connect_planning
    elif arguments.stage15_multitarget:
        from stage15_multitarget_evaluation import run_stage15_multitarget_perception
        action = run_stage15_multitarget_perception
    elif arguments.stage14_rgbd:
        from stage14_rgbd_evaluation import run_stage14_rgbd_localization
        action = run_stage14_rgbd_localization
    elif arguments.stage13:
        from stage13_evaluation import run_stage13_predictive_visual_servo_evaluation
        action = run_stage13_predictive_visual_servo_evaluation
    elif arguments.stage12_estimation:
        from stage12_motion_estimation import run_stage12_motion_estimation
        action = run_stage12_motion_estimation
    elif arguments.stage10:
        from stage10_evaluation import run_stage10_robustness_evaluation
        action = run_stage10_robustness_evaluation
    elif arguments.stage9:
        from robotics_core import run_stage9_dynamic_tracking_evaluation
        action = run_stage9_dynamic_tracking_evaluation
    elif arguments.stage8_manual:
        from robotics_core import run_stage8_manual_target_drag_gui
        action = run_stage8_manual_target_drag_gui
    elif arguments.stage7_trial is not None:
        from robotics_core import run_stage7_static_robustness_validation
        action = lambda: run_stage7_static_robustness_validation([arguments.stage7_trial])
    elif arguments.stage7:
        from robotics_core import run_stage7_static_robustness_validation
        action = run_stage7_static_robustness_validation
    else:
        from robotics_core import run_rgb_detection_hold_gui
        action = run_rgb_detection_hold_gui
    action()


def main(argv: Sequence[str] | None = None) -> None:
    dispatch_mode(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
