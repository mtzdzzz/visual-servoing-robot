"""One fixed Stage 20 demonstration; no trial loop, scene reset, or auto-retry.

Formal safety uses the UNCHANGED Stage 17 visual occupancy and Stage 18
checker. Ground-truth contact is a write-only evaluation observation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pybullet as p
import pybullet_data

import robotics_core as sim
import config
import visual_servo_runtime as runtime
import camera_geometry
import scene_factory
import planning_visualization
import obstacle_perception_runtime
import target_selection_runtime
from demo_config import OBSTACLE_DEMO_ACTIVE_START, OBSTACLE_DEMO_POSITION
from camera_observation import (
    EyeInHandRgbDisplay, render_live_eye_in_hand_rgbd_frame,
    CAMERA_FOV_Y_DEGREES, CAMERA_NEAR_PLANE, CAMERA_FAR_PLANE,
)
from collision_checker import AxisAlignedBox, CollisionChecker, _aabb_overlap
from obstacle_detector import YellowObstacleDetector
from obstacle_localization import ObstacleLocalizer
from rgbd_localization import camera_intrinsics_from_fov
from rrt_connect_planner import RRTConnectConfig, RRTConnectPlanner
from trajectory_executor import TrajectoryExecutor, EXECUTION_RESOLUTION_RAD

DEBUG_STAGE20 = True
DEMO_SEED = 1
DEMO_ACTIVE_GOAL = (-1.4, 0.8, 0.0)
# This static-obstacle Demo does not servo from RGB. Rendering is display-only
# after acquisition; keep expensive Windows OpenGL renders out of the 240 Hz
# motor/safety loop. Existing Camera and Stage13-19 update logic are untouched.
DEMO_DISPLAY_INTERVAL_STEPS = 240
LOG_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "logs"
# This is input-domain handling only: never inflate/shrink obstacle geometry.
NUMERICAL_LIMIT_RESIDUE_RAD = 0.0001


class UserExit(Exception):
    pass


class Demo:
    def __init__(self, gui, realtime, log_name="stage20_demo_execution.jsonl",
                 mode_title="STAGE 20 - FIXED SINGLE DEMO",
                 window_title="Stage 20 - Obstacle-Aware Execution"):
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        self.gui, self.realtime = gui, realtime
        self.mode_title = mode_title
        self.display = EyeInHandRgbDisplay(window_title) if gui else None
        self.display_was_open = bool(self.display and self.display.is_open)
        self.checker = CollisionChecker()
        self.state, self.source = "IDLE", "NONE"
        self.context = None
        self.obstacle = None
        self.occupancy = None
        self.executor = None
        self.t = 0.0
        self.steps = 0
        self.debug_id = None
        self.contacts = 0
        self.max_tracking_error = 0.0
        self.limit_residue_reported = False
        self.last_ee = None
        self.snapshot_states = set()
        self.summary = {"result": "NOT_COMPLETED", "gt_collision_count": 0,
                        "scene_creations": 0, "gt_used_for_planning": False,
                        "gt_used_for_execution_decision": False,
                        "self_collision_checking": "NOT_IMPLEMENTED"}
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        self.log = (LOG_ROOT / log_name).open("w", encoding="utf-8")

    def set_state(self, value):
        self.state = value
        print(f"STAGE 20 | STATE: {value} | SOURCE: {self.source}", flush=True)
        if self.context is not None and self.gui:
            self.render()

    def poll_exit(self):
        if not p.isConnected(self.client):
            raise UserExit("PyBullet window closed")
        if self.gui:
            events = p.getKeyboardEvents(physicsClientId=self.client)
            if any(events.get(key, 0) & p.KEY_WAS_TRIGGERED for key in (ord('q'), ord('Q'), 27)):
                raise UserExit("Q/ESC")
            if self.display_was_open and not self.display.is_open:
                raise UserExit("RGB window closed")

    def actual(self):
        return np.array([p.getJointState(self.context.robot_id, i, physicsClientId=self.client)[0]
                         for i in self.context.arm_joint_indices])

    def edge_configuration(self, actual):
        """Tiny physics residues at URDF limits are logged, not silently ignored.

        Stage 18 rejects even 1e-9 limit overflow. Actual live-link AABBs are
        ALSO checked, without clipping, every physics step below. Only the
        offline segment check requires an in-domain endpoint.
        """
        bounds = np.asarray(self.checker.arm_joint_limits)
        clipped = np.clip(actual, bounds[:, 0], bounds[:, 1])
        residue = float(np.max(np.abs(actual - clipped)))
        if residue > NUMERICAL_LIMIT_RESIDUE_RAD:
            raise ValueError(f"JOINT_LIMIT_VIOLATION: {residue:.8f} rad")
        if residue and not self.limit_residue_reported:
            print(f"URDF boundary numerical residue: {residue:.3g} rad; live geometry checked UNCLIPPED.")
            self.limit_residue_reported = True
        return clipped

    def current_collision(self, actual):
        # All live Panda link geometry, NOT target/obstacle GT geometry.
        live_collision = False
        for link in self.checker.link_indices:
            lo, hi = p.getAABB(self.context.robot_id, link, physicsClientId=self.client)
            live_collision |= _aabb_overlap(AxisAlignedBox(tuple(lo), tuple(hi)), self.occupancy)
        bounds = np.asarray(self.checker.arm_joint_limits)
        if np.all(actual >= bounds[:, 0]) and np.all(actual <= bounds[:, 1]):
            live_collision |= self.checker.check_configuration(actual, self.occupancy).collision
        else:
            self.edge_configuration(actual)  # validate only tiny numerical residue
        return bool(live_collision)

    def edge_safe(self, a, b):
        return not self.checker.check_path(a, b, self.occupancy, EXECUTION_RESOLUTION_RAD).collision

    def tick(self):
        self.poll_exit()
        start = time.perf_counter()
        runtime.step_physics(self.context, self.client)  # frozen motor interface
        self.steps += 1
        self.t += sim.TIME_STEP
        if self.obstacle is not None:
            # Evaluation ONLY. This value never selects a waypoint or triggers a stop.
            touched = bool(p.getContactPoints(self.context.robot_id, self.obstacle, physicsClientId=self.client))
            self.contacts += int(touched)
        if self.gui and self.steps % DEMO_DISPLAY_INTERVAL_STEPS == 0:
            self.render()
        if self.realtime:
            time.sleep(max(0.0, sim.TIME_STEP - (time.perf_counter() - start)))

    def render(self):
        self.poll_exit()
        pos, quat = sim.get_camera_optical_center_pose(self.context.robot_id, self.client)
        rgbd = render_live_eye_in_hand_rgbd_frame(pos, quat, self.client)
        frame = rgbd.live_rgb_frame
        sim.update_camera_reference_axes(self.context.robot_id, self.client, self.context.camera_axis_debug_item_ids)
        text = f"STAGE 20 | {self.state} | {self.source}"
        if self.executor:
            text += f" | Waypoint {self.executor.index + 1}/{len(self.executor.path)}"
        if self.summary.get("direct_path_collision"):
            text += " | DIRECT COLLISION"
        if self.summary.get("final_validation_pass"):
            text += " | PATH VALID"
        self.debug_id = sim.update_motion_debug_text(text, self.debug_id, self.client)
        if self.display:
            rgba = np.asarray(frame.rgba_buffer, dtype=np.uint8).reshape(frame.image_height, frame.image_width, 4).copy()
            lines = [self.mode_title, f"STATE: {self.state}",
                     f"Execution source: {self.source}", "Q / ESC: quit | No auto restart"]
            if self.executor:
                lines += [f"Waypoint: {self.executor.index + 1} / {len(self.executor.path)}",
                          f"Goal joint error: {np.max(np.abs(self.actual()[4:] - self.executor.path[-1, 4:])):.5f} rad"]
            if self.summary.get("final_validation_pass"):
                lines.append("DIRECT: COLLISION | RRT: SUCCESS | PATH: VALID" if self.source == "RRT_CONNECT" else "DIRECT SAFE | PATH VALID")
            for i, line in enumerate(lines):
                cv2.putText(rgba, line, (10, 22 + 21 * i), cv2.FONT_HERSHEY_SIMPLEX, .47, (255, 255, 255, 255), 1)
            self.display.show(frame, rgba)
        if self.executor and self.state in ("TRAJECTORY_EXECUTION", "DIRECT_EXECUTION"):
            ee = p.getLinkState(self.context.robot_id, self.checker.end_effector_link_index,
                                computeForwardKinematics=True, physicsClientId=self.client)[4]
            if self.last_ee:
                p.addUserDebugLine(self.last_ee, ee, (0, .8, 1), 2, physicsClientId=self.client)
            self.last_ee = ee
        if self.gui and self.state in ("READY_TO_EXECUTE", "GOAL_REACHED", "SAFE_STOP") and self.state not in self.snapshot_states:
            # Display-only images of the same physical simulation. Never used
            # by perception, planner, motors, or formal safety decisions.
            self.snapshot_states.add(self.state)
            rgba = np.asarray(frame.rgba_buffer, dtype=np.uint8).reshape(frame.image_height, frame.image_width, 4)
            cv2.imwrite(str(LOG_ROOT.parent / f"stage20_demo_{self.state.lower()}_rgb.png"), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR))
            view = p.getDebugVisualizerCamera(physicsClientId=self.client)
            w, h, pixels, _, _ = p.getCameraImage(view[0], view[1], viewMatrix=view[2],
                projectionMatrix=view[3], renderer=p.ER_TINY_RENDERER, physicsClientId=self.client)
            rgb = np.asarray(pixels, dtype=np.uint8).reshape(h, w, 4)
            cv2.imwrite(str(LOG_ROOT.parent / f"stage20_demo_{self.state.lower()}_scene.png"), cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR))
        return rgbd

    def initialize(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, 0, physicsClientId=self.client)
        p.setTimeStep(sim.TIME_STEP, physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, physicsClientId=self.client)
        joints = sim.get_panda_arm_joint_indices(robot, self.client)
        names = sim.get_panda_arm_joint_names(robot, self.client)
        locks = {names[name]: p.getJointState(robot, names[name], physicsClientId=self.client)[0]
                 for name in sim.BASELINE_LOCKED_JOINT_NAMES}
        self.context = SimpleNamespace(robot_id=robot, arm_joint_indices=joints,
            locked_initial_positions=locks, commanded_joint_targets=[0.0] * 7,
            camera_axis_debug_item_ids=[], max_locked_deviation=0.0)
        if set(locks) != set(joints[:4]):
            raise ValueError("Demo requires the frozen joints1-4 baseline")
        p.resetDebugVisualizerCamera(1.5, 45, -25, (0, 0, .65), physicsClientId=self.client)
        # A visible, motor-driven setup before spawning the obstacle, never teleport.
        initial = self.actual()
        desired = initial.copy()
        desired[4:] = OBSTACLE_DEMO_ACTIVE_START
        count = int(np.ceil(np.max(np.abs(desired - initial)) / (0.12 * sim.TIME_STEP)))
        print("IDLE: motor-driven scene setup; obstacle is not present yet (no resetJointState).")
        for i in range(1, count + 1):
            self.context.commanded_joint_targets = list(initial + (desired - initial) * i / count)
            self.tick()
        for _ in range(240):
            self.tick()
        if np.max(np.abs(self.actual() - desired)) > 0.002:
            raise ValueError("START_SETUP_NOT_REACHED")
        for spec in target_selection_runtime.DEFAULT_TARGET_SPECS:
            body = scene_factory.create_coloured_sphere(sim.GROUND_TARGET_RADIUS, spec.rgba, self.client)
            scene_factory.set_static_body_pose(
                body, target_selection_runtime.DEFAULT_TARGET_WORLD_POSITIONS[spec.target_id], self.client
            )
        self.obstacle = scene_factory.create_box_obstacle(
            config.OBSTACLE_DIMENSIONS_M, config.OBSTACLE_RGBA, self.client
        )
        scene_factory.set_static_body_pose(self.obstacle, OBSTACLE_DEMO_POSITION, self.client)
        self.summary["scene_creations"] += 1

    def perceive(self):
        self.set_state("PERCEPTION")
        localizer = ObstacleLocalizer(camera_intrinsics_from_fov(640, 480, CAMERA_FOV_Y_DEGREES),
            CAMERA_NEAR_PLANE, CAMERA_FAR_PLANE,
            config.OBSTACLE_DIMENSIONS_M, config.OBSTACLE_SAFETY_MARGIN_M)
        detector = YellowObstacleDetector()
        minima, maxima = [], []
        # Only RGB, synchronized depth, and camera pose enter Stage 17.
        for _ in range(60):
            for _ in range(sim.CAMERA_UPDATE_INTERVAL_STEPS):
                self.tick()
            rgbd = self.render()
            frame = rgbd.live_rgb_frame
            detection = detector.detect(frame.rgba_buffer, frame.image_width, frame.image_height)
            transform = camera_geometry.matrix_from_pose(frame.camera_world_position, frame.camera_world_orientation)
            if detection.valid:
                loc = localizer.localize(detection.mask, rgbd.depth_buffer, transform,
                                         camera_geometry.render_to_camera_axis_transform(transform, frame))
                if loc.valid:
                    minima.append(loc.occupancy_aabb_min_m)
                    maxima.append(loc.occupancy_aabb_max_m)
            if len(minima) >= 20:
                break
        if len(minima) < 20:
            raise ValueError(f"OBSTACLE_PERCEPTION_INVALID: {len(minima)}/20 frames")
        self.set_state("OBSTACLE_ESTIMATION")
        self.occupancy = AxisAlignedBox(tuple(np.median(minima, axis=0)), tuple(np.median(maxima, axis=0)))
        obstacle_perception_runtime.OccupancyDebugBox(self.client, []).update(
            self.occupancy.minimum, self.occupancy.maximum
        )
        print("EST OCC MIN:", self.occupancy.minimum)
        print("EST OCC MAX:", self.occupancy.maximum)
        print("SIZE:", np.subtract(self.occupancy.maximum, self.occupancy.minimum))
        print("Estimated obstacle center:", self.occupancy.center)
        print("Static scene: this fresh visual occupancy is latched, WITHOUT second inflation.")
        self.summary.update(estimated_aabb_min=self.occupancy.minimum, estimated_aabb_max=self.occupancy.maximum)

    def plan(self, seed):
        self.set_state("DIRECT_PATH_CHECK")
        actual = self.actual()
        if self.current_collision(actual):
            raise ValueError("CURRENT_START_COLLISION")
        start = self.edge_configuration(actual)
        locks = {i: self.context.locked_initial_positions[j] for i, j in enumerate(self.context.arm_joint_indices) if j in self.context.locked_initial_positions}
        for i, value in locks.items():
            start[i] = value
        goal = start.copy()
        goal[4:] = DEMO_ACTIVE_GOAL
        direct = self.checker.check_path(
            start, goal, self.occupancy, config.COLLISION_INTERPOLATION_RESOLUTION_RAD
        )
        print("Direct path result:", "COLLISION" if direct.collision else "SAFE")
        self.summary["direct_path_collision"] = direct.collision
        planning_visualization.draw_waypoint_path([start, goal], self.checker, self.client, (1, 0, 0) if direct.collision else (0, 1, 0), 2, "DIRECT")
        if direct.collision:
            self.source = "RRT_CONNECT"
            self.set_state("DIRECT_COLLISION")
            self.set_state("RRT_PLANNING")
            planner_config = RRTConnectConfig(random_seed=seed)  # unchanged Stage19 defaults
            result = RRTConnectPlanner(self.checker, self.occupancy, (4, 5, 6), locks,
                                       self.checker.arm_joint_limits, planner_config).plan(start, goal)
            print("RRT result:", "SUCCESS" if result.success else result.failure_reason)
            self.summary.update(rrt_success=result.success, seed=seed, planning_time_s=result.planning_time_s)
            if not result.success:
                raise ValueError("RRT_FAILED: " + result.failure_reason)
            path = result.path
        else:
            self.source = "DIRECT"
            path = (start, goal)
        self.set_state("RRT_PATH_VALIDATION" if direct.collision else "DIRECT_PATH_VALIDATION")
        # Constructor checks the actual dense path with the UNCHANGED checker.
        self.executor = TrajectoryExecutor(path, locks, self.checker.arm_joint_limits, self.edge_safe)
        self.summary.update(final_validation_pass=True, planned_waypoints=len(path),
                            execution_waypoints=len(self.executor.path), execution_source=self.source)
        print("Final validation: PASS")
        print(f"Planned waypoints: {len(path)} | Execution waypoints: {len(self.executor.path)}")
        print("Current q:", actual, "Goal q:", goal)
        planning_visualization.draw_waypoint_path(path, self.checker, self.client, (0, 1, 0), 4, "VALIDATED RRT" if direct.collision else "VALIDATED DIRECT")
        np.savetxt(LOG_ROOT / "stage20_demo_planned_path.csv", path, delimiter=",", header="q1,q2,q3,q4,q5,q6,q7", comments="")
        np.savetxt(LOG_ROOT / "stage20_demo_execution_path.csv", self.executor.path, delimiter=",", header="q1,q2,q3,q4,q5,q6,q7", comments="")
        self.set_state("READY_TO_EXECUTE")
        for _ in range(240):
            self.tick()

    def execute(self):
        self.set_state("TRAJECTORY_EXECUTION" if self.source == "RRT_CONNECT" else "DIRECT_EXECUTION")
        previous_index = -1
        errors = []
        while self.executor.state not in ("GOAL_REACHED", "SAFE_STOP"):
            q = self.actual()
            collision = self.current_collision(q)
            command = self.executor.update(q, sim.TIME_STEP, collision, self.edge_configuration(q))
            self.context.commanded_joint_targets = command.tolist()
            if self.executor.index != previous_index or self.executor.state == "SAFE_STOP":
                print(f"Waypoint {self.executor.index+1}/{len(self.executor.path)} | current collision={collision}\n"
                      f" q_command={np.round(command, 7)}\n q_actual={np.round(q, 7)}", flush=True)
                previous_index = self.executor.index
            self.tick()
            actual_after = self.actual()
            # Check immediately AFTER each physics step as well as BEFORE commands.
            after_collision = self.current_collision(actual_after)
            if after_collision:
                self.context.commanded_joint_targets = self.executor.stop(actual_after, "ACTUAL_CONFIGURATION_COLLISION").tolist()
            error = float(np.max(np.abs(actual_after[4:] - command[4:])))
            errors.append(error)
            self.log.write(json.dumps(dict(time=self.t, state=self.executor.state,
                waypoint=self.executor.index, q_command=command.tolist(), q_actual=actual_after.tolist(),
                tracking_error=error, actual_collision=after_collision,
                locked_max_deviation=self.context.max_locked_deviation)) + "\n")
        self.log.flush()
        self.summary.update(execution_time_s=self.executor.elapsed,
            mean_tracking_error=float(np.mean(errors)), rmse_tracking_error=float(np.sqrt(np.mean(np.square(errors)))),
            max_tracking_error=max(errors), goal_joint_error=float(np.max(np.abs(self.actual()[4:] - self.executor.path[-1, 4:]))))
        if self.executor.state == "SAFE_STOP":
            raise ValueError(self.executor.reason)
        self.set_state("GOAL_REACHED")
        self.context.commanded_joint_targets = self.executor.command.tolist()
        # Goal HOLD verification, still with actual geometry safety monitoring.
        for _ in range(5 * 240):
            self.tick()
            if self.current_collision(self.actual()):
                raise ValueError("ACTUAL_COLLISION_IN_GOAL_HOLD")
        self.set_state("HOLD")
        self.summary["goal_error_after_hold"] = float(np.max(np.abs(self.actual()[4:] - self.executor.path[-1, 4:])))
        self.summary["verified_goal_hold_s"] = 5.0
        self.summary["result"] = "PASS" if self.contacts == 0 and self.context.max_locked_deviation < .005 else "FAIL"

    def safe_stop(self, reason):
        if self.context is not None and p.isConnected(self.client):
            q = self.actual()
            for i, joint in enumerate(self.context.arm_joint_indices):
                if joint in self.context.locked_initial_positions:
                    q[i] = self.context.locked_initial_positions[joint]
            self.context.commanded_joint_targets = q.tolist()
        self.summary.update(result="FAIL", safe_stop_reason=str(reason))
        self.set_state("SAFE_STOP")
        print("SAFE_STOP latched. No home, no retry, no scene recreation:", reason, flush=True)

    def save_summary(self):
        self.summary.update(gt_collision_count=self.contacts,
            locked_joint_max_deviation=self.context.max_locked_deviation if self.context else None,
            terminal_state=self.state)
        (LOG_ROOT / "stage20_demo_summary.json").write_text(json.dumps(self.summary, indent=2), encoding="utf-8")
        print(json.dumps(self.summary, indent=2), flush=True)

    def close(self):
        self.log.close()
        self.checker.close()
        if self.display:
            self.display.close()
        if p.isConnected(self.client):
            p.disconnect(physicsClientId=self.client)


def run_stage20_demo(*, gui=True, realtime=True, keep_open=True, seed=DEMO_SEED,
                     mode_title="STAGE 20 - FIXED SINGLE DEMO",
                     window_title="Stage 20 - Obstacle-Aware Execution"):
    """CLI keeps the SAME scene in HOLD/SAFE_STOP until Q/ESC/window close.

    Headless bounded tests may set keep_open=False; CLI never does.
    """
    demo = Demo(gui, realtime, mode_title=mode_title, window_title=window_title)
    print("DEBUG_STAGE20 =", DEBUG_STAGE20)
    print("GT used for planning/execution/controller: False; contacts: evaluation only")
    print("Self-collision checking: NOT IMPLEMENTED")
    try:
        try:
            demo.set_state("INITIALIZE")
            demo.initialize()
            demo.perceive()
            demo.plan(seed)
            demo.execute()
        except (ValueError, RuntimeError) as error:
            demo.safe_stop(error)
        demo.save_summary()
        if keep_open:
            print("Demo remains open. Q/ESC, RGB window close, PyBullet close or Ctrl+C to quit.", flush=True)
            while True:
                demo.tick()
                if demo.occupancy and demo.state != "SAFE_STOP" and demo.current_collision(demo.actual()):
                    demo.safe_stop("ACTUAL_COLLISION_IN_HOLD")
                    demo.save_summary()
    except (UserExit, KeyboardInterrupt) as error:
        print("User exit:", error)
    except p.error:
        if p.isConnected(demo.client):
            raise
        print("PyBullet disconnected by user; no scene restart.")
    finally:
        demo.close()
    return demo.summary

