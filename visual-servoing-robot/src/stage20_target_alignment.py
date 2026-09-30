"""RED startup alignment, then manual Stage16 selection with a safety gate.

No fixed goal q means 'centered'. Only NEW selected-target RGB can declare
centering. Stage13/16 supply proposals; Stage18/19 authorize motor targets.
This is incremental visual path planning, not a GT-derived endpoint planner.
"""
from __future__ import annotations

import json
import time
import cv2
import numpy as np
import pybullet as p

import robotics_core as sim
import visual_servo_runtime as control
import predictive_measurement as predictive
import stage16_target_switching as selection
import target_selection_runtime as selection_runtime
from multi_target_detector import MultiTargetDetector
from target_manager import TargetManager
from camera_observation import render_live_eye_in_hand_rgbd_frame
from rrt_connect_planner import RRTConnectPlanner, RRTConnectConfig
from trajectory_executor import TrajectoryExecutor
from obstacle_motion_runtime import Demo, UserExit, LOG_ROOT


class SelectedTargetDemo(Demo):
    def __init__(self, gui=True, realtime=True):
        super().__init__(gui, realtime, "stage20_selection_execution.jsonl")
        self.manager = TargetManager(selection.STAGE16_PREDICTION_ALPHA,
                                     selection.STAGE16_PREDICTION_HORIZON_S)
        self.detector = MultiTargetDetector()
        self.manual_enabled = False
        self.alignment_active = False
        self.controller_source = "CURRENT_SWITCH_FALLBACK"
        self.ready_frames = 0
        self.centered = False
        self.blocked = False
        self.last_key_step = -1
        self.last_rgb_step = -1
        self.last_measurement = None
        self.switch_results = []
        self.selection_started_at = 0.0
        self.direct_checks = self.rrt_calls = 0
        self.last_status_second = -1

    def poll_exit(self):
        if not p.isConnected(self.client):
            raise UserExit("PyBullet closed")
        if self.gui and self.last_key_step != self.steps:
            self.last_key_step = self.steps
            # Consume the keyboard only once/step; Stage16 owns ID mapping.
            if self.manual_enabled:
                event, quit_requested = selection_runtime.manual_keyboard_input(self.manager, self.t, self.client)
                if event is not None:
                    self.begin_selection(event)
            else:
                events = p.getKeyboardEvents(physicsClientId=self.client)
                quit_requested = any(events.get(k, 0) & p.KEY_WAS_TRIGGERED for k in (ord('q'), ord('Q'), 27))
            if quit_requested:
                raise UserExit("Q/ESC")
        if self.display_was_open and not self.display.is_open:
            raise UserExit("RGB window closed")

    def hold_current(self):
        q = self.actual()
        for i, joint in enumerate(self.context.arm_joint_indices):
            if joint in self.context.locked_initial_positions:
                q[i] = self.context.locked_initial_positions[joint]
        self.context.commanded_joint_targets = q.tolist()
        self.executor = None

    def render(self):
        if not self.alignment_active:
            return super().render()
        self.poll_exit()
        pos, quat = sim.get_camera_optical_center_pose(self.context.robot_id, self.client)
        sim.update_camera_reference_axes(self.context.robot_id, self.client, self.context.camera_axis_debug_item_ids)
        # Capture once; the selected-target overlay below is the ONLY display.
        return render_live_eye_in_hand_rgbd_frame(pos, quat, self.client)

    def begin_selection(self, event):
        self.hold_current()  # cancel old target's route immediately
        self.context.servo = sim.TwoDimensionalVisualServo()
        self.ready_frames = 0
        self.centered = self.blocked = False
        self.selection_started_at = self.t
        self.source = "NONE"
        self.state = "SELECTED_TARGET_ALIGNMENT"
        self.controller_source = "CURRENT_SWITCH_FALLBACK"
        print(f"USER SWITCH: {event.previous_class} -> {event.selected_class}; CURRENT POSE; {event.source}", flush=True)
        self.log.write(json.dumps({"event": "selection", "time": self.t,
            "old": event.previous_class, "new": event.selected_class, "source": event.source}) + "\n")

    def tick(self):
        if not self.alignment_active:
            return super().tick()
        self.poll_exit()
        before = time.perf_counter()
        actual = self.actual()
        if self.current_collision(actual):
            raise ValueError("ACTUAL_CONFIGURATION_COLLISION")
        if self.executor is not None and not self.blocked:
            command = self.executor.update(actual, sim.TIME_STEP, False, self.edge_configuration(actual))
            self.context.commanded_joint_targets = command.tolist()
            if self.executor.state == "SAFE_STOP":
                raise ValueError(self.executor.reason)
            if self.executor.state == "GOAL_REACHED":
                self.hold_current()
                self.state = "VISUAL_RECHECK"  # waypoint arrival != visual success
        # Also gate tracking deviation toward the most recent direct command.
        if not self.edge_safe(self.edge_configuration(actual), self.edge_configuration(np.asarray(self.context.commanded_joint_targets))):
            raise ValueError("CURRENT_TO_COMMAND_PATH_COLLISION")
        control.step_physics(self.context, self.client)
        self.steps += 1
        self.t += sim.TIME_STEP
        if self.context.max_locked_deviation >= .005:
            raise ValueError("LOCKED_JOINT_DEVIATION")
        if self.current_collision(self.actual()):
            raise ValueError("ACTUAL_CONFIGURATION_COLLISION")
        self.contacts += int(bool(p.getContactPoints(self.context.robot_id, self.obstacle, physicsClientId=self.client)))
        if self.realtime:
            time.sleep(max(0, sim.TIME_STEP - (time.perf_counter() - before)))

    def initialize_alignment(self):
        self.initialize()
        self.perceive()
        ctx = self.context
        ctx.locked_joint_indices = list(ctx.locked_initial_positions)
        ctx.active_joint_indices = [j for j in ctx.arm_joint_indices if j not in ctx.locked_initial_positions]
        ctx.active_hold_targets = sim.capture_active_joint_hold_targets(ctx.robot_id, ctx.active_joint_indices, self.client)
        ctx.servo = sim.TwoDimensionalVisualServo()
        self.hold_current()
        self.alignment_active = True
        self.selection_started_at = self.t
        self.state = "INITIAL_RED_ALIGNMENT"
        print("Initial RED alignment -> RAW +/-2 px for 5 NEW frames -> HOLD -> enable manual 1/2/3.", flush=True)

    def authorize_proposal(self, proposed):
        actual = self.actual()
        start = self.edge_configuration(actual)
        goal = self.edge_configuration(np.asarray(proposed))
        locks = {i: self.context.locked_initial_positions[j] for i, j in enumerate(self.context.arm_joint_indices)
                 if j in self.context.locked_initial_positions}
        for i, v in locks.items():
            start[i] = goal[i] = v
        self.direct_checks += 1
        if self.edge_safe(start, goal):
            self.source = "DIRECT"
            self.context.commanded_joint_targets = goal.tolist()
            self.state = "VISUAL_ALIGNMENT"
            return
        self.hold_current()
        # RRT cannot rescue an endpoint already inside the occupancy.
        if self.checker.check_configuration(goal, self.occupancy).collision:
            self.blocked = True
            self.state = "BLOCKED_VISUAL_GOAL / HOLD"
            print("Visual goal COLLISION: HOLD. No unsafe command, no fixed alternate endpoint.", flush=True)
            return
        self.rrt_calls += 1
        self.state = "RRT_PLANNING"
        result = RRTConnectPlanner(self.checker, self.occupancy, (4, 5, 6), locks,
            self.checker.arm_joint_limits, RRTConnectConfig()).plan(start, goal)
        if not result.success:
            self.blocked = True
            self.state = "PLANNING_FAILED / HOLD"
            print("RRT failed:", result.failure_reason, flush=True)
            return
        self.executor = TrajectoryExecutor(result.path, locks, self.checker.arm_joint_limits, self.edge_safe)
        self.source = "RRT_CONNECT"
        self.state = "VALIDATED_RRT_EXECUTION"
        print(f"DIRECT COLLISION -> RRT SUCCESS -> DENSE PATH VALID ({len(self.executor.path)} points).", flush=True)

    def observation(self):
        if self.last_rgb_step == self.steps:
            return
        self.last_rgb_step = self.steps
        rgbd = self.render()
        frame = rgbd.live_rgb_frame
        detections = self.detector.detect(frame.rgba_buffer, frame.image_width, frame.image_height)
        estimates = self.manager.update_detections(self.t, detections, selected_only=not self.manual_enabled)
        raw = selection_runtime.controller_detection_from_selected(self.manager.selected_detection(),
            (frame.image_width // 2, frame.image_height // 2), frame.rgba_buffer)
        self.last_measurement = raw
        source = self.manager.controller_source_for_selected(estimates[self.manager.selected_target_id])
        measured = raw
        if source == "PREDICTED":
            measured, source, _ = predictive.predicted_measurement(raw, estimates[self.manager.selected_target_id])
        self.controller_source = source
        newly_centered = False
        if not raw.detected:
            self.hold_current()
            self.ready_frames = 0
            self.centered = False
            self.state = "TARGET LOST / HOLD"
            self.context.servo = sim.TwoDimensionalVisualServo()
        elif not self.blocked:
            ex, ey = raw.pixel_error
            within = abs(ex) <= 2 and abs(ey) <= 2
            if within:
                if not self.centered:
                    self.hold_current()
                self.ready_frames += 1
                self.state = "VERIFYING_RAW_CENTER"
                if self.ready_frames >= 5:
                    if not self.centered:
                        result = dict(target=self.manager.selected_class, raw_ex=ex, raw_ey=ey,
                            response_time_s=self.t - self.selection_started_at, gt_contacts=self.contacts,
                            locked_max_deviation=self.context.max_locked_deviation)
                        self.switch_results.append(result)
                        newly_centered = True
                        print("VISUALLY CENTERED:", result, flush=True)
                    self.centered = True
                    self.manual_enabled = True
                    self.state = "ALIGNED / HOLD - PRESS 1/2/3"
            elif self.executor is None:
                self.centered = False
                self.ready_frames = 0
                self.state = "VISUAL_ALIGNMENT"
                # Frozen controller writes a PROPOSAL to context, NOT to motors.
                previous = list(self.context.commanded_joint_targets)
                issued, _ = control.control_latest_measurement(self.context, measured, self.client)
                proposed = list(self.context.commanded_joint_targets)
                self.context.commanded_joint_targets = previous
                if issued:
                    self.authorize_proposal(proposed)
                else:
                    self.hold_current()
        if self.display or newly_centered:
            overlay = selection_runtime.annotated_overlay(frame.rgba_buffer, frame.image_width,
                frame.image_height, detections, self.manager, "STAGE 20 - SELECT / PLAN / ALIGN",
                self.state, self.controller_source, self.state)
            if self.display:
                self.display.show(frame, overlay)
            if newly_centered:
                pixels = np.frombuffer(overlay, dtype=np.uint8).reshape(frame.image_height, frame.image_width, 4)
                cv2.imwrite(str(LOG_ROOT.parent / f"stage20_selected_{self.manager.selected_class.lower()}.png"),
                            cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR))
                self.save_report()
        self.debug_id = sim.update_motion_debug_text(
            f"STAGE 20 | {self.state}\nSelected: {self.manager.selected_class} | {self.source}\n"
            f"Raw ex/ey: {raw.pixel_error if raw.detected else 'N/A'} | [1] RED [2] GREEN [3] BLUE", self.debug_id, self.client)
        row = dict(time=self.t, selected=self.manager.selected_class, phase=self.state,
            detected=raw.detected, raw_error=raw.pixel_error, controller_source=self.controller_source,
            execution_source=self.source, ready_frames=self.ready_frames,
            current_q=self.actual().tolist(), commanded_q=self.context.commanded_joint_targets,
            locked_max_deviation=self.context.max_locked_deviation, gt_contacts_evaluation_only=self.contacts)
        self.log.write(json.dumps(row) + "\n")
        if int(self.t) != self.last_status_second:
            self.last_status_second = int(self.t)
            print(f"{self.t:.1f}s {self.manager.selected_class} raw={raw.pixel_error} {self.state}", flush=True)
            self.log.flush()

    def save_report(self):
        report = dict(results=self.switch_results, final_state=self.state,
            selected=self.manager.selected_class, raw_error=self.last_measurement.pixel_error if self.last_measurement else None,
            direct_checks=self.direct_checks, rrt_calls=self.rrt_calls, gt_contacts=self.contacts,
            locked_max_deviation=self.context.max_locked_deviation if self.context else None,
            gt_used_for_controller=False, gt_used_for_planning=False)
        (LOG_ROOT / "stage20_selection_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def run_stage20_selection(*, gui=True, realtime=True, max_frames=None, test_switch_ids=()):
    """Production CLI is indefinite/manual. Test driver selection is explicit."""
    demo = SelectedTargetDemo(gui, realtime)
    frames = 0
    tests = iter(test_switch_ids)
    next_test = next(tests, None)
    try:
        demo.initialize_alignment()
        while True:
            demo.tick()
            if demo.steps % sim.CAMERA_UPDATE_INTERVAL_STEPS == 0:
                demo.observation()
                frames += 1
                # Test driver observes ~1 s of stable HOLD before emulating
                # the next key. Production has no automatic switch at all.
                test_hold_verified = demo.centered and demo.ready_frames >= 35
                if not gui and test_hold_verified and next_test:
                    event = demo.manager.select_target(next_test, demo.t, "TEST_KEYBOARD")
                    if event:
                        demo.begin_selection(event)
                    next_test = next(tests, None)
                if max_frames is not None and (frames >= max_frames or demo.blocked or (test_hold_verified and demo.centered and next_test is None)):
                    break
    except (UserExit, KeyboardInterrupt):
        pass
    except p.error:
        if p.isConnected(demo.client):
            raise
        print("PyBullet closed; no scene recreation.")
    except (ValueError, RuntimeError) as error:
        demo.hold_current()
        demo.blocked = True
        demo.manual_enabled = False  # a latched safety stop cannot be cleared by selection
        demo.state = "SAFE_STOP"
        print("SAFE_STOP:", error, flush=True)
        if gui:
            # Latch current pose. Never rebuild the scene or restart a trial.
            demo.alignment_active = False
            while p.isConnected(demo.client):
                try:
                    demo.tick()
                except (UserExit, KeyboardInterrupt):
                    break
    finally:
        report = demo.save_report()
        demo.close()
    return report
