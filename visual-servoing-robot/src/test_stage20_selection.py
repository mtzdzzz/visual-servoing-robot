"""Selected-target safety and raw-RGB completion regressions."""
import io
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import numpy as np

from multi_target_detector import DetectedTarget
from target_manager import TargetManager
from stage20_target_alignment import SelectedTargetDemo
from stage20_trajectory_execution import Demo


class SelectionTests(unittest.TestCase):
    def make_demo(self):
        d = SelectedTargetDemo.__new__(SelectedTargetDemo)
        d.manager = TargetManager(.20, .30)
        d.context = NS(arm_joint_indices=list(range(7)), locked_initial_positions=dict.fromkeys(range(4), 0.),
                       commanded_joint_targets=[0.] * 7, max_locked_deviation=0., servo=NS())
        d.actual = Mock(return_value=np.array([0., 0., 0., 0., .2, .4, -.3]))
        d.edge_configuration = lambda q: np.asarray(q).copy()
        d.checker = NS(arm_joint_limits=[(-3., 3.)] * 7,
                       check_configuration=Mock(return_value=NS(collision=True)))
        d.edge_safe = Mock(return_value=True)
        d.occupancy = object()
        d.direct_checks = d.rrt_calls = 0
        d.executor = object()
        d.manual_enabled = False
        d.blocked = d.centered = False
        d.ready_frames = 0
        d.state = "INITIAL_RED_ALIGNMENT"
        d.source = "NONE"
        d.last_rgb_step = -1
        d.steps = 8
        d.t = 1.
        d.client = 0
        d.display = None
        d.debug_id = None
        d.log = io.StringIO()
        d.switch_results = []
        d.selection_started_at = 0.
        d.contacts = 0
        d.last_status_second = 1
        d.save_report = Mock()
        return d

    def test_direct_safe_proposal_preserves_locks(self):
        d = self.make_demo()
        with patch('stage20_target_alignment.RRTConnectPlanner') as planner:
            d.authorize_proposal([.001, 0, 0, 0, .25, .4, -.3])
        planner.assert_not_called()
        self.assertEqual(d.context.commanded_joint_targets[:4], [0.] * 4)
        self.assertEqual(d.source, "DIRECT")

    def test_collision_endpoint_never_reaches_motor_or_rrt(self):
        d = self.make_demo()
        d.edge_safe.return_value = False
        with patch('stage20_target_alignment.RRTConnectPlanner') as planner:
            d.authorize_proposal([0, 0, 0, 0, 2, 2, 2])
        planner.assert_not_called()
        np.testing.assert_allclose(d.context.commanded_joint_targets, d.actual())
        self.assertTrue(d.blocked)
        self.assertIsNone(d.executor)

    def test_safe_endpoint_blocked_edge_calls_rrt_then_validates(self):
        d = self.make_demo()
        d.edge_safe.side_effect = [False] + [True] * 100
        d.checker.check_configuration.return_value = NS(collision=False)
        goal = np.array([0., 0., 0., 0., .3, .4, -.3])
        result = NS(success=True, path=[d.actual(), goal])
        with patch('stage20_target_alignment.RRTConnectPlanner') as planner:
            planner.return_value.plan.return_value = result
            d.authorize_proposal(goal)
        planner.return_value.plan.assert_called_once()
        self.assertEqual(d.source, "RRT_CONNECT")
        self.assertIsNotNone(d.executor)
        np.testing.assert_allclose(d.context.commanded_joint_targets, d.actual())
        self.assertGreater(d.edge_safe.call_count, 1)

    def test_switch_cancels_old_path_and_holds_current(self):
        d = self.make_demo()
        event = d.manager.select_target("target_2", d.t, "MANUAL_KEYBOARD")
        d.begin_selection(event)
        self.assertIsNone(d.executor)
        np.testing.assert_allclose(d.context.commanded_joint_targets, d.actual())
        self.assertEqual(d.manager.selected_class, "GREEN")
        self.assertEqual(d.controller_source, "CURRENT_SWITCH_FALLBACK")

    def test_stage16_keys_only_enabled_after_initial_red_alignment(self):
        d = self.make_demo()
        d.gui = True
        d.last_key_step = -1
        d.display_was_open = False
        with patch('stage20_target_alignment.p.isConnected', return_value=True), \
             patch('stage20_target_alignment.p.getKeyboardEvents', return_value={ord('2'): 2}):
            d.poll_exit()
        self.assertEqual(d.manager.selected_class, "RED")
        d.manual_enabled = True
        for key, expected in [('2', 'GREEN'), ('3', 'BLUE'), ('1', 'RED')]:
            d.steps += 1
            with patch('stage20_target_alignment.p.isConnected', return_value=True), \
                 patch('stage20_target_alignment.p.getKeyboardEvents', return_value={ord(key): 2}):
                d.poll_exit()
            self.assertEqual(d.manager.selected_class, expected)
            self.assertEqual(d.manager.last_selection_source, "MANUAL_KEYBOARD")
        d.steps += 30000
        with patch('stage20_target_alignment.p.isConnected', return_value=True), \
             patch('stage20_target_alignment.p.getKeyboardEvents', return_value={}):
            d.poll_exit()
        self.assertEqual(d.manager.selected_class, "RED")

    def observe(self, d, detections):
        frame = NS(rgba_buffer=np.zeros((480, 640, 4), dtype=np.uint8), image_width=640, image_height=480)
        d.detector = NS(detect=Mock(return_value=detections))
        d.steps += 8
        d.t += 1/30
        with patch.object(SelectedTargetDemo, 'render', return_value=NS(live_rgb_frame=frame)), \
             patch('stage20_target_alignment.cv2.imwrite'), \
             patch('stage20_target_alignment.sim.update_motion_debug_text', return_value=1), \
             patch('stage20_target_alignment.control.control_latest_measurement', return_value=(False, False)) as controller:
            d.observation()
            return controller.call_count

    def target(self, name="RED", xy=(320, 240)):
        target_id = {"RED": "target_1", "GREEN": "target_2"}[name]
        contour = np.array([[[300, 220]], [[340, 220]], [[340, 260]], [[300, 260]]], dtype=np.int32)
        return DetectedTarget(target_id, name, True, True, 1., xy, contour,
                              np.zeros((480, 640), dtype=np.uint8), (300, 220, 40, 40), 100.)

    def test_five_new_raw_frames_required_no_auto_switch(self):
        d = self.make_demo()
        for _ in range(4):
            self.observe(d, [self.target(xy=(318, 242))])
            self.assertFalse(d.manual_enabled)
        self.observe(d, [self.target(xy=(318, 242))])
        self.assertTrue(d.manual_enabled)
        self.assertTrue(d.centered)
        self.assertEqual(d.manager.selected_class, "RED")
        count = d.ready_frames
        d.observation()  # same physics-step/frame rejected
        self.assertEqual(d.ready_frames, count)

    def test_lost_selected_does_not_follow_visible_other(self):
        d = self.make_demo()
        self.assertEqual(self.observe(d, [self.target("GREEN")]), 0)
        self.assertEqual(d.manager.selected_class, "RED")
        self.assertEqual(d.state, "TARGET LOST / HOLD")
        self.assertIsNone(d.last_measurement.pixel_error)
        self.assertIsNone(d.executor)

    def test_outside_raw_tolerance_cannot_stay_centered(self):
        d = self.make_demo()
        d.executor = None
        d.centered = True
        self.observe(d, [self.target(xy=(323, 240))])
        self.assertFalse(d.centered)


if __name__ == '__main__':
    unittest.main()
