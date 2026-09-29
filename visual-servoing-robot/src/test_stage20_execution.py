"""Execution-architecture regressions, without modifying the frozen checker."""
import unittest
import numpy as np
from trajectory_executor import TrajectoryExecutor, densify_path


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.start = np.zeros(7)
        self.goal = np.array([0., 0., 0., 0., .04, .03, -.02])
        self.limits = [(-3., 3.)] * 7
        self.locks = dict.fromkeys(range(4), 0.)

    def make(self, edge=lambda a, b: True):
        return TrajectoryExecutor([self.start, self.goal], self.locks, self.limits, edge)

    def test_collision_path_rejected_before_execution(self):
        with self.assertRaises(ValueError):
            self.make(lambda a, b: False)

    def test_exact_dense_path_and_limits(self):
        dense = densify_path([self.start, self.goal])
        self.assertLessEqual(np.max(np.abs(np.diff(dense, axis=0))), .0100000001)
        np.testing.assert_allclose(dense[-1], self.goal)
        invalid = self.goal.copy()
        invalid[0] = .01
        with self.assertRaises(ValueError):
            TrajectoryExecutor([self.start, invalid], self.locks, self.limits, lambda a, b: True)

    def test_collision_stop_latches_current_not_initial(self):
        exe = self.make()
        current = self.goal / 2
        held = exe.update(current, 1/240, True)
        np.testing.assert_allclose(held, current)
        np.testing.assert_allclose(exe.update(self.start, 1/240, False), current)
        self.assertEqual(exe.state, "SAFE_STOP")

    def test_timeout(self):
        exe = self.make()
        for _ in range(2000):
            exe.update(self.start, 1/240, False)
        self.assertEqual(exe.state, "SAFE_STOP")
        self.assertEqual(exe.reason, "WAYPOINT_TIMEOUT")

    def test_goal_hold_and_locked_targets(self):
        exe = self.make()
        actual = self.start.copy()
        for _ in range(3000):
            actual = exe.update(actual, 1/240, False)
            np.testing.assert_array_equal(actual[:4], self.start[:4])
            if exe.state == "GOAL_REACHED":
                break
        self.assertEqual(exe.state, "GOAL_REACHED")
        np.testing.assert_allclose(actual, self.goal, atol=.0005)
        np.testing.assert_allclose(exe.update(actual, 1/240, False), actual)

    def test_upcoming_segment_rechecked(self):
        exe = self.make()
        exe.edge_safe = lambda a, b: False
        exe.update(self.start, 1/240, False)
        self.assertEqual(exe.reason, "UPCOMING_SEGMENT_COLLISION")

    def test_invalid_measurement_never_commands_nan(self):
        exe = self.make()
        command = exe.update(np.full(7, np.nan), 1/240, False)
        self.assertTrue(np.isfinite(command).all())
        self.assertEqual(exe.state, "SAFE_STOP")


if __name__ == "__main__":
    unittest.main()
