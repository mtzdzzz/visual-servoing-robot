"""Stage 21 integration-layer regressions; frozen algorithms are mocked, not changed."""

import unittest
from unittest.mock import patch
import inspect
import numpy as np

import demo_config
from demo_tracking import run_demo_tracking
from demo_multitarget import run_demo_multitarget
from demo_obstacle import run_demo_obstacle
from demo_ui import status_panel
from target_manager import TargetManager


class Stage21IntegrationTests(unittest.TestCase):
    def test_status_panel_accepts_bytes_and_array_without_changing_size(self):
        array = np.zeros((48, 64, 4), dtype=np.uint8)
        self.assertEqual(len(status_panel(array, 64, 48, ["A", "B"])), array.size)
        self.assertEqual(len(status_panel(array.tobytes(), 64, 48, ["A"])), array.size)

    def test_obstacle_demo_uses_fixed_seed_and_keeps_open(self):
        with patch("demo_obstacle.run_stage20_demo", return_value={"result": "PASS"}) as runner:
            result = run_demo_obstacle()
        self.assertEqual(result["result"], "PASS")
        runner.assert_called_once_with(
            gui=True, realtime=True, keep_open=True,
            seed=demo_config.OBSTACLE_DEMO_RANDOM_SEED,
            mode_title=demo_config.DEMO_OBSTACLE_TITLE,
            window_title=demo_config.DEMO_OBSTACLE_TITLE,
        )

    def test_multitarget_default_never_changes_without_manual_selection(self):
        manager = TargetManager(.20, .30, demo_config.DEFAULT_SELECTED_TARGET_ID)
        self.assertEqual(manager.selected_class, "RED")
        self.assertEqual(manager.last_selection_source, "MANUAL_DEFAULT")
        # Time and convergence do not exist in TargetManager selection logic.
        for _ in range(1000):
            manager.update_detections(float(_), [])
        self.assertEqual(manager.selected_class, "RED")

    def test_demo_config_contains_no_controller_tuning(self):
        forbidden = ("KP", "MAX_STEP", "DEADBAND", "GAIN", "FORCE", "VELOCITY")
        names = tuple(name.upper() for name in vars(demo_config))
        self.assertFalse(any(token in name for token in forbidden for name in names))

    def test_production_demos_have_no_frame_limit(self):
        self.assertIsNone(inspect.signature(run_demo_tracking).parameters["max_frames"].default)
        self.assertIsNone(inspect.signature(run_demo_multitarget).parameters["max_frames"].default)


if __name__ == "__main__":
    unittest.main()
