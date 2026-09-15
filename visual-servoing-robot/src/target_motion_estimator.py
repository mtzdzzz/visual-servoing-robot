"""Image-plane target motion estimation from timestamped RGB centroids only.

The estimator deliberately has no PyBullet, robot, world-coordinate, camera,
trajectory, or controller dependency.  Its complete input contract is the
timestamp and the OpenCV centroid measured in the current RGB frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


DEFAULT_EMA_ALPHA = 0.35
DEFAULT_PREDICTION_HORIZON_SECONDS = 0.10


@dataclass(frozen=True)
class MotionEstimate:
    """One finite image-plane velocity and prediction observation."""

    timestamp_s: float
    u: float
    v: float
    estimator_valid: bool
    u_dot_raw: float
    v_dot_raw: float
    u_dot_filtered: float
    v_dot_filtered: float
    u_pred: float
    v_pred: float
    prediction_horizon_s: float
    status: str


class TargetMotionEstimator:
    """Finite-difference velocity plus fixed-alpha EMA smoothing.

    With two strictly time-ordered detections:

    ``u_dot_raw = (u_t - u_previous) / dt``
    ``v_dot_raw = (v_t - v_previous) / dt``

    and the filtered velocity uses:

    ``v_filtered = alpha * v_raw + (1 - alpha) * v_filtered_previous``.

    Predicted centroids are display/log-only:

    ``u_pred = u + u_dot_filtered * tau``
    ``v_pred = v + v_dot_filtered * tau``.
    """

    def __init__(
        self,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        prediction_horizon_s: float = DEFAULT_PREDICTION_HORIZON_SECONDS,
    ) -> None:
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1].")
        if not prediction_horizon_s > 0.0:
            raise ValueError("prediction_horizon_s must be positive.")
        self.ema_alpha = float(ema_alpha)
        self.prediction_horizon_s = float(prediction_horizon_s)
        self.reset()

    def reset(self) -> None:
        """Invalidate state after target loss; the next detection initializes."""
        self._previous_timestamp_s: float | None = None
        self._previous_u: float | None = None
        self._previous_v: float | None = None
        self._previous_u_dot_filtered = 0.0
        self._previous_v_dot_filtered = 0.0
        self._has_filtered_velocity = False

    def target_lost(self) -> None:
        """Explicit, stable handling for an invalid or missing RGB detection."""
        self.reset()

    def update(self, timestamp_s: float, u: float, v: float) -> MotionEstimate:
        """Estimate velocity from one valid, timestamped OpenCV centroid.

        A first detection (including first detection after loss) initializes
        the position with zero velocity. Non-positive timestamp intervals are
        ignored rather than divided by, preserving the last valid state.
        """
        timestamp_s = float(timestamp_s)
        u = float(u)
        v = float(v)
        if not all(isfinite(value) for value in (timestamp_s, u, v)):
            self.reset()
            raise ValueError("TargetMotionEstimator requires finite timestamp, u, and v values.")

        if self._previous_timestamp_s is None:
            self._set_previous(timestamp_s, u, v)
            return self._initial_estimate(timestamp_s, u, v, "INITIALIZED")

        dt = timestamp_s - self._previous_timestamp_s
        if dt <= 0.0:
            # Do not overwrite the valid history with out-of-order data and
            # never divide by zero. The returned estimate is visibly invalid.
            return MotionEstimate(
                timestamp_s, u, v, False, 0.0, 0.0,
                self._previous_u_dot_filtered, self._previous_v_dot_filtered,
                u, v, self.prediction_horizon_s, "NONPOSITIVE_DT_IGNORED",
            )

        assert self._previous_u is not None and self._previous_v is not None
        u_dot_raw = (u - self._previous_u) / dt
        v_dot_raw = (v - self._previous_v) / dt
        if self._has_filtered_velocity:
            u_dot_filtered = (
                self.ema_alpha * u_dot_raw
                + (1.0 - self.ema_alpha) * self._previous_u_dot_filtered
            )
            v_dot_filtered = (
                self.ema_alpha * v_dot_raw
                + (1.0 - self.ema_alpha) * self._previous_v_dot_filtered
            )
        else:
            u_dot_filtered = u_dot_raw
            v_dot_filtered = v_dot_raw
            self._has_filtered_velocity = True

        self._previous_u_dot_filtered = u_dot_filtered
        self._previous_v_dot_filtered = v_dot_filtered
        self._set_previous(timestamp_s, u, v)
        return MotionEstimate(
            timestamp_s, u, v, True, u_dot_raw, v_dot_raw,
            u_dot_filtered, v_dot_filtered,
            u + u_dot_filtered * self.prediction_horizon_s,
            v + v_dot_filtered * self.prediction_horizon_s,
            self.prediction_horizon_s, "VALID",
        )

    def _set_previous(self, timestamp_s: float, u: float, v: float) -> None:
        self._previous_timestamp_s = timestamp_s
        self._previous_u = u
        self._previous_v = v

    def _initial_estimate(self, timestamp_s: float, u: float, v: float, status: str) -> MotionEstimate:
        return MotionEstimate(
            timestamp_s, u, v, False, 0.0, 0.0, 0.0, 0.0,
            u, v, self.prediction_horizon_s, status,
        )
