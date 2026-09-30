"""Display-only helpers for planned robot paths."""
from __future__ import annotations

from typing import Sequence
import pybullet as p
from collision_checker import CollisionChecker


def draw_waypoint_path(path: Sequence[Sequence[float]], checker: CollisionChecker, client_id: int,
                       colour: tuple[float, float, float], line_width: float,
                       label: str | None = None) -> None:
    if not path:
        return
    points = [checker.end_effector_position(configuration) for configuration in path]
    for first, second in zip(points, points[1:]):
        p.addUserDebugLine(first, second, lineColorRGB=colour, lineWidth=line_width,
                           lifeTime=0, physicsClientId=client_id)
    if label:
        endpoint = points[-1]
        p.addUserDebugText(label, (endpoint[0], endpoint[1], endpoint[2] + 0.025),
                           textColorRGB=colour, textSize=0.75, lifeTime=0,
                           physicsClientId=client_id)
