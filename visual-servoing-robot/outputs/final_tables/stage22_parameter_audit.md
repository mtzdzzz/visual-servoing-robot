# Stage 22 Parameter Freeze Audit

Stage 22 performed structural extraction only. Values below were read from the authoritative runtime modules after refactoring and compared with the pre-refactor audit captured before edits.

| Parameter | Before | After | Result |
|---|---:|---:|---|
| Visual-servo Kp | 0.0001 m/px | 0.0001 m/px | UNCHANGED |
| Predictor EMA alpha | 0.20 | 0.20 | UNCHANGED |
| Online prediction tau | 0.30 s | 0.30 s | UNCHANGED |
| Base MAX_STEP | 0.002 m | 0.002 m | UNCHANGED |
| Deadband X / Y | 2 / 2 px | 2 / 2 px | UNCHANGED |
| Camera resolution | 640 x 480 | 640 x 480 | UNCHANGED |
| Camera vertical FOV | 60 deg | 60 deg | UNCHANGED |
| Camera near / far | 0.01 / 3.0 m | 0.01 / 3.0 m | UNCHANGED |
| T_E_C translation | (0, 0.07, 0) m | (0, 0.07, 0) m | UNCHANGED |
| T_E_C quaternion | (-0.0226656354, -0.1285432061, -0.1721625934, 0.9763825862) | same | UNCHANGED |
| Locked arm joints | panda_joint1-4 | panda_joint1-4 | UNCHANGED |
| Active arm joints | panda_joint5-7 | panda_joint5-7 | UNCHANGED |
| positionGain | 0.03 | 0.03 | UNCHANGED |
| velocityGain | 0.8 | 0.8 | UNCHANGED |
| motor force | 20.0 | 20.0 | UNCHANGED |
| maxVelocity | 0.20 rad/s | 0.20 rad/s | UNCHANGED |
| Occupancy safety margin | 0.020 m | 0.020 m | UNCHANGED |
| Collision interpolation | 0.05 rad | 0.05 rad | UNCHANGED |
| RRT step | 0.15 rad | 0.15 rad | UNCHANGED |
| RRT goal bias | 0.10 | 0.10 | UNCHANGED |
| RRT max iterations / time | 2500 / 5.0 s | 2500 / 5.0 s | UNCHANGED |
| Execution interpolation | 0.01 rad | 0.01 rad | UNCHANGED |
| Waypoint tolerance | 0.0005 rad | 0.0005 rad | UNCHANGED |

Overall result: **UNCHANGED**.

The live audit source is `src/config.py::parameter_snapshot()`. It imports values from their owning modules instead of defining a second numeric copy.
