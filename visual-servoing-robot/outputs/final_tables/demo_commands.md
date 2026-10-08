# Final System Demos

## Demo 1 - Predictive Visual Tracking

```powershell
python .\src\simulation.py --demo-tracking
```

- Drag the red sphere with the left mouse button in the PyBullet window.
- The Eye-in-Hand RGB detector and predictive visual servo continuously recenter it.
- `Q` / `Esc` / `Ctrl+C`: quit.

## Demo 2 - Multi-Target Selection

```powershell
python .\src\simulation.py --demo-multitarget
```

- `1`: select RED.
- `2`: select GREEN.
- `3`: select BLUE.
- The robot continues from its current pose; there is no automatic switching.
- `Q` / `Esc` / `Ctrl+C`: quit.

## Demo 3 - Obstacle-Aware Motion

```powershell
python .\src\simulation.py --demo-obstacle
```

- A deterministic scene is perceived with RGB-D.
- The colliding direct path is rejected; RRT-Connect plans and validates a safe path.
- Panda executes the validated path with POSITION_CONTROL, reaches the goal, then holds.
- `Q` / `Esc` / `Ctrl+C`: quit.
