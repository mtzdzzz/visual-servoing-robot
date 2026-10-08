# Reproducing Experiments

Run commands from the repository root in an activated Python 3.10 virtual environment. PyBullet GUI experiments require a desktop OpenGL session. Existing files under `outputs/logs/` may be overwritten by a rerun; back them up when preservation matters.

| Stage | Purpose | Command | Primary output | Primary metric |
|---:|---|---|---|---|
| 7 | Five-position static visual-servo validation | `python .\src\simulation.py --stage7` | `outputs/logs/stage7_trial_*.csv` | Convergence success, final pixel error, convergence time |
| 9 | SLOW/MEDIUM/FAST dynamic tracking | `python .\src\simulation.py --stage9` | `outputs/logs/stage9_*.csv` | Mean error, RMSE, P95, detection rate |
| 10 | Target-loss recovery and visual-noise robustness | `python .\src\simulation.py --stage10` | `outputs/logs/stage10_*.csv` | Reacquisition/recovery time, noise RMSE |
| 12 | Online image-plane motion estimation | `python .\src\simulation.py --stage12-estimation` | `outputs/logs/stage12_motion_estimation.csv` | Future-centroid baseline vs EMA prediction error |
| 12B | Offline alpha/horizon grid | `python .\src\evaluate_stage12b.py` | `outputs/logs/stage12b_prediction_grid.csv` | Prediction RMSE and improvement across 25 settings |
| 13 | Baseline/predictive closed-loop A/B test | `python .\src\simulation.py --stage13` | `outputs/logs/stage13_*.csv` | Raw RGB tracking RMSE improvement |
| 14 | Five-position RGB-D target localization | `python .\src\simulation.py --stage14-rgbd` | `outputs/logs/stage14_rgbd_*.csv` | Mean/RMSE/median/max 3D center error |
| 15 | R/G/B multi-target RGB-D perception | `python .\src\simulation.py --stage15-multitarget` | `outputs/logs/stage15_*.csv` | Per-class detection/localization, confusion count |
| 16 | Scheduled target-switching evaluation | `python .\src\simulation.py --stage16-eval` | `outputs/logs/stage16_*.csv` | Switch success and response time |
| 17 | Five-scene obstacle perception | `python .\src\simulation.py --stage17-obstacle` | `outputs/logs/stage17_*.csv` | Center error, AABB IoU, GT coverage |
| 18 | Estimated-occupancy collision evaluation | `python .\src\simulation.py --stage18-collision` | `outputs/logs/stage18_*.csv` | TP/TN/FP/FN and safety recall |
| 19 | RRT-Connect planning without execution | `python .\src\simulation.py --stage19-rrt-connect` | `outputs/logs/stage19_*.csv` | Success rate, planning time, path length |
| 20 | Collision-aware path execution | `python .\src\simulation.py --stage20-execute` | `outputs/logs/stage20_*.csv` | Goal/tracking error, locked-joint deviation, GT collision |

## Interactive and Final Demo Entries

| Mode | Command | Interaction |
|---|---|---|
| Stage 16 manual selection | `python .\src\simulation.py --stage16-selection` | `1`/`2`/`3` selects R/G/B; no automatic switch |
| Predictive tracking demo | `python .\src\simulation.py --demo-tracking` | Drag the red target in PyBullet |
| Multi-target demo | `python .\src\simulation.py --demo-multitarget` | Select R/G/B with `1`/`2`/`3` |
| Obstacle-aware demo | `python .\src\simulation.py --demo-obstacle` | Deterministic perception → RRT → execution |

All interactive modes support `Q`, `Esc`, `Ctrl+C`, or PyBullet GUI closure.

## Analysis-Only Commands

The final Stage 23 report can be regenerated from existing CSV files without rerunning PyBullet:

```powershell
python .\src\stage23_final_summary.py
```

This command reads historical logs and writes `outputs/final_report/`. It does not change the controller, planner, or historical CSV data.
