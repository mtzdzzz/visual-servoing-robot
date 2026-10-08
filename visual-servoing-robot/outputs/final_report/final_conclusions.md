# Final Experimental Conclusions

## 1. Visual Servo Convergence

The five Stage 7 raw trials produced a 100.0% static convergence success rate under the ±2 px final criterion. The historical Stage 7 summary is incomplete, so the final values are recomputed from all five raw trial files.

## 2. Dynamic Tracking

After READY and with warm-up excluded, Stage 9 tracking RMSE was 6.597 px for SLOW, 8.091 px for MEDIUM, and 9.912 px for FAST. Error increased with target speed in these three formal runs, while detection remained complete.

## 3. Predictive Visual Servo

Using the best online-tested horizon in the recorded Stage 13 grid (tau=0.30 s), predictive compensation reduced raw RGB tracking RMSE by 9.37% for MEDIUM and 15.89% for FAST. This conclusion is limited to the tested speeds, estimator setting, and horizons.

## 4. RGB-D Localization

The Stage 14 sphere-center estimates had a mean 3D error of 17.67 mm across five static positions. Surface-depth and center estimates remain distinct; the reported center result uses the documented known-radius compensation.

## 5. Multi-Target Perception

Stage 15 detected RED, GREEN, and BLUE at 100% in the recorded evaluation, with zero color-confusion events. The historical Stage 16 formal switching log is less successful: its overall switch success rate is 50.0% and its recorded result is FAIL. Later manual Demo behavior is not substituted for this historical formal dataset.

## 6. Obstacle Perception

Stage 17 achieved a mean obstacle-center error of 8.00 mm. The estimated occupancy covered 100.0% of the ground-truth AABB with 40.82% IoU. The lower IoU reflects the deliberately enlarged conservative proxy rather than a missed obstacle.

## 7. Collision Safety

Stage 18 recorded 0 false negatives and 100.0% safety recall. False positives are the expected cost of conservative occupancy: safety coverage improved while usable free space decreased.

## 8. RRT Planning

RRT-Connect achieved 100.0% success across the recorded Stage 19 RRT trials and tested seeds, with no ground-truth collision on final planned paths. This is an empirical result for the tested scenarios, not a guarantee of complete planning success.

## 9. Trajectory Execution

Stage 20 completed all five formal execution trials (100.0% success) with 0 ground-truth obstacle collisions. The raw execution trace confirms commanded and actual active-joint trajectories were recorded during motor-driven execution.

## 10. Overall Conclusion

The project demonstrates an integrated simulation pipeline from Eye-in-Hand RGB-D perception through target selection, predictive visual servoing, conservative collision checking, RRT-Connect planning, and motor-controlled Panda execution. Results support stable behavior in the tested scenarios, while the stated perception, planning-space, self-collision, static-obstacle, and simulation limitations remain material.

## Data Quality Findings

- stage7_trial_03.csv: time_s contains 1 missing timestamp value(s) in group all.
- stage7_summary.csv is incomplete: it records trial(s) [2], while five raw CSVs provide [1, 2, 3, 4, 5]. Stage 23 recomputes Stage 7 metrics from the raw files.
- stage7_summary.csv Trial 2 disagrees with stage7_trial_02.csv on initial_ex, initial_ey, initial_error_norm, final_ex, final_error_norm, convergence_time_s; Stage 23 uses the raw trial values.
- stage16_summary.csv: historical formal switching evaluation is FAIL (50.0% success; locked-joint maximum deviation 0.020151 rad).
- stage16_target_switching.csv contains mixed switch sources ['AUTOMATED_TEST', 'MANUAL_KEY']; it is not a manual-only Demo log.
- stage19_rrt_connect.csv: 2 NEAR_BOUNDARY rows are marked rrt_required but were not counted as RRT trials because GOAL_CONFIGURATION_COLLISION stopped planning before a seeded search. The summary's 25-trial denominator is retained.
- stage20_execution.csv: no EXECUTING rows were found.

## Source CSVs Scanned

- `outputs/logs/stage10_noise_0.csv`
- `outputs/logs/stage10_noise_1.csv`
- `outputs/logs/stage10_noise_3.csv`
- `outputs/logs/stage10_summary.csv`
- `outputs/logs/stage10_target_lost.csv`
- `outputs/logs/stage12_motion_estimation.csv`
- `outputs/logs/stage12b_prediction_grid.csv`
- `outputs/logs/stage13_baseline_fast.csv`
- `outputs/logs/stage13_baseline_medium.csv`
- `outputs/logs/stage13_predictive_tau150_fast.csv`
- `outputs/logs/stage13_predictive_tau150_medium.csv`
- `outputs/logs/stage13_predictive_tau200_fast.csv`
- `outputs/logs/stage13_predictive_tau200_medium.csv`
- `outputs/logs/stage13_predictive_tau300_fast.csv`
- `outputs/logs/stage13_predictive_tau300_medium.csv`
- `outputs/logs/stage13_summary.csv`
- `outputs/logs/stage14_rgbd_localization.csv`
- `outputs/logs/stage14_rgbd_summary.csv`
- `outputs/logs/stage15_multitarget.csv`
- `outputs/logs/stage15_summary.csv`
- `outputs/logs/stage16_summary.csv`
- `outputs/logs/stage16_target_switching.csv`
- `outputs/logs/stage17_obstacle_perception.csv`
- `outputs/logs/stage17_summary.csv`
- `outputs/logs/stage18_collision_checking.csv`
- `outputs/logs/stage18_summary.csv`
- `outputs/logs/stage19_rrt_connect.csv`
- `outputs/logs/stage19_summary.csv`
- `outputs/logs/stage20_demo_execution_path.csv`
- `outputs/logs/stage20_demo_planned_path.csv`
- `outputs/logs/stage20_execution.csv`
- `outputs/logs/stage20_summary.csv`
- `outputs/logs/stage7_summary.csv`
- `outputs/logs/stage7_trial_01.csv`
- `outputs/logs/stage7_trial_02.csv`
- `outputs/logs/stage7_trial_03.csv`
- `outputs/logs/stage7_trial_04.csv`
- `outputs/logs/stage7_trial_05.csv`
- `outputs/logs/stage9_fast.csv`
- `outputs/logs/stage9_medium.csv`
- `outputs/logs/stage9_slow.csv`
- `outputs/logs/stage9_summary.csv`
