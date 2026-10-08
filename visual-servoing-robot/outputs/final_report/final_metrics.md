# Final Experimental Metrics

All values are derived from immutable CSV evidence under `outputs/logs/`.

## 1. Visual Servo

| Stage | Condition | Metric | Value | Source |
|---|---|---|---:|---|
| Stage 7 | Trial 1 | Final ex | -2.000 px | `outputs/logs/stage7_trial_01.csv` |
| Stage 7 | Trial 1 | Final ey | -2.000 px | `outputs/logs/stage7_trial_01.csv` |
| Stage 7 | Trial 1 | Final error norm | 2.828 px | `outputs/logs/stage7_trial_01.csv` |
| Stage 7 | Trial 1 | Convergence time | 5.067 s | `outputs/logs/stage7_trial_01.csv` |
| Stage 7 | Trial 2 | Final ex | -1.000 px | `outputs/logs/stage7_trial_02.csv` |
| Stage 7 | Trial 2 | Final ey | -2.000 px | `outputs/logs/stage7_trial_02.csv` |
| Stage 7 | Trial 2 | Final error norm | 2.236 px | `outputs/logs/stage7_trial_02.csv` |
| Stage 7 | Trial 2 | Convergence time | 8.000 s | `outputs/logs/stage7_trial_02.csv` |
| Stage 7 | Trial 3 | Final ex | -2.000 px | `outputs/logs/stage7_trial_03.csv` |
| Stage 7 | Trial 3 | Final ey | 2.000 px | `outputs/logs/stage7_trial_03.csv` |
| Stage 7 | Trial 3 | Final error norm | 2.828 px | `outputs/logs/stage7_trial_03.csv` |
| Stage 7 | Trial 3 | Convergence time | 9.333 s | `outputs/logs/stage7_trial_03.csv` |
| Stage 7 | Trial 4 | Final ex | 0.000 px | `outputs/logs/stage7_trial_04.csv` |
| Stage 7 | Trial 4 | Final ey | 2.000 px | `outputs/logs/stage7_trial_04.csv` |
| Stage 7 | Trial 4 | Final error norm | 2.000 px | `outputs/logs/stage7_trial_04.csv` |
| Stage 7 | Trial 4 | Convergence time | 10.533 s | `outputs/logs/stage7_trial_04.csv` |
| Stage 7 | Trial 5 | Final ex | -2.000 px | `outputs/logs/stage7_trial_05.csv` |
| Stage 7 | Trial 5 | Final ey | 2.000 px | `outputs/logs/stage7_trial_05.csv` |
| Stage 7 | Trial 5 | Final error norm | 2.828 px | `outputs/logs/stage7_trial_05.csv` |
| Stage 7 | Trial 5 | Convergence time | 15.600 s | `outputs/logs/stage7_trial_05.csv` |
| Stage 7 | 5 static trials | Static convergence success rate | 100.00% | `outputs/logs/stage7_trial_01.csv; outputs/logs/stage7_trial_02.csv; outputs/logs/stage7_trial_03.csv; outputs/logs/stage7_trial_04.csv; outputs/logs/stage7_trial_05.csv` |
| Stage 7 | 5 static trials | Mean convergence time | 9.707 s | `outputs/logs/stage7_trial_01.csv; outputs/logs/stage7_trial_02.csv; outputs/logs/stage7_trial_03.csv; outputs/logs/stage7_trial_04.csv; outputs/logs/stage7_trial_05.csv` |
| Stage 7 | 5 static trials | Mean final absolute ex | 1.400 px | `outputs/logs/stage7_trial_01.csv; outputs/logs/stage7_trial_02.csv; outputs/logs/stage7_trial_03.csv; outputs/logs/stage7_trial_04.csv; outputs/logs/stage7_trial_05.csv` |
| Stage 7 | 5 static trials | Mean final absolute ey | 2.000 px | `outputs/logs/stage7_trial_01.csv; outputs/logs/stage7_trial_02.csv; outputs/logs/stage7_trial_03.csv; outputs/logs/stage7_trial_04.csv; outputs/logs/stage7_trial_05.csv` |
| Stage 9 | SLOW | Mean tracking error | 6.028 px | `outputs/logs/stage9_slow.csv` |
| Stage 9 | SLOW | Tracking RMSE | 6.597 px | `outputs/logs/stage9_slow.csv` |
| Stage 9 | SLOW | P95 tracking error | 10.198 px | `outputs/logs/stage9_slow.csv` |
| Stage 9 | SLOW | Maximum tracking error | 11.180 px | `outputs/logs/stage9_slow.csv` |
| Stage 9 | SLOW | Detection rate | 100.00% | `outputs/logs/stage9_slow.csv` |
| Stage 9 | MEDIUM | Mean tracking error | 7.501 px | `outputs/logs/stage9_medium.csv` |
| Stage 9 | MEDIUM | Tracking RMSE | 8.091 px | `outputs/logs/stage9_medium.csv` |
| Stage 9 | MEDIUM | P95 tracking error | 11.000 px | `outputs/logs/stage9_medium.csv` |
| Stage 9 | MEDIUM | Maximum tracking error | 11.180 px | `outputs/logs/stage9_medium.csv` |
| Stage 9 | MEDIUM | Detection rate | 100.00% | `outputs/logs/stage9_medium.csv` |
| Stage 9 | FAST | Mean tracking error | 9.047 px | `outputs/logs/stage9_fast.csv` |
| Stage 9 | FAST | Tracking RMSE | 9.912 px | `outputs/logs/stage9_fast.csv` |
| Stage 9 | FAST | P95 tracking error | 14.142 px | `outputs/logs/stage9_fast.csv` |
| Stage 9 | FAST | Maximum tracking error | 14.142 px | `outputs/logs/stage9_fast.csv` |
| Stage 9 | FAST | Detection rate | 100.00% | `outputs/logs/stage9_fast.csv` |
| Stage 10 | Target lost and recovery | Lost duration | 2.500 s | `outputs/logs/stage10_summary.csv` |
| Stage 10 | Target lost and recovery | Reacquisition time | 0.000 s | `outputs/logs/stage10_summary.csv` |
| Stage 10 | Target lost and recovery | Recovery time | 0.133 s | `outputs/logs/stage10_summary.csv` |
| Stage 10 | Target lost and recovery | Target lost event count | 1 | `outputs/logs/stage10_summary.csv` |
| Stage 10 | Noise 0 px | Mean error | 7.501 px | `outputs/logs/stage10_noise_0.csv` |
| Stage 10 | Noise 0 px | RMSE | 8.091 px | `outputs/logs/stage10_noise_0.csv` |
| Stage 10 | Noise 0 px | P95 error | 11.000 px | `outputs/logs/stage10_noise_0.csv` |
| Stage 10 | Noise 0 px | Maximum error | 11.180 px | `outputs/logs/stage10_noise_0.csv` |
| Stage 10 | Noise 1 px | Mean error | 7.300 px | `outputs/logs/stage10_noise_1.csv` |
| Stage 10 | Noise 1 px | RMSE | 7.913 px | `outputs/logs/stage10_noise_1.csv` |
| Stage 10 | Noise 1 px | P95 error | 11.045 px | `outputs/logs/stage10_noise_1.csv` |
| Stage 10 | Noise 1 px | Maximum error | 11.045 px | `outputs/logs/stage10_noise_1.csv` |
| Stage 10 | Noise 3 px | Mean error | 6.596 px | `outputs/logs/stage10_noise_3.csv` |
| Stage 10 | Noise 3 px | RMSE | 7.165 px | `outputs/logs/stage10_noise_3.csv` |
| Stage 10 | Noise 3 px | P95 error | 10.000 px | `outputs/logs/stage10_noise_3.csv` |
| Stage 10 | Noise 3 px | Maximum error | 11.000 px | `outputs/logs/stage10_noise_3.csv` |

## 2. Prediction

| Stage | Condition | Metric | Value | Source |
|---|---|---|---:|---|
| Stage 12 | Online estimator log | Baseline future mean error | 0.629 px | `outputs/logs/stage12_motion_estimation.csv` |
| Stage 12 | Online estimator log | Baseline future RMSE | 0.823 px | `outputs/logs/stage12_motion_estimation.csv` |
| Stage 12 | Online estimator log | EMA future mean error | 0.694 px | `outputs/logs/stage12_motion_estimation.csv` |
| Stage 12 | Online estimator log | EMA future RMSE | 0.899 px | `outputs/logs/stage12_motion_estimation.csv` |
| Stage 12 | Online estimator log | Mean-error improvement | -10.33% | `outputs/logs/stage12_motion_estimation.csv` |
| Stage 12B | alpha=0.2, tau=0.05 s | Lowest prediction RMSE | 0.626 px | `outputs/logs/stage12b_prediction_grid.csv` |
| Stage 12B | alpha=0.2, tau=0.3 s | Best mean-error improvement | 22.02% | `outputs/logs/stage12b_prediction_grid.csv` |
| Stage 13 | MEDIUM baseline | Mean raw tracking error | 7.501 px | `outputs/logs/stage13_baseline_medium.csv` |
| Stage 13 | MEDIUM baseline | Raw tracking RMSE | 8.091 px | `outputs/logs/stage13_baseline_medium.csv` |
| Stage 13 | MEDIUM baseline | P95 raw tracking error | 11.000 px | `outputs/logs/stage13_baseline_medium.csv` |
| Stage 13 | MEDIUM baseline | Maximum raw tracking error | 11.180 px | `outputs/logs/stage13_baseline_medium.csv` |
| Stage 13 | MEDIUM predictive tau=0.15 s | Mean raw tracking error | 7.081 px | `outputs/logs/stage13_predictive_tau150_medium.csv` |
| Stage 13 | MEDIUM predictive tau=0.15 s | Raw tracking RMSE | 7.615 px | `outputs/logs/stage13_predictive_tau150_medium.csv` |
| Stage 13 | MEDIUM predictive tau=0.15 s | P95 raw tracking error | 10.050 px | `outputs/logs/stage13_predictive_tau150_medium.csv` |
| Stage 13 | MEDIUM predictive tau=0.15 s | Maximum raw tracking error | 10.050 px | `outputs/logs/stage13_predictive_tau150_medium.csv` |
| Stage 13 | MEDIUM predictive tau=0.20 s | Mean raw tracking error | 7.013 px | `outputs/logs/stage13_predictive_tau200_medium.csv` |
| Stage 13 | MEDIUM predictive tau=0.20 s | Raw tracking RMSE | 7.553 px | `outputs/logs/stage13_predictive_tau200_medium.csv` |
| Stage 13 | MEDIUM predictive tau=0.20 s | P95 raw tracking error | 10.050 px | `outputs/logs/stage13_predictive_tau200_medium.csv` |
| Stage 13 | MEDIUM predictive tau=0.20 s | Maximum raw tracking error | 10.050 px | `outputs/logs/stage13_predictive_tau200_medium.csv` |
| Stage 13 | MEDIUM predictive tau=0.30 s | Mean raw tracking error | 6.803 px | `outputs/logs/stage13_predictive_tau300_medium.csv` |
| Stage 13 | MEDIUM predictive tau=0.30 s | Raw tracking RMSE | 7.333 px | `outputs/logs/stage13_predictive_tau300_medium.csv` |
| Stage 13 | MEDIUM predictive tau=0.30 s | P95 raw tracking error | 10.050 px | `outputs/logs/stage13_predictive_tau300_medium.csv` |
| Stage 13 | MEDIUM predictive tau=0.30 s | Maximum raw tracking error | 10.050 px | `outputs/logs/stage13_predictive_tau300_medium.csv` |
| Stage 13 | FAST baseline | Mean raw tracking error | 9.047 px | `outputs/logs/stage13_baseline_fast.csv` |
| Stage 13 | FAST baseline | Raw tracking RMSE | 9.912 px | `outputs/logs/stage13_baseline_fast.csv` |
| Stage 13 | FAST baseline | P95 raw tracking error | 14.142 px | `outputs/logs/stage13_baseline_fast.csv` |
| Stage 13 | FAST baseline | Maximum raw tracking error | 14.142 px | `outputs/logs/stage13_baseline_fast.csv` |
| Stage 13 | FAST predictive tau=0.15 s | Mean raw tracking error | 8.398 px | `outputs/logs/stage13_predictive_tau150_fast.csv` |
| Stage 13 | FAST predictive tau=0.15 s | Raw tracking RMSE | 9.102 px | `outputs/logs/stage13_predictive_tau150_fast.csv` |
| Stage 13 | FAST predictive tau=0.15 s | P95 raw tracking error | 12.042 px | `outputs/logs/stage13_predictive_tau150_fast.csv` |
| Stage 13 | FAST predictive tau=0.15 s | Maximum raw tracking error | 13.038 px | `outputs/logs/stage13_predictive_tau150_fast.csv` |
| Stage 13 | FAST predictive tau=0.20 s | Mean raw tracking error | 8.163 px | `outputs/logs/stage13_predictive_tau200_fast.csv` |
| Stage 13 | FAST predictive tau=0.20 s | Raw tracking RMSE | 8.836 px | `outputs/logs/stage13_predictive_tau200_fast.csv` |
| Stage 13 | FAST predictive tau=0.20 s | P95 raw tracking error | 12.042 px | `outputs/logs/stage13_predictive_tau200_fast.csv` |
| Stage 13 | FAST predictive tau=0.20 s | Maximum raw tracking error | 12.042 px | `outputs/logs/stage13_predictive_tau200_fast.csv` |
| Stage 13 | FAST predictive tau=0.30 s | Mean raw tracking error | 7.708 px | `outputs/logs/stage13_predictive_tau300_fast.csv` |
| Stage 13 | FAST predictive tau=0.30 s | Raw tracking RMSE | 8.337 px | `outputs/logs/stage13_predictive_tau300_fast.csv` |
| Stage 13 | FAST predictive tau=0.30 s | P95 raw tracking error | 11.045 px | `outputs/logs/stage13_predictive_tau300_fast.csv` |
| Stage 13 | FAST predictive tau=0.30 s | Maximum raw tracking error | 12.042 px | `outputs/logs/stage13_predictive_tau300_fast.csv` |
| Stage 13 | MEDIUM | RMSE improvement | 9.37% | `outputs/logs/stage13_summary.csv` |
| Stage 13 | MEDIUM | Mean error improvement | 9.31% | `outputs/logs/stage13_summary.csv` |
| Stage 13 | MEDIUM | P95 error improvement | 8.64% | `outputs/logs/stage13_summary.csv` |
| Stage 13 | FAST | RMSE improvement | 15.89% | `outputs/logs/stage13_summary.csv` |
| Stage 13 | FAST | Mean error improvement | 14.81% | `outputs/logs/stage13_summary.csv` |
| Stage 13 | FAST | P95 error improvement | 21.90% | `outputs/logs/stage13_summary.csv` |

## 3. RGB-D Perception

| Stage | Condition | Metric | Value | Source |
|---|---|---|---:|---|
| Stage 14 | center | 3D center error | 17.338 mm | `outputs/logs/stage14_rgbd_summary.csv` |
| Stage 14 | left | 3D center error | 10.080 mm | `outputs/logs/stage14_rgbd_summary.csv` |
| Stage 14 | right | 3D center error | 26.256 mm | `outputs/logs/stage14_rgbd_summary.csv` |
| Stage 14 | near | 3D center error | 14.540 mm | `outputs/logs/stage14_rgbd_summary.csv` |
| Stage 14 | far | 3D center error | 20.116 mm | `outputs/logs/stage14_rgbd_summary.csv` |
| Stage 14 | 5 static positions | Mean 3D error | 17.666 mm | `outputs/logs/stage14_rgbd_summary.csv` |
| Stage 14 | 5 static positions | RMSE 3D error | 18.481 mm | `outputs/logs/stage14_rgbd_summary.csv` |
| Stage 14 | 5 static positions | Median 3D error | 17.338 mm | `outputs/logs/stage14_rgbd_summary.csv` |
| Stage 14 | 5 static positions | Maximum 3D error | 26.256 mm | `outputs/logs/stage14_rgbd_summary.csv` |

## 4. Multi-Target

| Stage | Condition | Metric | Value | Source |
|---|---|---|---:|---|
| Stage 15 | RED | Detection rate | 100.00% | `outputs/logs/stage15_summary.csv` |
| Stage 15 | RED | Mean 3D localization error | 4.997 mm | `outputs/logs/stage15_summary.csv` |
| Stage 15 | RED | RMSE 3D localization error | 5.462 mm | `outputs/logs/stage15_summary.csv` |
| Stage 15 | RED | Maximum 3D localization error | 9.211 mm | `outputs/logs/stage15_summary.csv` |
| Stage 15 | GREEN | Detection rate | 100.00% | `outputs/logs/stage15_summary.csv` |
| Stage 15 | GREEN | Mean 3D localization error | 35.797 mm | `outputs/logs/stage15_summary.csv` |
| Stage 15 | GREEN | RMSE 3D localization error | 38.549 mm | `outputs/logs/stage15_summary.csv` |
| Stage 15 | GREEN | Maximum 3D localization error | 63.702 mm | `outputs/logs/stage15_summary.csv` |
| Stage 15 | BLUE | Detection rate | 100.00% | `outputs/logs/stage15_summary.csv` |
| Stage 15 | BLUE | Mean 3D localization error | 31.279 mm | `outputs/logs/stage15_summary.csv` |
| Stage 15 | BLUE | RMSE 3D localization error | 32.871 mm | `outputs/logs/stage15_summary.csv` |
| Stage 15 | BLUE | Maximum 3D localization error | 50.715 mm | `outputs/logs/stage15_summary.csv` |
| Stage 15 | RED+GREEN+BLUE | All-target detection rate | 100.00% | `outputs/logs/stage15_summary.csv` |
| Stage 15 | RED+GREEN+BLUE | Color confusion count | 0 | `outputs/logs/stage15_summary.csv` |
| Stage 16 | Historical formal switching log | Switch trial count | 6 | `outputs/logs/stage16_summary.csv` |
| Stage 16 | Historical formal switching log | Switch success rate | 50.00% | `outputs/logs/stage16_summary.csv` |
| Stage 16 | Historical formal switching log | Mean successful switch response time | 6.299 s | `outputs/logs/stage16_summary.csv` |
| Stage 16 | Historical formal switching log | Maximum successful switch response time | 6.996 s | `outputs/logs/stage16_summary.csv` |
| Stage 16 | Historical formal switching log | Target lost event count | 0 | `outputs/logs/stage16_summary.csv` |
| Stage 16 | Historical formal switching log | Locked-joint maximum deviation | 0.020 rad | `outputs/logs/stage16_summary.csv` |

## 5. Obstacle Perception

| Stage | Condition | Metric | Value | Source |
|---|---|---|---:|---|
| Stage 17 | 5 obstacle scenes | Obstacle detection rate | 100.00% | `outputs/logs/stage17_summary.csv` |
| Stage 17 | 5 obstacle scenes | Mean center error | 7.998 mm | `outputs/logs/stage17_summary.csv` |
| Stage 17 | 5 obstacle scenes | RMSE center error | 8.927 mm | `outputs/logs/stage17_summary.csv` |
| Stage 17 | 5 obstacle scenes | Maximum center error | 14.888 mm | `outputs/logs/stage17_summary.csv` |
| Stage 17 | 5 obstacle scenes | Mean AABB IoU | 40.82% | `outputs/logs/stage17_summary.csv` |
| Stage 17 | 5 obstacle scenes | Mean GT coverage | 100.00% | `outputs/logs/stage17_summary.csv` |
| Stage 17 | 5 obstacle scenes | Estimated-to-GT volume ratio | 2.450× | `outputs/logs/stage17_summary.csv` |
| Stage 17 | 5 obstacle scenes | Target color confusion count | 0 | `outputs/logs/stage17_summary.csv` |

## 6. Collision Safety

| Stage | Condition | Metric | Value | Source |
|---|---|---|---:|---|
| Stage 18 | 20 candidate paths | TP | 7 | `outputs/logs/stage18_summary.csv` |
| Stage 18 | 20 candidate paths | TN | 7 | `outputs/logs/stage18_summary.csv` |
| Stage 18 | 20 candidate paths | FP | 6 | `outputs/logs/stage18_summary.csv` |
| Stage 18 | 20 candidate paths | FN | 0 | `outputs/logs/stage18_summary.csv` |
| Stage 18 | 20 candidate paths | Precision | 53.85% | `outputs/logs/stage18_summary.csv` |
| Stage 18 | 20 candidate paths | Recall | 100.00% | `outputs/logs/stage18_summary.csv` |
| Stage 18 | 20 candidate paths | Safety recall | 100.00% | `outputs/logs/stage18_summary.csv` |

## 7. Planning

| Stage | Condition | Metric | Value | Source |
|---|---|---|---:|---|
| Stage 19 | Tested scenarios and seeds | Scenario count | 10 | `outputs/logs/stage19_summary.csv` |
| Stage 19 | Tested scenarios and seeds | RRT trial count | 25 | `outputs/logs/stage19_summary.csv` |
| Stage 19 | Tested scenarios and seeds | RRT planning success rate | 100.00% | `outputs/logs/stage19_summary.csv` |
| Stage 19 | Tested scenarios and seeds | Mean planning time | 0.119 s | `outputs/logs/stage19_summary.csv` |
| Stage 19 | Tested scenarios and seeds | Median planning time | 0.096 s | `outputs/logs/stage19_summary.csv` |
| Stage 19 | Tested scenarios and seeds | Maximum planning time | 0.490 s | `outputs/logs/stage19_summary.csv` |
| Stage 19 | Tested scenarios and seeds | Mean RRT path length | 2.459 rad | `outputs/logs/stage19_summary.csv` |
| Stage 19 | Tested scenarios and seeds | Final GT path collision count | 0 | `outputs/logs/stage19_summary.csv` |

## 8. Execution

| Stage | Condition | Metric | Value | Source |
|---|---|---|---:|---|
| Stage 20 | 2 direct + 3 RRT | Execution trial count | 5 | `outputs/logs/stage20_summary.csv` |
| Stage 20 | 2 direct + 3 RRT | Execution success rate | 100.00% | `outputs/logs/stage20_summary.csv` |
| Stage 20 | DIRECT | Mean execution time | 2.767 s | `outputs/logs/stage20_summary.csv` |
| Stage 20 | RRT_CONNECT | Mean execution time | 18.637 s | `outputs/logs/stage20_summary.csv` |
| Stage 20 | 5 formal trials | Mean goal joint error | 0.001965 rad | `outputs/logs/stage20_summary.csv` |
| Stage 20 | 5 formal trials | Maximum goal joint error | 0.001979 rad | `outputs/logs/stage20_summary.csv` |
| Stage 20 | All raw trajectory samples | Mean joint tracking error | 0.015 rad | `outputs/logs/stage20_execution.csv` |
| Stage 20 | All raw trajectory samples | RMSE joint tracking error | 0.020 rad | `outputs/logs/stage20_execution.csv` |
| Stage 20 | All raw trajectory samples | Maximum joint tracking error | 0.050 rad | `outputs/logs/stage20_execution.csv` |
| Stage 20 | 5 formal trials | Locked-joint maximum deviation | 0.000063 rad | `outputs/logs/stage20_summary.csv` |
| Stage 20 | 5 formal trials | GT collision count | 0 | `outputs/logs/stage20_summary.csv` |
