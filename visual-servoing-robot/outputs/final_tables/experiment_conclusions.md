# Experimental Conclusions

## 1. Static Convergence

The five available Stage 7 raw trial logs each finish in `2D CENTERED PRECISE` with final component errors within ±2 px. The derived raw-log success count is 5/5, with convergence times from 5.067 s to 15.600 s (mean 9.707 s).

## 2. Dynamic Tracking

Stage 9 uses formal 20 s measurements after READY, so warm-up is excluded. The measured RMSE values are SLOW: 6.597 px, MEDIUM: 8.091 px, FAST: 9.912 px. In these three runs, error increases with the tested target-speed level, showing the current system has measurable dynamic tracking capability with larger lag at the faster condition.

## 3. Target Lost Recovery

The planned 2.5 s out-of-view interval produced one detected loss event. Reacquisition was 0.000 s and the recorded return to the precision condition took 0.133 s. The run completed without a return-to-home command.

## 4. Visual Noise Robustness

For the single fixed-seed tests at 0, 1, and 3 px injected centroid noise, all runs retained 100% detection and no target-lost events. Their RMSE values were 0 px: 8.091 px, 1 px: 7.913 px, 3 px: 7.165 px. The values are not monotonic, so this single-run result must not be interpreted as more noise improving tracking. It supports only that the system remained stable within this 0–3 px measurement-noise range.

## 5. Overall Conclusion

Within the recorded static, dynamic, target-lost, and visual-noise experiments, the Eye-in-Hand visual-servo system reached its tested static precision condition and maintained continuous RGB/OpenCV-based tracking. These conclusions describe only the tested PyBullet conditions and do not claim industrial performance or comparison with other algorithms.

## Data Quality Notes

- stage7_trial_03.csv: time_s contains 1 NaN values.
- stage7_trial_03.csv: 1 no-detection rows are retained; visible measurements are used for metrics.
- stage7_summary.csv is incomplete: it records trials [2], while raw CSV files provide trials [1, 2, 3, 4, 5]. The final Stage 7 table is derived from raw trial files.
- stage7_summary.csv Trial 2 disagrees with its current raw CSV on initial_error_norm, final_error_norm, convergence_time_s.
