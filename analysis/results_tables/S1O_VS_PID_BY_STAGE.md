# S1O against the PID-baseline by stage, both turbulence levels (amplifying; decides no hypothesis)

| σ | stage | S1O captures | PID-baseline captures | same-seed difference, S1O minus PID-baseline [95% interval] (pp) | interval excludes zero |
|---|---|---|---|---|---|
| 0.25 | Calm Control | 38 of 60 | 57 of 60 | -31.7 [-45.0, -18.3] | yes |
| 0.25 | 0 m/s | 38 of 60 | 35 of 60 | +5.0 [-13.3, +23.3] | no |
| 0.25 | 1 m/s | 34 of 60 | 27 of 60 | +11.7 [-6.7, +31.7] | no |
| 0.25 | 2 m/s | 38 of 60 | 37 of 60 | +1.7 [-16.7, +20.0] | no |
| 0.25 | 3 m/s | 41 of 60 | 35 of 60 | +10.0 [-6.7, +26.7] | no |
| 0.25 | 4 m/s | 47 of 60 | 34 of 60 | +21.7 [+5.0, +36.7] | yes |
| 0.25 | 5 m/s | 36 of 60 | 33 of 60 | +5.0 [-11.7, +21.7] | no |
| 0.25 | 6 m/s | 44 of 60 | 36 of 60 | +13.3 [-5.0, +31.7] | no |
| 0.25 | 7 m/s | 42 of 60 | 36 of 60 | +10.0 [-5.0, +25.0] | no |
| 0.25 | 8 m/s | 43 of 60 | 39 of 60 | +6.7 [-10.0, +23.3] | no |
| 0.25 | Nine wind stages pooled | 363 of 540 | 312 of 540 | +9.4 [+4.3, +14.6] | yes |
| 0.35 | Calm Control | 47 of 60 | 60 of 60 | -21.7 [-33.3, -10.0] | yes |
| 0.35 | 0 m/s | 30 of 60 | 11 of 60 | +31.7 [+13.3, +48.3] | yes |
| 0.35 | 1 m/s | 15 of 60 | 8 of 60 | +11.7 [-1.7, +26.7] | no |
| 0.35 | 2 m/s | 26 of 60 | 14 of 60 | +20.0 [+5.0, +35.0] | yes |
| 0.35 | 3 m/s | 24 of 60 | 18 of 60 | +10.0 [-6.7, +26.7] | no |
| 0.35 | 4 m/s | 27 of 60 | 7 of 60 | +33.3 [+20.0, +45.0] | yes |
| 0.35 | 5 m/s | 24 of 60 | 13 of 60 | +18.3 [+3.3, +33.3] | yes |
| 0.35 | 6 m/s | 22 of 60 | 13 of 60 | +15.0 [-5.0, +35.0] | no |
| 0.35 | 7 m/s | 24 of 60 | 14 of 60 | +16.7 [+5.0, +28.3] | yes |
| 0.35 | 8 m/s | 25 of 60 | 15 of 60 | +16.7 [-1.7, +33.3] | no |
| 0.35 | Nine wind stages pooled | 217 of 540 | 113 of 540 | +19.3 [+13.3, +24.8] | yes |

Note. 60 same-seed pairs per stage (30 seeds, two trajectories), 540 in the pool. Difference: S1O captures minus PID-baseline captures on the same (stage, seed, trajectory) legs. Interval: two-sided 95% seed-cluster bootstrap (30 seeds resampled whole, 10,000 draws, fixed seed). A 60-pair stage carries a wide interval; the pooled row at σ = 0.25 is the H2 statistic and the only graded row. The σ = 0.35 rows are not graded. Calm Control (no wind, no turbulence) enters no pool.

Summary: {"0.25|summary": {"stages_positive": 9, "stages_interval_above_zero": 1, "min_d": 1.6666666666666667, "max_d": 21.666666666666668}, "0.35|summary": {"stages_positive": 9, "stages_interval_above_zero": 5, "min_d": 10.0, "max_d": 33.333333333333336}}