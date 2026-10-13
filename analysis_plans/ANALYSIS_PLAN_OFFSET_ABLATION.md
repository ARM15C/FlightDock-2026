# Analysis Plan of Record — PTA Ablation — Version 5, 2026-09-22

22 September 2026. Amplifying Analyses. Descriptive; no hypothesis verdict is decided in this plan.

## 1. Question

What does the post-training augmentation (PTA), the 0.25 m downward vertical offset, contribute to each
controller's vertical clearance and capture success, and what does the learned correction contribute with the
PTA state held fixed? Within each controller only the PTA differs between its two arms; the residual bound,
curriculum and safety guards are held constant, so no individual effect is claimed for them. The ablation is a
diagnostic.

## 2. Arms and corpora

Each arm is one controller configuration flown on the same benchmark legs. The 0.25 m downward vertical
offset is the post-training augmentation (PTA); it is not a learned correction.

| Arm | Controller | Policy checkpoint (SHA-256) | PTA (0.25 m downward offset) |
|---|---|---|---|
| S1O | FlightDock-S1 with PTA | `34c7f1161a84c0665000dd3960232fddba6690483cb93632e365e7b45449fb1c` | yes |
| S1 | FlightDock-S1 | the same S1 checkpoint | no |
| A2O | FlightDock-A2 with PTA | `3d1cf6cac098ef3e53fd5c0ed759d14be58448b242297bde18b68f1f7801d442` | yes |
| A2 | FlightDock-A2 | the same A2 checkpoint | no |
| PIDO | PID-baseline with PTA | none (learned correction zero) | yes |
| PID | PID-baseline | none (learned correction zero) | no |

Every arm flew the same 600-leg benchmark once under light turbulence (σ = 0.25) and once under moderate
turbulence (σ = 0.35): nine wind stages at mean wind speeds of 0, 1, 2, 3, 4, 5, 6, 7 and 8 m/s from 180°,
60 legs each (30 seeds × straight and racetrack trajectories), and the Calm Control stage (0 m/s, turbulence
off, 60 legs). Seed block 10775–10804 in every arm. The (seed, stage, trajectory) keys are identical across
arms and turbulence levels, so every comparison is paired within seed. Calm Control enters no pool. Benchmark
seed-cell tables (SHA-256): σ = 0.25 `6be7b251698e4f776f336c4a6ac5522b021918640ff8fc0283d2288620fc3e18`;
σ = 0.35 `29c13944f7aaea07bc06bce7cf80b1b349787a613acbb880c4a56d217737e9ce`.

## 3. Outcome measures

Simulated docking-envelope capture is a three-second capture of the simulated M3b envelope. No physical docking
or contact is modeled. The approach runs from the Precontact Position (M2: 10 ± 2 m behind and 10 ± 2 m below
the DHC-2, within 2 m cross-track) to the M3b envelope, the box within 0.7 m behind, 0.7 m below and 0.5 m
cross-track of the boom-tip. The terminal approach corridor is every tick at which the QuadPlane is within
10.0 m straight-line range of the boom-tip.

A leg is a successful simulated docking-envelope capture when all four conditions hold:
1. the QuadPlane maintains position inside the M3b envelope for at least 3.0 s continuously, with the truth and
   perception channels both inside the envelope on the same timestamps;
2. its truth-channel peak upward deviation never exceeds 2.00 m while inside the terminal approach corridor;
3. it is never ahead of the DHC-2's CG while in formation (range ≤ 30 m, lateral offset ≤ 10 m);
4. no airframe contact occurs.

A leg that fails condition 2, 3 or 4 is a failed leg whether or not it met the M3b capture criterion.

Peak upward deviation, per leg: the largest truth-channel height of the QuadPlane above the boom-tip over the
ticks inside the terminal approach corridor. A leg that never enters the corridor has no corridor peak; it
leaves the H1 pool and stays in every capture rate as a failed leg. Geometry: boom-tip 0 m; H1 bar 2.00 m;
airframe-contact plane 3.42 m; reserve between them 1.42 m.

## 4. Statistics

- Every interval is a two-sided 95 % seed-cluster bootstrap: the 30 seed labels are resampled whole, with
  replacement, 10,000 times from one fixed random-number seed (20260915); the interval is the 2.5th and
  97.5th percentiles of the resampled statistic. Only leg-level values enter an interval.
- A percentile (p95) is computed by linear interpolation between order statistics.
- No hypothesis is tested in this plan; every interval describes precision in both directions. Where a
  two-sided bootstrap p-value is printed, it is twice the smaller of the two tail shares (resamples at or below
  zero, resamples at or above zero), capped at 1.
- No other interval method, test or correction is used.

## 5. Comparisons

Nine wind stages pooled (0–8 m/s), 540 legs per arm, at σ = 0.25 and again at σ = 0.35. Every difference is
paired within seed and carries the two-sided 95 % seed-cluster bootstrap interval of §4 and the two-sided
bootstrap p-value defined there. An effect is read as helped or hurt when the interval excludes zero.

- The table "PTA Ablation: Vertical Offset Impact by Controller": with PTA minus without, same controller: S1O minus S1, A2O minus A2,
  PIDO minus PID. Quantities: capture rate (% and pp) and corridor p95 of the per-leg peak upward deviation,
  truth channel (m). Each arm's pooled rate is also read against the historical crewed AAR docking success rate of 35 %.
- The table "PTA Ablation: Learned-Correction Impact by Controller": FlightDock arm minus the PID arm in the same PTA state: S1 and A2
  minus PID; S1O and A2O minus PIDO. Same quantities.

## 6. Outputs

Hypothesis testing: none. Amplifying: the tables "PTA Ablation: Vertical Offset Impact by Controller" and
"PTA Ablation: Learned-Correction Impact by Controller".

## 7. Signature

OPERATOR-SIGNATURE: PLAN-ABL-20260922 -- ARM -- I sign this ANALYSIS PLAN in full; plan sha e4b7c41309e49f8ca57d6a4cae1bd704c3c6f4679d30b0c00c4e945daa22783a
