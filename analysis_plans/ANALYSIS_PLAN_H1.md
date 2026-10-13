# Analysis Plan of Record — H1 (Vertical Deconfliction) — Version 5, 2026-09-22

22 September 2026. Hypothesis Testing.

## 1. Hypothesis

> H1: Under light (σ = 0.25) turbulence across wind speeds (0–8 m/s), FlightDock's 95th percentile (p95) peak
> upward deviation stays below 2.00 m, preserving required vertical deconfliction (one-sided bootstrap test,
> α = 0.025).

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
- The hypothesis is tested one-sided at α = 0.025 through its two-sided 95 % interval: it is met when the
  whole interval lies on the required side of its bar. The point estimate alone decides nothing.
- The one-sided bootstrap p-value is the share of the 10,000 resamples whose statistic falls on the null side of
  the bar (ties counted to the null). It is reported beside the point estimate and the interval.
- No other interval method, test or correction is used.

## 5. Decision rule

- Statistic: the p95 of the per-leg peak upward deviation, truth channel, inside the terminal approach corridor.
- Pool: the nine wind stages at σ = 0.25; the legs that entered the corridor (n printed with the result).
- Met when: the upper bound of the two-sided 95 % seed-cluster bootstrap interval is < 2.00 m.
- Verdicts: one each for S1O, S1, A2O and A2. PIDO and PID are read under the same rule as comparators and are
  not graded.
- Reported beside the verdict: legs in the corridor; the share and count of corridor legs over 2.00 m; the
  clearance below the airframe-contact plane at the p95 (3.42 m minus the p95) and at the worst corridor peak;
  the one-sided bootstrap p-value (share of resamples whose p95 fell at or above 2.00 m).

## 6. Outputs

- Hypothesis testing: the table and the figure titled "H1: p95 Peak Upward Deviation, Light Turbulence".
- Amplifying, decides nothing: the table "Clearance and Guard Record by Wind Stage, σ = 0.25"; the table
  "Truth vs. Perception: H1 Statistic by Arm" and the figure "Per-Leg Peak Upward Deviation, Truth and
  Perception"; the figure "H1 Statistic by Trajectory"; the table "Vertical Clearance at σ = 0.35" (not
  graded).

## 7. Signature

OPERATOR-SIGNATURE: PLAN-H1-20260922 -- ARM -- I sign this ANALYSIS PLAN in full; plan sha b226d9033bc40685dd3338e630592bb2490db6be18761931519bc258f23f9573
