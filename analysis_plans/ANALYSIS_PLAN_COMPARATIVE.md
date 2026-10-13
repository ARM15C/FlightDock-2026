# Analysis Plan of Record — Amplifying Analyses — Version 5, 2026-09-22

22 September 2026. Amplifying Analyses. Every result below describes safety, recovery, disturbance, corpus
integrity or arm-against-arm behavior beyond the three hypothesis tests. Nothing in this plan decides a
hypothesis.

## 1. Arms and corpora

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

## 2. Outcome measures

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

## 3. Statistics

- Every interval is a two-sided 95 % seed-cluster bootstrap: the 30 seed labels are resampled whole, with
  replacement, 10,000 times from one fixed random-number seed (20260915); the interval is the 2.5th and
  97.5th percentiles of the resampled statistic. Only leg-level values enter an interval.
- A percentile (p95) is computed by linear interpolation between order statistics.
- No hypothesis is tested in this plan; every interval describes precision in both directions. Where a
  two-sided bootstrap p-value is printed, it is twice the smaller of the two tail shares (resamples at or below
  zero, resamples at or above zero), capped at 1.
- No other interval method, test or correction is used.

Unless a line below says otherwise: nine wind stages pooled (0–8 m/s), 540 legs per arm and turbulence level,
differences paired within seed, and every rate or difference carries the two-sided 95 % seed-cluster bootstrap
interval of §3.

## 4. Analyses (one line each: what is computed, how)

Corpus and disturbance
- The table "Benchmark Corpus Accounting: Six Arms, Two Turbulence Levels": clean, fault-flagged and attrited legs per arm and turbulence level;
  a stage is INCONCLUSIVE when more than 9 of its 60 legs are lost; legs whose realized gust intensity fell
  outside the band on one axis are counted and kept in every denominator.
- The table "Commanded vs. Realized Gusts": per-leg gust standard deviation on the north, east and down axes
  from the simulator's gust record; median per turbulence level; band ±0.08 m/s at σ 0.25 and ±0.11 m/s at
  σ 0.35; legs outside the band counted.
- The table "Identical Gust-Stream Pair Analysis": the arm-minus-PID capture difference on same-seed pairs,
  split into pairs that flew the same gust stream and pairs where one arm was re-rolled by a warm-up retry.
- The table "Calm Control Repeatability": the 60 Calm Control legs flown on each turbulence ladder; captures per
  flight, flipped, lost and gained legs, and the difference.

Why legs failed, and safety record
- The table "Why Legs Were Unsuccessful": one reason per leg, assigned in column order: never within 10 m of
  the boom-tip; past the DHC-2 CG; over 2.00 m inside the corridor; no M3b entry on both channels; M3b dwell
  under 3.0 s. Each as legs, share of the 540 and interval, beside the unsuccessful total.
- The tables "Guard Record and Breakaway Cost, Light Turbulence", "Guard Record and Breakaway Cost, Moderate
  Turbulence" and "Guard Record and Breakaway Cost", with the per-stage tables "Clearance and Guard Record by
  Wind Stage, σ = 0.25" and "Clearance and Guard Record by Wind Stage, σ = 0.35": altitude-guard and
  overtake-guard fires and breakaways inside the corridor; legs with at least one; false fires (no truth
  crossing of the guard's line within 1.0 s) and missed crossings (guard armed, no fire within 1.0 s); the
  capture-rate difference between legs with and without a breakaway, with its two-sided bootstrap p-value; the
  median truth height at the altitude-guard fire.
- The table "Breakaway Recovery and Stopping Distance": breakaways that exited under 2.00 m and at or beyond
  15.0 m behind the boom-tip; peak truth height in altitude-guard breakaways; farthest distance ahead of the
  boom-tip on eligible corridor runs and runs past the DHC-2 CG line; legs over the 3.42 m airframe-contact
  plane inside the corridor.
- The table and the figure titled "Approach Funnel by Arm": share of each arm's legs that reached each step at least once
  (within 10 m, 3 m and 1.5 m of the boom-tip; inside M3b; inside for 1 s and 3 s cumulative; capture).

Channels and position
- The table "Truth vs. Perception: H1 Statistic by Arm" and the figure "Per-Leg Peak Upward Deviation, Truth
  and Perception": the corridor p95 on each
  channel, legs over 2.00 m on each, and the perception-minus-truth difference; H1 is graded on truth.
- The table "Truth vs. Perception: Corridor Error by Axis": per-tick perception-minus-truth error inside the
  corridor; no interval.
- The table "Held Position and Applied Correction in the M3b Envelope": medians over every tick inside M3b of
  the held position and the applied correction; no interval.
- The table "Vertical Clearance at σ = 0.35": the corridor p95 and legs over 2.00 m under moderate turbulence,
  truth channel; not graded.

Rates by stage, trajectory and reference
- The figure "Captures by Wind Stage and Arm" and the table "S1O vs. PID by Wind Stage": per-stage capture
  rates (60 legs or pairs) and same-seed differences; only the pooled σ 0.25 row is graded, under the H2 plan.
- The table "Captures by Trajectory" and the figures "H1 Statistic by Trajectory", "H2 Statistic by
  Trajectory" and "H3 Statistic by Trajectory": rates and the H1, H2 and H3 statistics on the
  straight and racetrack legs separately (270 legs or pairs per arm), each arm's straight-minus-racetrack
  difference and that difference relative to PID's; no trajectory panel is graded.
- The table "Simulated Docking-Envelope Capture Success Rate vs. 35% Reference Floor": each arm's pooled rate
  read against the historical crewed AAR docking success rate of 35 %; whole interval above, below or
  straddling; the share of resamples at or below 35 %. A reference, not a hypothesis.

## 5. Outputs

Hypothesis testing: none. Amplifying: the tables and figures named in §4, each carrying the pool, the
statistic and the interval method stated above in its note.

## 6. Signature

OPERATOR-SIGNATURE: PLAN-CMP-20260922 -- ARM -- I sign this ANALYSIS PLAN in full; plan sha 511a8e933e90ad6a3f387d2fe1456a2d653f5c9b652f97884ef40522565551f9
