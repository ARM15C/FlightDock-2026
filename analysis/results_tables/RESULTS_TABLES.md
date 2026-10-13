# Results tables (built 2026-10-12 by results_tables.py)

Every cell is computed from the shipped board, readouts and derived files; nothing is typed. Arm order: S1O, S1, A2O, A2, PIDO, PID (each controller with the 0.25 m offset first, then without; the PID-baseline last).

### T1. Benchmark corpus accounting, six arms, two turbulence levels (600 legs each; 10 stages × 30 seeds × 2 trajectories)

| σ | arm | clean legs of 600 | fault-flagged, kept | worst-stage attrition | σ assertion failed on one axis, legs kept |
|---|---|---|---|---|---|
| 0.25 | S1O | 600 | 0 | 0 of 60 (bar: more than 9) | 23 |
| 0.25 | S1 | 600 | 0 | 0 of 60 (bar: more than 9) | 22 |
| 0.25 | A2O | 600 | 0 | 0 of 60 (bar: more than 9) | 23 |
| 0.25 | A2 | 600 | 0 | 0 of 60 (bar: more than 9) | 22 |
| 0.25 | PIDO | 599 | 1 | 0 of 60 (bar: more than 9) | 23 |
| 0.25 | PID | 600 | 0 | 0 of 60 (bar: more than 9) | 24 |
| 0.35 | S1O | 600 | 0 | 0 of 60 (bar: more than 9) | 24 |
| 0.35 | S1 | 600 | 0 | 0 of 60 (bar: more than 9) | 25 |
| 0.35 | A2O | 600 | 0 | 0 of 60 (bar: more than 9) | 24 |
| 0.35 | A2 | 599 | 1 | 0 of 60 (bar: more than 9) | 29 |
| 0.35 | PIDO | 600 | 0 | 0 of 60 (bar: more than 9) | 22 |
| 0.35 | PID | 600 | 0 | 0 of 60 (bar: more than 9) | 25 |

Note. Every arm flew 600 of 600 legs on the same 30 seeds (block 10775-10804); every leg was gradable and none was quarantined. Attrition bar: the document, a stage is INCONCLUSIVE when more than 15% of its 60 legs (more than nine) fail. No stage lost a leg. The σ assertion checks the realized gust intensity on each axis against the commanded value; legs that failed it on one axis stay in every denominator.

### T2. Realized turbulence intensity by arm and commanded σ, nine wind stages (per-leg values from the σ gate; band ±0.08 m/s at σ 0.25 and ±0.11 m/s at σ 0.35 (the gate's own band, 0.32 σ))

| σ commanded | arm | legs with a measured σ | realized σ, median, north / east / down (m/s) | legs outside the band on one axis |
|---|---|---|---|---|
| 0.25 | S1O | 540 | 0.241 / 0.238 / 0.235 | 23 |
| 0.25 | S1 | 540 | 0.241 / 0.238 / 0.235 | 22 |
| 0.25 | A2O | 540 | 0.241 / 0.238 / 0.235 | 23 |
| 0.25 | A2 | 540 | 0.241 / 0.238 / 0.234 | 22 |
| 0.25 | PIDO | 540 | 0.241 / 0.238 / 0.235 | 23 |
| 0.25 | PID | 540 | 0.241 / 0.238 / 0.234 | 24 |
| 0.35 | S1O | 540 | 0.337 / 0.333 / 0.328 | 24 |
| 0.35 | S1 | 540 | 0.336 / 0.333 / 0.328 | 25 |
| 0.35 | A2O | 540 | 0.337 / 0.333 / 0.328 | 24 |
| 0.35 | A2 | 540 | 0.336 / 0.333 / 0.327 | 29 |
| 0.35 | PIDO | 540 | 0.337 / 0.333 / 0.328 | 22 |
| 0.35 | PID | 540 | 0.337 / 0.333 / 0.328 | 25 |

Note. Each leg's σ gate records the realized gust standard deviation on the three axes from the FDM's gust record. The commanded mean wind matched the manifest on every leg (wind_check column).

### T3. H1 statistic by arm: corridor p95 of the peak upward deviation above the boom-tip, truth channel, σ = 0.25, nine wind stages pooled, with the clearance that remained to the 3.42 m contact plane

| arm | configuration | legs in the corridor (n) | 95th-percentile peak height above the boom-tip (m) | legs over 2.00 m (% of n) | 95% interval (m) | clearance below the contact plane at the p95 (m) | clearance below the contact plane, worst leg (m) | p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| S1O | S1 with the 0.25 m offset | 513 | 1.904 | 17 (3.3%) | [1.800, 1.990] | 1.52 | 0.88 | 0.0096 | H1 met |
| S1 | S1, no offset | 521 | 2.110 | 52 (10.0%) | [2.064, 2.146] | 1.31 | 0.47 | 1.0000 | H1 not met |
| A2O | A2 with the 0.25 m offset | 525 | 1.690 | 7 (1.3%) | [1.550, 1.780] | 1.73 | 1.17 | 0.0000 | H1 met |
| A2 | A2, no offset | 522 | 1.690 | 9 (1.7%) | [1.600, 1.800] | 1.73 | 1.04 | 0.0000 | H1 met |
| PIDO | PID-baseline with the 0.25 m offset | 517 | 2.150 | 98 (19.0%) | [2.130, 2.210] | 1.27 | 0.49 | 1.0000 | comparator (not graded) |
| PID | PID-baseline, no offset | 517 | 2.344 | 171 (33.1%) | [2.265, 2.410] | 1.08 | 0.66 | 1.0000 | comparator (not graded) |

Note. H1 pool = legs that entered the terminal approach corridor (within 10 m of the boom-tip); the legs-over-2.00 m share is of that pool. Bar: upper bound of the interval below 2.00 m (the table “Hypothesis Tests”). The contact plane is where the QuadPlane's top would touch the DHC-2's belly, 3.42 m above the boom-tip; clearance below it = 3.42 m minus the p95, and 3.42 m minus the highest corridor peak among the arm's legs; both include the 1.42 m the guard keeps in reserve. Interval: two-sided 95% seed-cluster bootstrap (30 seeds resampled whole, 10,000 draws, fixed seed). The one-sided bootstrap p is the share of the 10,000 whole-seed resamples whose p95 fell at or above the bar.

### T4. Perception beside truth: the H1 statistic and the legs over the 2.00 m bar on both channels, σ = 0.25, nine wind stages pooled (perception not graded)

| arm | legs in the corridor (n) | truth: 95th-percentile peak height (m) [95% interval] | perception: 95th-percentile peak height (m) [95% interval] | perception minus truth at the p95 (m) | legs over 2.00 m, truth | legs over 2.00 m, perception | verdict (truth) |
|---|---|---|---|---|---|---|---|
| S1O | 513 | 1.904 [1.800, 1.990] | 2.239 [2.175, 2.295] | +0.335 | 17 | 56 | H1 met |
| S1 | 521 | 2.110 [2.064, 2.146] | 2.467 [2.403, 2.569] | +0.357 | 52 | 117 | H1 not met |
| A2O | 525 | 1.690 [1.550, 1.780] | 1.948 [1.896, 2.063] | +0.258 | 7 | 25 | H1 met |
| A2 | 522 | 1.690 [1.600, 1.800] | 2.036 [1.954, 2.249] | +0.347 | 9 | 33 | H1 met |
| PIDO | 517 | 2.150 [2.130, 2.210] | 2.570 [2.513, 2.701] | +0.420 | 98 | 213 | comparator (not graded) |
| PID | 517 | 2.344 [2.265, 2.410] | 2.755 [2.672, 2.807] | +0.411 | 171 | 311 | comparator (not graded) |

Note. Truth = the simulation state; perception = what the controller and the altitude guard read (the GPS-direct height). H1 is tested on truth. Both p95s carry the two-sided 95% seed-cluster bootstrap (30 seeds resampled whole, 10,000 draws, fixed seed). The verdict column is the truth verdict of the H1 table.

### T5. H2 statistic by arm: same-seed difference in docking-envelope captures against the PID-baseline, σ = 0.25

| arm | captures of 540, arm | captures of 540, PID-baseline | difference on the same flights (pp) | 95% interval (pp) | p | verdict |
|---|---|---|---|---|---|---|
| S1O | 363 (67.2%) | 312 (57.8%) | +9.44 | [+4.26, +14.63] | 0.0004 | H2 met |
| S1 | 376 (69.6%) | 312 (57.8%) | +11.85 | [+6.48, +17.59] | 0.0001 | H2 met |
| A2O | 300 (55.6%) | 312 (57.8%) | -2.22 | [-7.96, +3.33] | 0.7903 | H2 not met |
| A2 | 334 (61.9%) | 312 (57.8%) | +4.07 | [-1.85, +10.19] | 0.0981 | H2 not met |
| PIDO | 365 (67.6%) | 312 (57.8%) | +9.81 | [+4.07, +15.74] | 0.0004 | comparator (not graded) |

Note. A pair is one (stage, seed, trajectory) flown by both controllers. Bar: lower bound of the nine-stage interval above 0 pp (the table “Hypothesis Tests”). PIDO is a comparator: reported with the same statistic, not graded. A pair is the same stage, seed and trajectory flown by both controllers. Interval: two-sided 95% seed-cluster bootstrap (30 seeds resampled whole, 10,000 draws, fixed seed); p: one-sided, the share of the 10,000 resamples whose difference fell at or below 0 pp.

### T6. Pooled docking-envelope capture rate by arm and turbulence level against the 35% minimum capture rate of reference (reported; decides no hypothesis)

| σ | arm | captures of 540 | capture rate [95% interval] (%) | p (rate at or below 35%) | against the 35% floor |
|---|---|---|---|---|---|
| 0.25 | S1O | 363 | 67.2 [63.5, 71.1] | 0.0000 | above |
| 0.25 | S1 | 376 | 69.6 [65.6, 73.5] | 0.0000 | above |
| 0.25 | A2O | 300 | 55.6 [51.1, 59.4] | 0.0000 | above |
| 0.25 | A2 | 334 | 61.9 [57.4, 66.1] | 0.0000 | above |
| 0.25 | PIDO | 365 | 67.6 [63.3, 71.7] | 0.0000 | above |
| 0.25 | PID | 312 | 57.8 [53.3, 61.9] | 0.0000 | above |
| 0.35 | S1O | 217 | 40.2 [36.3, 44.3] | 0.0045 | above |
| 0.35 | S1 | 168 | 31.1 [27.2, 35.0] | 0.9770 | straddles |
| 0.35 | A2O | 207 | 38.3 [33.7, 43.0] | 0.0841 | straddles |
| 0.35 | A2 | 216 | 40.0 [36.5, 43.3] | 0.0026 | above |
| 0.35 | PIDO | 146 | 27.0 [23.3, 30.7] | 1.0000 | below |
| 0.35 | PID | 113 | 20.9 [17.4, 24.6] | 1.0000 | below |

Note. Nine wind stages pooled, 540 legs per arm and σ, seed block 10775-10804. The reference floor is the crewed aerial-refueling docking success rate of about 35% reported by Ren and Quan (2024) (the document). Intervals: two-sided 95% seed-cluster bootstrap (30 seeds resampled whole, 10,000 draws). "Against the 35% floor" reads the whole interval: above = the whole interval above 35%, below = the whole interval below, straddles = the interval contains 35%; the floor is a reference, not a hypothesis bar. The p is the one-sided bootstrap p of the H1 to H3 tables applied to the floor: the share of the 10,000 whole-seed resamples whose pooled rate fell at or below 35%. It says how firmly an arm cleared the reference, nothing more; the floor decides no hypothesis, so no α applies and no verdict follows from it.

### T7. H3 statistic by arm: same-seed decrease in docking-envelope captures from σ = 0.25 to σ = 0.35, same controller

| arm | captures of 540, σ 0.25 | captures of 540, σ 0.35 | decrease on the same flights (pp) | 95% interval (pp) | p | verdict | σ 0.35 rate against the 35% floor |
|---|---|---|---|---|---|---|---|
| S1O | 363 (67.2%) | 217 (40.2%) | 27.04 | [21.48, 32.59] | 0.0038 | H3 met | above |
| S1 | 376 (69.6%) | 168 (31.1%) | 38.52 | [32.22, 44.63] | 0.8654 | H3 not met | straddles |
| A2O | 300 (55.6%) | 207 (38.3%) | 17.22 | [11.30, 22.96] | 0.0000 | H3 met | straddles |
| A2 | 334 (61.9%) | 216 (40.0%) | 21.85 | [16.67, 26.85] | 0.0000 | H3 met | above |
| PIDO | 365 (67.6%) | 146 (27.0%) | 40.56 | [36.30, 44.63] | 0.9949 | comparator (not graded) | below |
| PID | 312 (57.8%) | 113 (20.9%) | 36.85 | [31.11, 42.41] | 0.7455 | comparator (not graded) | below |

Note. Bar: upper bound of the nine-stage interval below 35 pp (the table “Hypothesis Tests”). The PID-baseline and PIDO are comparators: reported with the same statistic, not graded. Last column: the pooled σ 0.35 capture rate of the reference-floor table read against the 35% minimum capture rate of reference (Ren and Quan, 2024): above = the whole interval above 35%, below = the whole interval below, straddles = the interval contains 35%; a reference, not a verdict. Interval: two-sided 95% seed-cluster bootstrap (30 seeds resampled whole, 10,000 draws, fixed seed); p: one-sided, the share of the 10,000 resamples whose decrease fell at or above 35 pp.

### T8. Vertical clearance at σ = 0.35, nine wind stages (not graded for H1)

| arm | legs that entered the corridor | 95th-percentile peak height above the boom-tip (m) | 95% interval (m) | legs over 2.00 m | wind stages with p95 over 2.00 m (of 9) | legs over the 3.42 m plane in the corridor |
|---|---|---|---|---|---|---|
| S1O | 480 | 2.281 | [2.218, 2.344] | 126 | 9 | 0 |
| S1 | 482 | 2.440 | [2.350, 2.510] | 179 | 9 | 0 |
| A2O | 493 | 2.254 | [2.170, 2.290] | 100 | 9 | 0 |
| A2 | 496 | 2.273 | [2.203, 2.375] | 146 | 9 | 0 |
| PIDO | 491 | 2.505 | [2.454, 2.590] | 278 | 9 | 0 |
| PID | 478 | 2.562 | [2.500, 2.616] | 322 | 9 | 0 |

Note. H1 is graded at σ = 0.25 only. These rows describe the σ = 0.35 legs and decide nothing. Interval: two-sided 95% seed-cluster bootstrap (30 seeds resampled whole, 10,000 draws, fixed seed).

### T9. PTA ablation, factor 1: what the 0.25 m vertical setpoint offset changed, controller by controller, nine wind stages (amplifying; decides no hypothesis)

| outcome | σ | controller | flights | without the offset | with the offset | change, with minus without [95% interval] | p | against the 35% floor, without → with | reading |
|---|---|---|---|---|---|---|---|---|---|
| captures (pp) | 0.25 | FlightDock-S1 | 540 | 69.6 | 67.2 | -2.41 [-6.48, +1.85] | 0.2860 | above → above | no clear change |
| peak height, p95 (m) | 0.25 | FlightDock-S1 | 513 | 2.110 | 1.904 | -0.206 [-0.320, -0.119] | 0.0000 |  | helped |
| captures (pp) | 0.25 | FlightDock-A2 | 540 | 61.9 | 55.6 | -6.30 [-11.85, -0.56] | 0.0348 | above → above | hurt |
| peak height, p95 (m) | 0.25 | FlightDock-A2 | 525 | 1.690 | 1.690 | +0.001 [-0.160, +0.120] | 0.9908 |  | no clear change |
| captures (pp) | 0.25 | PID-baseline | 540 | 57.8 | 67.6 | +9.81 [+4.07, +15.74] | 0.0008 | above → above | helped |
| peak height, p95 (m) | 0.25 | PID-baseline | 517 | 2.344 | 2.150 | -0.194 [-0.266, -0.100] | 0.0000 |  | helped |
| captures (pp) | 0.35 | FlightDock-S1 | 540 | 31.1 | 40.2 | +9.07 [+3.15, +15.00] | 0.0026 | straddles → above | helped |
| peak height, p95 (m) | 0.35 | FlightDock-S1 | 480 | 2.440 | 2.281 | -0.159 [-0.260, -0.026] | 0.0178 |  | helped |
| captures (pp) | 0.35 | FlightDock-A2 | 540 | 40.0 | 38.3 | -1.67 [-6.48, +3.33] | 0.5206 | above → straddles | no clear change |
| peak height, p95 (m) | 0.35 | FlightDock-A2 | 493 | 2.273 | 2.254 | -0.019 [-0.127, +0.040] | 0.4742 |  | no clear change |
| captures (pp) | 0.35 | PID-baseline | 540 | 20.9 | 27.0 | +6.11 [+1.67, +10.74] | 0.0044 | below → below | helped |
| peak height, p95 (m) | 0.35 | PID-baseline | 491 | 2.562 | 2.505 | -0.056 [-0.140, +0.032] | 0.2822 |  | no clear change |

Note. Change = with the offset minus without, on the same (stage, seed, trajectory) legs; the interval is the two-sided 95% seed-cluster bootstrap (30 seeds resampled whole, 10,000 draws, fixed seed) on both outcomes. "Reading" is the analyst's class rule (helped, hurt, no clear change by whether the interval excludes zero), not a verdict. p: two-sided bootstrap p, twice the smaller share of the 10,000 whole-seed resamples at or below and at or above zero; the ablation decides no hypothesis, so no α applies. "Against the 35% floor": each arm's pooled capture rate read against the 35% crewed aerial-refueling docking success rate (Ren and Quan, 2024) from the reference-floor table (above = whole interval above 35%, below = whole interval below, straddles = the interval contains 35%), without the factor → with it; the floor is read per arm and is unpaired, so a change of category is read with the paired change beside it. Amplifying; decides no hypothesis.

### T10. PTA ablation, factor 2: what the learned corrections changed with the offset held fixed, nine wind stages (amplifying; decides no hypothesis)

| outcome | σ | arms (with minus without) | flights | without the learned corrections | with the learned corrections | change, with minus without [95% interval] | p | against the 35% floor, without → with | reading |
|---|---|---|---|---|---|---|---|---|---|
| captures (pp) | 0.25 | S1 minus PID | 540 | 57.8 | 69.6 | +11.85 [+6.48, +17.59] | 0.0002 | above → above | helped |
| peak height, p95 (m) | 0.25 | S1 minus PID | 521 | 2.344 | 2.110 | -0.234 [-0.318, -0.150] | 0.0000 |  | helped |
| captures (pp) | 0.25 | A2 minus PID | 540 | 57.8 | 61.9 | +4.07 [-1.85, +10.19] | 0.1962 | above → above | no clear change |
| peak height, p95 (m) | 0.25 | A2 minus PID | 522 | 2.344 | 1.690 | -0.655 [-0.761, -0.537] | 0.0000 |  | helped |
| captures (pp) | 0.25 | S1O minus PIDO | 540 | 67.6 | 67.2 | -0.37 [-6.30, +5.75] | 0.9220 | above → above | no clear change |
| peak height, p95 (m) | 0.25 | S1O minus PIDO | 513 | 2.150 | 1.904 | -0.246 [-0.357, -0.160] | 0.0000 |  | helped |
| captures (pp) | 0.25 | A2O minus PIDO | 540 | 67.6 | 55.6 | -12.04 [-17.22, -6.85] | 0.0000 | above → above | hurt |
| peak height, p95 (m) | 0.25 | A2O minus PIDO | 525 | 2.150 | 1.690 | -0.460 [-0.613, -0.361] | 0.0000 |  | helped |
| captures (pp) | 0.35 | S1 minus PID | 540 | 20.9 | 31.1 | +10.19 [+5.00, +15.19] | 0.0000 | below → straddles | helped |
| peak height, p95 (m) | 0.35 | S1 minus PID | 482 | 2.562 | 2.440 | -0.122 [-0.240, -0.020] | 0.0180 |  | helped |
| captures (pp) | 0.35 | A2 minus PID | 540 | 20.9 | 40.0 | +19.07 [+14.63, +23.70] | 0.0000 | below → above | helped |
| peak height, p95 (m) | 0.35 | A2 minus PID | 496 | 2.562 | 2.273 | -0.289 [-0.374, -0.164] | 0.0000 |  | helped |
| captures (pp) | 0.35 | S1O minus PIDO | 540 | 27.0 | 40.2 | +13.15 [+7.78, +18.52] | 0.0000 | below → above | helped |
| peak height, p95 (m) | 0.35 | S1O minus PIDO | 480 | 2.505 | 2.281 | -0.224 [-0.328, -0.123] | 0.0004 |  | helped |
| captures (pp) | 0.35 | A2O minus PIDO | 540 | 27.0 | 38.3 | +11.30 [+6.11, +16.67] | 0.0000 | below → straddles | helped |
| peak height, p95 (m) | 0.35 | A2O minus PIDO | 493 | 2.505 | 2.254 | -0.251 [-0.322, -0.190] | 0.0000 |  | helped |

Note. Change = the FlightDock arm minus the PID-baseline in the same offset state (without the offset: S1 or A2 minus the PID-baseline; with the offset: S1O or A2O minus PIDO), on the same (stage, seed, trajectory) legs; two-sided 95% seed-cluster bootstrap interval on both outcomes (30 seeds resampled whole, 10,000 draws, fixed seed). The without-offset capture rows at σ 0.25 are the H2 statistics of the H2 table. "Reading" is the analyst's class rule (interval excludes zero), not a verdict. p: two-sided bootstrap p, twice the smaller share of the 10,000 whole-seed resamples at or below and at or above zero; the ablation decides no hypothesis, so no α applies. "Against the 35% floor": each arm's pooled capture rate read against the 35% crewed aerial-refueling docking success rate (Ren and Quan, 2024) from the reference-floor table (above = whole interval above 35%, below = whole interval below, straddles = the interval contains 35%), without the factor → with it; the floor is read per arm and is unpaired, so a change of category is read with the paired change beside it. Amplifying; decides no hypothesis.

### T11. Why legs were unsuccessful: one class per leg, nine wind stages, six arms

| σ | arm | unsuccessful legs (of 540) | never came within 10 m | overtook the DHC-2 in close formation | climbed above the 2.00 m bar (exclusion-zone violation) | came within 10 m, never inside the envelope on both channels | reached the envelope, held it under 3.0 s |
|---|---|---|---|---|---|---|---|
| 0.25 | S1O | 177 | 27 | 16 | 17 | 3 | 114 |
| 0.25 | S1 | 164 | 19 | 13 | 50 | 1 | 81 |
| 0.25 | A2O | 240 | 15 | 8 | 7 | 1 | 209 |
| 0.25 | A2 | 206 | 18 | 13 | 10 | 3 | 162 |
| 0.25 | PIDO | 175 | 23 | 20 | 98 | 2 | 32 |
| 0.25 | PID | 228 | 23 | 16 | 169 | 3 | 17 |
| 0.35 | S1O | 323 | 60 | 12 | 121 | 6 | 124 |
| 0.35 | S1 | 372 | 58 | 26 | 170 | 15 | 103 |
| 0.35 | A2O | 333 | 47 | 10 | 104 | 14 | 158 |
| 0.35 | A2 | 324 | 44 | 18 | 144 | 13 | 105 |
| 0.35 | PIDO | 394 | 49 | 20 | 273 | 10 | 42 |
| 0.35 | PID | 427 | 62 | 33 | 302 | 9 | 21 |

Note. Classes are assigned in the order listed, so each leg appears once: never within 10 m of the boom-tip; ahead of the DHC-2 in close formation (overtake); exclusion-zone violation above 2.00 m inside the corridor; never inside the M3b envelope on both channels; inside the envelope but the continuous dwell fell short of 3.0 s. The classification is the analyst's reading of the per-leg fail reasons on the board, not a scorer output.

### T12. Where each controller holds inside the M3b envelope and what it commands there (per tick, truth channel, nine wind stages)

| σ | arm | held height above the tip, truth median (m) | perception median (m) | ticks held below the tip (%) | applied altitude correction, median | ticks asking up on altitude (%) |
|---|---|---|---|---|---|---|
| 0.25 | S1O | -0.31 | -0.405 | 75.6 | -0.201 | 65.7 |
| 0.25 | S1 | -0.25 | -0.355 | 70.7 | 0.042 | 68.1 |
| 0.25 | A2O | -0.37 | -0.485 | 81.6 | -1.038 | 14.1 |
| 0.25 | A2 | -0.34 | -0.449 | 79.4 | -0.804 | 14.0 |
| 0.25 | PIDO | -0.21 | -0.298 | 66.7 | -0.25 | 67.7 |
| 0.25 | PID | -0.1 | -0.174 | 58.2 | 0.0 | 21.0 |
| 0.35 | S1O | -0.23 | -0.328 | 67.7 | -0.202 | 62.8 |
| 0.35 | S1 | -0.18 | -0.273 | 63.7 | 0.039 | 65.7 |
| 0.35 | A2O | -0.25 | -0.345 | 69.5 | -1.04 | 14.8 |
| 0.35 | A2 | -0.22 | -0.315 | 67.2 | -0.81 | 14.2 |
| 0.35 | PIDO | -0.14 | -0.219 | 60.7 | -0.25 | 67.1 |
| 0.35 | PID | -0.09 | -0.161 | 56.7 | 0.0 | 23.6 |

Note. Per-tick medians over every tick inside the envelope; no interval (ticks are not resampled). Negative height = below the boom-tip. The PID-baseline and PIDO apply no learned correction; PIDO's altitude change is the offset.

### T13. Calm Control repeatability: the same arm on the same 60 legs, flown twice

| arm | paired legs | first flight (σ 0.25 ladder) | second flight (σ 0.35 ladder) | change, second minus first (pp) | change: two-sided 95% seed-cluster bootstrap (pp) | legs that flipped | success lost | success gained |
|---|---|---|---|---|---|---|---|---|
| S1O | 60 | 38 of 60, 63.3% [51.7, 75.0] | 47 of 60, 78.3% [66.7, 90.0] | 15.0 | [-1.7, +31.7] | 23 | 7 | 16 |
| S1 | 60 | 50 of 60, 83.3% [73.3, 91.7] | 49 of 60, 81.7% [71.7, 91.7] | -1.7 | [-15.0, +11.7] | 19 | 10 | 9 |
| A2O | 60 | 11 of 60, 18.3% [10.0, 26.7] | 6 of 60, 10.0% [3.3, 16.7] | -8.3 | [-18.3, +1.7] | 13 | 9 | 4 |
| A2 | 60 | 17 of 60, 28.3% [16.7, 41.7] | 15 of 60, 25.0% [13.3, 36.7] | -3.3 | [-21.7, +13.3] | 20 | 11 | 9 |
| PIDO | 60 | 54 of 60, 90.0% [81.7, 96.7] | 56 of 60, 93.3% [85.0, 100.0] | 3.3 | [-5.0, +11.7] | 8 | 3 | 5 |
| PID | 60 | 57 of 60, 95.0% [88.3, 100.0] | 60 of 60, 100.0% [100.0, 100.0] | 5.0 | [+0.0, +11.7] | 3 | 0 | 3 |

Note. The stage was flown once in each ladder. The change is the second flight minus the first on the same (seed, trajectory) legs; its interval is the two-sided 95% seed-cluster bootstrap of the paired change (30 seeds resampled whole, 10,000 draws, fixed seed).

### T14. Approach funnel, nine wind stages, truth channel: legs reaching each step (of 540)

| σ | arm | within 10 m of the boom-tip | within 3 m | within 1.5 m | inside the envelope | inside for 1 s or more (cumulative) | inside for 3 s or more (cumulative) | capture (flag of record) |
|---|---|---|---|---|---|---|---|---|
| 0.25 | S1O | 513 of 540, 95.0% [92.6, 97.2] | 510 of 540, 94.4% [92.0, 96.7] | 510 of 540, 94.4% [92.0, 96.7] | 509 of 540, 94.3% [91.9, 96.5] | 509 of 540, 94.3% [91.9, 96.5] | 508 of 540, 94.1% [91.7, 96.3] | 363 of 540, 67.2% [63.5, 71.1] |
| 0.25 | S1 | 521 of 540, 96.5% [94.4, 98.3] | 519 of 540, 96.1% [94.1, 98.0] | 519 of 540, 96.1% [94.1, 98.0] | 519 of 540, 96.1% [94.1, 98.0] | 519 of 540, 96.1% [94.1, 98.0] | 519 of 540, 96.1% [94.1, 98.0] | 376 of 540, 69.6% [65.6, 73.5] |
| 0.25 | A2O | 525 of 540, 97.2% [95.9, 98.5] | 524 of 540, 97.0% [95.6, 98.3] | 524 of 540, 97.0% [95.6, 98.3] | 523 of 540, 96.9% [95.4, 98.3] | 523 of 540, 96.9% [95.4, 98.3] | 523 of 540, 96.9% [95.4, 98.3] | 300 of 540, 55.6% [51.1, 59.4] |
| 0.25 | A2 | 522 of 540, 96.7% [95.4, 98.0] | 518 of 540, 95.9% [94.6, 97.2] | 518 of 540, 95.9% [94.6, 97.2] | 518 of 540, 95.9% [94.6, 97.2] | 518 of 540, 95.9% [94.6, 97.2] | 518 of 540, 95.9% [94.6, 97.2] | 334 of 540, 61.9% [57.4, 66.1] |
| 0.25 | PIDO | 517 of 540, 95.7% [93.7, 97.6] | 515 of 540, 95.4% [93.5, 97.2] | 515 of 540, 95.4% [93.5, 97.2] | 513 of 540, 95.0% [93.0, 96.9] | 513 of 540, 95.0% [93.0, 96.9] | 513 of 540, 95.0% [93.0, 96.9] | 365 of 540, 67.6% [63.3, 71.7] |
| 0.25 | PID | 517 of 540, 95.7% [93.9, 97.4] | 514 of 540, 95.2% [93.1, 97.0] | 513 of 540, 95.0% [93.0, 96.9] | 513 of 540, 95.0% [93.0, 96.9] | 512 of 540, 94.8% [92.8, 96.7] | 512 of 540, 94.8% [92.8, 96.7] | 312 of 540, 57.8% [53.3, 61.9] |
| 0.35 | S1O | 480 of 540, 88.9% [85.7, 91.7] | 473 of 540, 87.6% [84.3, 90.7] | 473 of 540, 87.6% [84.3, 90.7] | 473 of 540, 87.6% [84.3, 90.7] | 473 of 540, 87.6% [84.3, 90.7] | 471 of 540, 87.2% [84.1, 90.4] | 217 of 540, 40.2% [36.3, 44.3] |
| 0.35 | S1 | 482 of 540, 89.3% [86.5, 91.9] | 467 of 540, 86.5% [83.5, 89.4] | 467 of 540, 86.5% [83.5, 89.4] | 467 of 540, 86.5% [83.5, 89.4] | 466 of 540, 86.3% [83.3, 89.3] | 461 of 540, 85.4% [82.4, 88.3] | 168 of 540, 31.1% [27.2, 35.0] |
| 0.35 | A2O | 493 of 540, 91.3% [88.5, 93.9] | 479 of 540, 88.7% [85.4, 91.9] | 479 of 540, 88.7% [85.4, 91.9] | 478 of 540, 88.5% [85.2, 91.7] | 477 of 540, 88.3% [85.0, 91.5] | 472 of 540, 87.4% [83.9, 90.7] | 207 of 540, 38.3% [33.7, 43.0] |
| 0.35 | A2 | 496 of 540, 91.9% [89.4, 94.1] | 484 of 540, 89.6% [86.9, 92.4] | 484 of 540, 89.6% [86.9, 92.4] | 483 of 540, 89.4% [86.7, 92.0] | 480 of 540, 88.9% [86.1, 91.7] | 476 of 540, 88.1% [85.2, 90.9] | 216 of 540, 40.0% [36.5, 43.3] |
| 0.35 | PIDO | 491 of 540, 90.9% [87.6, 94.1] | 478 of 540, 88.5% [85.2, 91.9] | 477 of 540, 88.3% [85.0, 91.5] | 476 of 540, 88.1% [84.8, 91.5] | 476 of 540, 88.1% [84.8, 91.5] | 474 of 540, 87.8% [84.3, 91.1] | 146 of 540, 27.0% [23.3, 30.7] |
| 0.35 | PID | 478 of 540, 88.5% [85.6, 91.5] | 466 of 540, 86.3% [83.0, 89.6] | 465 of 540, 86.1% [82.8, 89.4] | 463 of 540, 85.7% [82.4, 89.1] | 461 of 540, 85.4% [82.0, 88.5] | 458 of 540, 84.8% [81.3, 88.1] | 113 of 540, 20.9% [17.4, 24.6] |

Note. Each column is the share of the 540 legs that reached that step at least once; from left to right the counts only fall; time inside the envelope is ticks times the tick period, not necessarily continuous. Intervals: two-sided 95% seed-cluster bootstrap (30 seeds resampled whole, 10,000 draws, fixed seed). The last column is the four-condition capture flag of record.

### T15. Docking-envelope captures by DHC-2 trajectory, nine wind stages pooled (270 legs per trajectory)

| σ | arm | straight: captures of 270, rate [interval] | racetrack: captures of 270, rate [interval] | straight minus racetrack, within seed (pp) [interval] | that difference minus the PID-baseline's (pp) [interval] |
|---|---|---|---|---|---|
| 0.25 | S1O | 192 of 270, 71.1% [63.7, 78.5] | 171 of 270, 63.3% [58.1, 68.9] | +7.8 [-2.6, +17.8] | +26.3 [+16.7, +35.9] |
| 0.25 | S1 | 198 of 270, 73.3% [67.8, 78.9] | 178 of 270, 65.9% [59.3, 72.2] | +7.4 [-1.5, +16.3] | +25.9 [+17.0, +35.2] |
| 0.25 | A2O | 166 of 270, 61.5% [54.4, 67.8] | 134 of 270, 49.6% [43.3, 55.9] | +11.9 [+1.9, +21.9] | +30.4 [+18.9, +41.9] |
| 0.25 | A2 | 182 of 270, 67.4% [61.5, 73.3] | 152 of 270, 56.3% [50.4, 61.9] | +11.1 [+3.3, +18.9] | +29.6 [+19.6, +39.3] |
| 0.25 | PIDO | 167 of 270, 61.9% [55.6, 67.8] | 198 of 270, 73.3% [68.9, 77.8] | -11.5 [-18.5, -5.2] | +7.0 [-2.2, +16.7] |
| 0.25 | PID | 131 of 270, 48.5% [41.9, 54.8] | 181 of 270, 67.0% [61.1, 72.6] | -18.5 [-27.4, -9.6] | 0.0, the baseline |
| 0.35 | S1O | 114 of 270, 42.2% [36.7, 47.8] | 103 of 270, 38.1% [32.6, 44.1] | +4.1 [-3.7, +12.2] | +10.4 [+0.4, +20.0] |
| 0.35 | S1 | 80 of 270, 29.6% [23.7, 35.2] | 88 of 270, 32.6% [28.1, 37.0] | -3.0 [-10.0, +3.7] | +3.3 [-4.1, +10.7] |
| 0.35 | A2O | 105 of 270, 38.9% [31.9, 45.9] | 102 of 270, 37.8% [31.9, 43.7] | +1.1 [-8.1, +10.0] | +7.4 [-4.1, +18.1] |
| 0.35 | A2 | 113 of 270, 41.9% [37.4, 45.9] | 103 of 270, 38.1% [32.2, 44.1] | +3.7 [-3.7, +11.5] | +10.0 [+1.1, +18.5] |
| 0.35 | PIDO | 66 of 270, 24.4% [19.3, 29.6] | 80 of 270, 29.6% [24.4, 34.5] | -5.2 [-12.6, +1.9] | +1.1 [-5.6, +8.1] |
| 0.35 | PID | 48 of 270, 17.8% [13.3, 22.6] | 65 of 270, 24.1% [19.6, 28.5] | -6.3 [-11.9, -0.4] | 0.0, the baseline |

Note. Two-sided 95% seed-cluster bootstrap intervals (30 seeds resampled whole, 10,000 draws, fixed seed); the differences are paired within seed. Amplifying; decides no hypothesis.

### T16. Same-seed gust-stream agreement against the PID-baseline, and the capture difference with and without the disagreeing pairs

| σ | arm | n_pairs | pairs with identical gust streams (%) | capture difference, all pairs (pp) | 95% interval, all pairs | capture difference, identical-stream pairs only (pp) | 95% interval, identical-stream pairs | identical-stream pairs |
|---|---|---|---|---|---|---|---|---|
| 0.25 | S1O | 540 | 94.81 | +9.44 | [+4.3, +14.6] | +10.55 | [+5.1, +15.9] | 512 |
| 0.25 | S1 | 540 | 93.15 | +11.85 | [+6.5, +17.6] | +12.52 | [+7.1, +18.2] | 503 |
| 0.25 | A2O | 540 | 95.19 | -2.22 | [-8.0, +3.3] | -1.75 | [-7.8, +4.1] | 514 |
| 0.25 | A2 | 540 | 94.63 | +4.07 | [-1.9, +10.2] | +3.91 | [-2.0, +10.2] | 511 |
| 0.25 | PIDO | 540 | 94.63 | +9.81 | [+4.1, +15.7] | +10.96 | [+5.1, +16.9] | 511 |
| 0.35 | S1O | 540 | 92.04 | +19.26 | [+13.3, +24.8] | +18.31 | [+12.1, +24.2] | 497 |
| 0.35 | S1 | 540 | 90.0 | +10.19 | [+5.0, +15.2] | +9.67 | [+4.9, +14.4] | 486 |
| 0.35 | A2O | 540 | 89.26 | +17.41 | [+11.8, +22.8] | +15.56 | [+9.0, +21.9] | 482 |
| 0.35 | A2 | 540 | 91.3 | +19.07 | [+14.6, +23.7] | +18.66 | [+13.7, +23.7] | 493 |
| 0.35 | PIDO | 540 | 91.48 | +6.11 | [+1.7, +10.7] | +3.85 | [-0.8, +8.8] | 494 |

Note. Across all six arms, 479 of 540 wind-stage legs at σ 0.25 and 421 of 540 at σ 0.35 flew on one and the same gust stream in every arm; the rest had at least one arm re-rolled by a warm-up retry. Intervals in this table are the two-sided 95% seed-cluster bootstrap of the source file. Sign: arm minus the PID-baseline in every row (the source lists the PID-baseline against PIDO once, as PID minus PIDO; that row is shown as PIDO minus the PID-baseline).

### T17. Docking-envelope captures by wind stage and arm, σ = 0.25 (60 legs per stage and arm; the nine-stage pool is the last row)

| wind stage | S1O | S1 | A2O | A2 | PIDO | PID |
|---|---|---|---|---|---|---|
| Calm Control | 38 of 60, 63.3% [51.7, 75.0] | 50 of 60, 83.3% [73.3, 91.7] | 11 of 60, 18.3% [10.0, 26.7] | 17 of 60, 28.3% [16.7, 41.7] | 54 of 60, 90.0% [81.7, 96.7] | 57 of 60, 95.0% [88.3, 100.0] |
| 0 m/s | 38 of 60, 63.3% [50.0, 76.7] | 38 of 60, 63.3% [56.7, 71.7] | 35 of 60, 58.3% [46.7, 70.0] | 39 of 60, 65.0% [51.7, 78.3] | 39 of 60, 65.0% [53.3, 76.7] | 35 of 60, 58.3% [45.0, 71.7] |
| 1 m/s | 34 of 60, 56.7% [45.0, 68.3] | 40 of 60, 66.7% [53.3, 80.0] | 37 of 60, 61.7% [51.7, 71.7] | 36 of 60, 60.0% [48.3, 71.7] | 37 of 60, 61.7% [48.3, 75.0] | 27 of 60, 45.0% [31.7, 60.0] |
| 2 m/s | 38 of 60, 63.3% [51.7, 75.0] | 36 of 60, 60.0% [45.0, 73.3] | 38 of 60, 63.3% [53.3, 73.3] | 42 of 60, 70.0% [60.0, 80.0] | 44 of 60, 73.3% [63.3, 83.3] | 37 of 60, 61.7% [48.3, 73.3] |
| 3 m/s | 41 of 60, 68.3% [55.0, 80.0] | 40 of 60, 66.7% [55.0, 78.3] | 29 of 60, 48.3% [35.0, 61.7] | 39 of 60, 65.0% [53.3, 76.7] | 44 of 60, 73.3% [63.3, 83.3] | 35 of 60, 58.3% [48.3, 68.3] |
| 4 m/s | 47 of 60, 78.3% [66.7, 88.3] | 47 of 60, 78.3% [66.7, 88.3] | 40 of 60, 66.7% [53.3, 80.0] | 36 of 60, 60.0% [46.7, 71.7] | 40 of 60, 66.7% [55.0, 76.7] | 34 of 60, 56.7% [45.0, 68.3] |
| 5 m/s | 36 of 60, 60.0% [50.0, 70.0] | 43 of 60, 71.7% [60.0, 81.7] | 27 of 60, 45.0% [33.3, 56.7] | 38 of 60, 63.3% [51.7, 75.0] | 38 of 60, 63.3% [50.0, 76.7] | 33 of 60, 55.0% [43.3, 66.7] |
| 6 m/s | 44 of 60, 73.3% [61.7, 83.3] | 45 of 60, 75.0% [61.7, 86.7] | 35 of 60, 58.3% [45.0, 71.7] | 37 of 60, 61.7% [48.3, 73.3] | 44 of 60, 73.3% [61.7, 83.3] | 36 of 60, 60.0% [46.7, 73.3] |
| 7 m/s | 42 of 60, 70.0% [56.7, 81.7] | 43 of 60, 71.7% [60.0, 81.7] | 28 of 60, 46.7% [33.3, 60.0] | 32 of 60, 53.3% [40.0, 66.7] | 43 of 60, 71.7% [60.0, 83.3] | 36 of 60, 60.0% [46.7, 71.7] |
| 8 m/s | 43 of 60, 71.7% [60.0, 83.3] | 44 of 60, 73.3% [60.0, 85.0] | 31 of 60, 51.7% [38.3, 65.0] | 35 of 60, 58.3% [45.0, 70.0] | 36 of 60, 60.0% [48.3, 71.7] | 39 of 60, 65.0% [50.0, 78.3] |
| All nine wind stages (H2 pool) | 363 of 540, 67.2% [63.5, 71.1] | 376 of 540, 69.6% [65.6, 73.5] | 300 of 540, 55.6% [51.1, 59.4] | 334 of 540, 61.9% [57.4, 66.1] | 365 of 540, 67.6% [63.3, 71.7] | 312 of 540, 57.8% [53.3, 61.9] |

Note. Cell: captures of 60 (of 540 for the pool), rate [two-sided 95% seed-cluster bootstrap interval] (30 seeds resampled whole, 10,000 draws). Calm Control (no wind, no turbulence) enters no pool. The paired statistics of record are in the H2 and H3 tables.

### T18. Docking-envelope captures by wind stage and arm, σ = 0.35 (60 legs per stage and arm; the nine-stage pool is the last row)

| wind stage | S1O | S1 | A2O | A2 | PIDO | PID |
|---|---|---|---|---|---|---|
| Calm Control | 47 of 60, 78.3% [66.7, 90.0] | 49 of 60, 81.7% [71.7, 91.7] | 6 of 60, 10.0% [3.3, 16.7] | 15 of 60, 25.0% [13.3, 36.7] | 56 of 60, 93.3% [85.0, 100.0] | 60 of 60, 100.0% [100.0, 100.0] |
| 0 m/s | 30 of 60, 50.0% [36.7, 63.3] | 15 of 60, 25.0% [15.0, 36.7] | 19 of 60, 31.7% [21.7, 41.7] | 25 of 60, 41.7% [30.0, 53.3] | 15 of 60, 25.0% [15.0, 36.7] | 11 of 60, 18.3% [10.0, 28.3] |
| 1 m/s | 15 of 60, 25.0% [13.3, 38.3] | 9 of 60, 15.0% [6.7, 23.3] | 22 of 60, 36.7% [25.0, 48.3] | 23 of 60, 38.3% [26.7, 50.0] | 16 of 60, 26.7% [16.7, 36.7] | 8 of 60, 13.3% [5.0, 23.3] |
| 2 m/s | 26 of 60, 43.3% [31.7, 55.0] | 26 of 60, 43.3% [31.7, 55.0] | 21 of 60, 35.0% [25.0, 45.0] | 28 of 60, 46.7% [35.0, 58.3] | 16 of 60, 26.7% [16.7, 36.7] | 14 of 60, 23.3% [13.3, 33.3] |
| 3 m/s | 24 of 60, 40.0% [28.3, 53.3] | 17 of 60, 28.3% [18.3, 38.3] | 27 of 60, 45.0% [31.7, 58.3] | 25 of 60, 41.7% [30.0, 55.0] | 15 of 60, 25.0% [15.0, 36.7] | 18 of 60, 30.0% [18.3, 43.3] |
| 4 m/s | 27 of 60, 45.0% [33.3, 56.7] | 17 of 60, 28.3% [16.7, 40.0] | 23 of 60, 38.3% [25.0, 51.7] | 25 of 60, 41.7% [31.7, 50.0] | 14 of 60, 23.3% [13.3, 33.3] | 7 of 60, 11.7% [5.0, 20.0] |
| 5 m/s | 24 of 60, 40.0% [26.7, 53.3] | 25 of 60, 41.7% [30.0, 53.3] | 28 of 60, 46.7% [35.0, 58.3] | 19 of 60, 31.7% [20.0, 43.3] | 14 of 60, 23.3% [15.0, 31.7] | 13 of 60, 21.7% [11.7, 31.7] |
| 6 m/s | 22 of 60, 36.7% [23.3, 50.0] | 13 of 60, 21.7% [10.0, 33.3] | 23 of 60, 38.3% [25.0, 51.7] | 15 of 60, 25.0% [15.0, 36.7] | 17 of 60, 28.3% [18.3, 38.3] | 13 of 60, 21.7% [11.7, 33.3] |
| 7 m/s | 24 of 60, 40.0% [28.3, 51.7] | 26 of 60, 43.3% [31.7, 55.0] | 22 of 60, 36.7% [23.3, 50.0] | 27 of 60, 45.0% [33.3, 58.3] | 16 of 60, 26.7% [15.0, 40.0] | 14 of 60, 23.3% [13.3, 35.0] |
| 8 m/s | 25 of 60, 41.7% [30.0, 53.3] | 20 of 60, 33.3% [20.0, 46.7] | 22 of 60, 36.7% [23.3, 50.0] | 29 of 60, 48.3% [36.7, 58.3] | 23 of 60, 38.3% [26.7, 51.7] | 15 of 60, 25.0% [15.0, 36.7] |
| All nine wind stages | 217 of 540, 40.2% [36.3, 44.3] | 168 of 540, 31.1% [27.2, 35.0] | 207 of 540, 38.3% [33.7, 43.0] | 216 of 540, 40.0% [36.5, 43.3] | 146 of 540, 27.0% [23.3, 30.7] | 113 of 540, 20.9% [17.4, 24.6] |

Note. Cell: captures of 60 (of 540 for the pool), rate [two-sided 95% seed-cluster bootstrap interval] (30 seeds resampled whole, 10,000 draws). Calm Control (no wind, no turbulence) enters no pool. The paired statistics of record are in the H2 and H3 tables.

### T19. Clearance and guard record by arm and wind stage, six arms, σ = 0.25, terminal approach corridor only

| arm | stage | captured | entered corridor | never reached corridor | legs over 2.00 m in corridor | legs over 3.42 m in corridor | altitude-guard fires in corridor | overtake-guard fires in corridor |
|---|---|---|---|---|---|---|---|---|
| S1O | S0 calm control stage | 38 | 60 | 0 | 0 | 0 | 0 | 105 |
| S1O | 0 m/s | 38 | 57 | 3 | 1 | 0 | 1 | 12 |
| S1O | 1 m/s | 34 | 57 | 3 | 2 | 0 | 3 | 19 |
| S1O | 2 m/s | 38 | 57 | 3 | 2 | 0 | 3 | 16 |
| S1O | 3 m/s | 41 | 58 | 2 | 2 | 0 | 1 | 17 |
| S1O | 4 m/s | 47 | 57 | 3 | 1 | 0 | 1 | 21 |
| S1O | 5 m/s | 36 | 56 | 4 | 4 | 0 | 4 | 17 |
| S1O | 6 m/s | 44 | 56 | 4 | 1 | 0 | 1 | 21 |
| S1O | 7 m/s | 42 | 58 | 2 | 2 | 0 | 2 | 22 |
| S1O | 8 m/s | 43 | 57 | 3 | 2 | 0 | 4 | 18 |
| S1 | S0 calm control stage | 50 | 60 | 0 | 0 | 0 | 0 | 126 |
| S1 | 0 m/s | 38 | 58 | 2 | 11 | 0 | 12 | 13 |
| S1 | 1 m/s | 40 | 59 | 1 | 8 | 0 | 8 | 24 |
| S1 | 2 m/s | 36 | 57 | 3 | 9 | 0 | 11 | 24 |
| S1 | 3 m/s | 40 | 58 | 2 | 4 | 0 | 5 | 19 |
| S1 | 4 m/s | 47 | 59 | 1 | 4 | 0 | 5 | 15 |
| S1 | 5 m/s | 43 | 56 | 4 | 4 | 0 | 4 | 22 |
| S1 | 6 m/s | 45 | 57 | 3 | 2 | 0 | 3 | 23 |
| S1 | 7 m/s | 43 | 57 | 3 | 6 | 0 | 6 | 24 |
| S1 | 8 m/s | 44 | 60 | 0 | 4 | 0 | 5 | 21 |
| A2O | S0 calm control stage | 11 | 60 | 0 | 0 | 0 | 0 | 39 |
| A2O | 0 m/s | 35 | 60 | 0 | 1 | 0 | 1 | 10 |
| A2O | 1 m/s | 37 | 58 | 2 | 1 | 0 | 2 | 12 |
| A2O | 2 m/s | 38 | 60 | 0 | 2 | 0 | 1 | 12 |
| A2O | 3 m/s | 29 | 59 | 1 | 1 | 0 | 1 | 15 |
| A2O | 4 m/s | 40 | 60 | 0 | 0 | 0 | 0 | 17 |
| A2O | 5 m/s | 27 | 58 | 2 | 0 | 0 | 0 | 16 |
| A2O | 6 m/s | 35 | 59 | 1 | 0 | 0 | 0 | 13 |
| A2O | 7 m/s | 28 | 55 | 5 | 1 | 0 | 1 | 13 |
| A2O | 8 m/s | 31 | 56 | 4 | 1 | 0 | 1 | 15 |
| A2 | S0 calm control stage | 17 | 60 | 0 | 0 | 0 | 0 | 65 |
| A2 | 0 m/s | 39 | 57 | 3 | 0 | 0 | 0 | 10 |
| A2 | 1 m/s | 36 | 57 | 3 | 1 | 0 | 1 | 17 |
| A2 | 2 m/s | 42 | 59 | 1 | 2 | 0 | 2 | 20 |
| A2 | 3 m/s | 39 | 60 | 0 | 2 | 0 | 2 | 15 |
| A2 | 4 m/s | 36 | 59 | 1 | 2 | 0 | 2 | 13 |
| A2 | 5 m/s | 38 | 58 | 2 | 0 | 0 | 0 | 11 |
| A2 | 6 m/s | 37 | 58 | 2 | 0 | 0 | 0 | 15 |
| A2 | 7 m/s | 32 | 58 | 2 | 1 | 0 | 1 | 17 |
| A2 | 8 m/s | 35 | 56 | 4 | 1 | 0 | 1 | 17 |
| PIDO | S0 calm control stage | 54 | 60 | 0 | 0 | 0 | 0 | 193 |
| PIDO | 0 m/s | 39 | 60 | 0 | 10 | 0 | 16 | 27 |
| PIDO | 1 m/s | 37 | 54 | 6 | 10 | 0 | 10 | 28 |
| PIDO | 2 m/s | 44 | 58 | 2 | 11 | 0 | 15 | 28 |
| PIDO | 3 m/s | 44 | 58 | 2 | 8 | 0 | 9 | 30 |
| PIDO | 4 m/s | 40 | 58 | 2 | 12 | 0 | 13 | 23 |
| PIDO | 5 m/s | 38 | 54 | 6 | 8 | 0 | 11 | 26 |
| PIDO | 6 m/s | 44 | 58 | 2 | 10 | 0 | 11 | 35 |
| PIDO | 7 m/s | 43 | 59 | 1 | 12 | 0 | 16 | 32 |
| PIDO | 8 m/s | 36 | 58 | 2 | 17 | 0 | 17 | 31 |
| PID | S0 calm control stage | 57 | 60 | 0 | 0 | 0 | 0 | 168 |
| PID | 0 m/s | 35 | 58 | 2 | 19 | 0 | 25 | 18 |
| PID | 1 m/s | 27 | 54 | 6 | 23 | 0 | 34 | 29 |
| PID | 2 m/s | 37 | 58 | 2 | 16 | 0 | 20 | 23 |
| PID | 3 m/s | 35 | 58 | 2 | 18 | 0 | 22 | 23 |
| PID | 4 m/s | 34 | 59 | 1 | 21 | 0 | 29 | 33 |
| PID | 5 m/s | 33 | 55 | 5 | 18 | 0 | 26 | 23 |
| PID | 6 m/s | 36 | 58 | 2 | 20 | 0 | 24 | 20 |
| PID | 7 m/s | 36 | 57 | 3 | 16 | 0 | 18 | 34 |
| PID | 8 m/s | 39 | 60 | 0 | 20 | 0 | 28 | 31 |

Note. Every cell holds the 60 legs of its stage. Legs over 2.00 m are exclusion-zone violations inside the corridor on the truth channel; a leg over 3.42 m would be airframe contact. Guard fires are counted only when the truth range to the boom-tip was 10 m or less at the fire (the terminal approach corridor), from the fire events of record.

### T20. Clearance and guard record by arm and wind stage, six arms, σ = 0.35, terminal approach corridor only

| arm | stage | captured | entered corridor | never reached corridor | legs over 2.00 m in corridor | legs over 3.42 m in corridor | altitude-guard fires in corridor | overtake-guard fires in corridor |
|---|---|---|---|---|---|---|---|---|
| S1O | S0 calm control stage | 47 | 60 | 0 | 0 | 0 | 0 | 130 |
| S1O | 0 m/s | 30 | 56 | 4 | 16 | 0 | 22 | 5 |
| S1O | 1 m/s | 15 | 50 | 10 | 20 | 0 | 22 | 5 |
| S1O | 2 m/s | 26 | 55 | 5 | 14 | 0 | 16 | 6 |
| S1O | 3 m/s | 24 | 56 | 4 | 15 | 0 | 18 | 5 |
| S1O | 4 m/s | 27 | 47 | 13 | 8 | 0 | 12 | 7 |
| S1O | 5 m/s | 24 | 55 | 5 | 15 | 0 | 18 | 9 |
| S1O | 6 m/s | 22 | 50 | 10 | 11 | 0 | 15 | 9 |
| S1O | 7 m/s | 24 | 52 | 8 | 10 | 0 | 11 | 10 |
| S1O | 8 m/s | 25 | 59 | 1 | 17 | 0 | 20 | 13 |
| S1 | S0 calm control stage | 49 | 60 | 0 | 0 | 0 | 0 | 124 |
| S1 | 0 m/s | 15 | 53 | 7 | 23 | 0 | 29 | 4 |
| S1 | 1 m/s | 9 | 54 | 6 | 26 | 0 | 31 | 5 |
| S1 | 2 m/s | 26 | 56 | 4 | 13 | 0 | 21 | 8 |
| S1 | 3 m/s | 17 | 57 | 3 | 25 | 0 | 32 | 9 |
| S1 | 4 m/s | 17 | 49 | 11 | 18 | 0 | 24 | 2 |
| S1 | 5 m/s | 25 | 53 | 7 | 16 | 0 | 23 | 10 |
| S1 | 6 m/s | 13 | 49 | 11 | 20 | 0 | 25 | 8 |
| S1 | 7 m/s | 26 | 54 | 6 | 18 | 0 | 20 | 9 |
| S1 | 8 m/s | 20 | 57 | 3 | 20 | 0 | 23 | 12 |
| A2O | S0 calm control stage | 6 | 60 | 0 | 0 | 0 | 0 | 33 |
| A2O | 0 m/s | 19 | 50 | 10 | 15 | 0 | 15 | 5 |
| A2O | 1 m/s | 22 | 54 | 6 | 14 | 0 | 18 | 6 |
| A2O | 2 m/s | 21 | 57 | 3 | 14 | 0 | 16 | 2 |
| A2O | 3 m/s | 27 | 56 | 4 | 10 | 0 | 11 | 6 |
| A2O | 4 m/s | 23 | 57 | 3 | 10 | 0 | 11 | 2 |
| A2O | 5 m/s | 28 | 56 | 4 | 9 | 0 | 10 | 6 |
| A2O | 6 m/s | 23 | 53 | 7 | 9 | 0 | 10 | 3 |
| A2O | 7 m/s | 22 | 56 | 4 | 10 | 0 | 11 | 4 |
| A2O | 8 m/s | 22 | 54 | 6 | 9 | 0 | 10 | 2 |
| A2 | S0 calm control stage | 15 | 60 | 0 | 0 | 0 | 0 | 71 |
| A2 | 0 m/s | 25 | 55 | 5 | 17 | 0 | 20 | 1 |
| A2 | 1 m/s | 23 | 55 | 5 | 21 | 0 | 23 | 4 |
| A2 | 2 m/s | 28 | 57 | 3 | 19 | 0 | 22 | 8 |
| A2 | 3 m/s | 25 | 58 | 2 | 14 | 0 | 16 | 5 |
| A2 | 4 m/s | 25 | 54 | 6 | 17 | 0 | 24 | 5 |
| A2 | 5 m/s | 19 | 52 | 8 | 15 | 0 | 19 | 8 |
| A2 | 6 m/s | 15 | 54 | 6 | 21 | 0 | 22 | 6 |
| A2 | 7 m/s | 27 | 54 | 6 | 10 | 0 | 11 | 9 |
| A2 | 8 m/s | 29 | 57 | 3 | 12 | 0 | 13 | 9 |
| PIDO | S0 calm control stage | 56 | 60 | 0 | 0 | 0 | 0 | 184 |
| PIDO | 0 m/s | 15 | 58 | 2 | 38 | 0 | 48 | 14 |
| PIDO | 1 m/s | 16 | 53 | 7 | 25 | 0 | 32 | 9 |
| PIDO | 2 m/s | 16 | 58 | 2 | 32 | 0 | 43 | 13 |
| PIDO | 3 m/s | 15 | 55 | 5 | 33 | 0 | 41 | 6 |
| PIDO | 4 m/s | 14 | 52 | 8 | 30 | 0 | 42 | 8 |
| PIDO | 5 m/s | 14 | 54 | 6 | 32 | 0 | 49 | 8 |
| PIDO | 6 m/s | 17 | 52 | 8 | 32 | 0 | 41 | 5 |
| PIDO | 7 m/s | 16 | 55 | 5 | 30 | 0 | 35 | 15 |
| PIDO | 8 m/s | 23 | 54 | 6 | 26 | 0 | 36 | 7 |
| PID | S0 calm control stage | 60 | 60 | 0 | 0 | 0 | 0 | 155 |
| PID | 0 m/s | 11 | 55 | 5 | 37 | 0 | 60 | 12 |
| PID | 1 m/s | 8 | 53 | 7 | 42 | 0 | 55 | 6 |
| PID | 2 m/s | 14 | 52 | 8 | 34 | 0 | 46 | 2 |
| PID | 3 m/s | 18 | 58 | 2 | 33 | 0 | 50 | 6 |
| PID | 4 m/s | 7 | 48 | 12 | 37 | 0 | 54 | 8 |
| PID | 5 m/s | 13 | 54 | 6 | 35 | 0 | 44 | 9 |
| PID | 6 m/s | 13 | 52 | 8 | 32 | 0 | 49 | 1 |
| PID | 7 m/s | 14 | 50 | 10 | 33 | 0 | 45 | 10 |
| PID | 8 m/s | 15 | 56 | 4 | 39 | 0 | 53 | 8 |

Note. Every cell holds the 60 legs of its stage. Legs over 2.00 m are exclusion-zone violations inside the corridor on the truth channel; a leg over 3.42 m would be airframe contact. Guard fires are counted only when the truth range to the boom-tip was 10 m or less at the fire (the terminal approach corridor), from the fire events of record.

### T21. Hypothesis verdicts by arm: which arm met which hypothesis (the four FlightDock arms are judged; the PID-baseline and PIDO are comparators, not graded)

| arm | H1: pooled corridor p95, truth channel, σ 0.25 (m); bar: upper bound < 2.00 m. Value [95% interval], verdict | H2: same-seed capture difference against the PID-baseline, σ 0.25 (pp); bar: lower bound > 0 pp. Value [95% interval], verdict | H3: same-seed capture decrease, σ 0.25 to σ 0.35 (pp); bar: upper bound < 35 pp. Value [95% interval], verdict | hypotheses met |
|---|---|---|---|---|
| S1O | 1.904 [1.800, 1.990] m, met | +9.44 [+4.26, +14.63] pp, met | 27.04 [21.48, 32.59] pp, met | all three |
| S1 | 2.110 [2.064, 2.146] m, not met | +11.85 [+6.48, +17.59] pp, met | 38.52 [32.22, 44.63] pp, not met | H2 only |
| A2O | 1.690 [1.550, 1.780] m, met | -2.22 [-7.96, +3.33] pp, not met | 17.22 [11.30, 22.96] pp, met | H1 and H3 |
| A2 | 1.690 [1.600, 1.800] m, met | +4.07 [-1.85, +10.19] pp, not met | 21.85 [16.67, 26.85] pp, met | H1 and H3 |
| PIDO | 2.150 [2.130, 2.210] m, comparator (not graded) | +9.81 [+4.07, +15.74] pp, comparator (not graded) | 40.56 [36.30, 44.63] pp, comparator (not graded) | comparator (not graded) |
| PID | 2.344 [2.265, 2.410] m, comparator (not graded) | 0 pp, baseline comparator (not graded) | 36.85 [31.11, 42.41] pp, comparator (not graded) | comparator (not graded) |

Note. One-sided test at α = 0.025 through the two-sided 95% seed-cluster bootstrap interval (the table “Hypothesis Tests”); the point estimate is reported beside the interval; the one-sided bootstrap p of each cell is in the H1, H2 and H3 tables.
