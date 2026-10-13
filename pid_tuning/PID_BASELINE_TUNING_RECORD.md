# PID-baseline Tuning Record

Early period: 26 Nov 2025 to 23 Feb 2026, zero wind and zero turbulence, 376 numbered tuning tests (668 flights) in ArduPilot SITL.

| Month | Numbered Tests | Flights | Reached Hold Precontact | Reached Docking Creep | Reached Contact |
|---|---|---|---|---|---|
| Nov 2025 | 69 | 69 | 63 | 0 | 0 |
| Dec 2025 | 247 | 247 | 160 | 36 | 4 |
| Jan 2026 | 60 | 343 | 56 | 33 | 1 |
| Feb 2026 | 1 | 9 | 0 | 0 | 0 |
| Total | 376 | 668 | 279 | 69 | 5 |

Late period: 20 June to 19 July 2026, mean wind 0 to 2 m/s, turbulence no higher than σ = 0.1, 411 tuning flights in ArduPilot SITL.

| Trajectory | Mean wind speed (m/s) | Turbulence Setting (σ) | Reached Contact, all flights | Reached Contact, final flights 3 to 19 Jul 2026 |
|---|---|---|---|---|
| Straight | 0 | 0.0 | 169 of 201 (84.1%) | 54 of 54 (100%) |
| Racetrack | 0 | 0.0 | 76 of 77 (98.7%) | 7 of 7 (100%) |
| Straight | 1 | 0.0 | 16 of 24 (66.7%) | 6 of 6 (100%) |
| Straight | 1 | 0.1 | 32 of 61 (52.5%) | 15 of 15 (100%) |
| Racetrack | 1 | 0.1 | 12 of 41 (29.3%) | none flown |
| Straight | 2 | 0.1 | 3 of 7 (42.9%) | none flown |
| All flights | - | - | 308 of 411 (74.9%) | 82 of 82 (100%) |

Notes

- Early period: a numbered test was often flown more than once; the three "Reached" columns count flights. "Reached Contact" = the state machine reached Hold Contact or an M3b entry exists in the flight's metrics record; "Reached Hold Precontact" and "Reached Docking Creep" = the state machine entered that state.
- Late period: "Reached Contact" = the M3b envelope was entered at least once (`m3b_entry`), no dwell required, overshoot allowed. Final flights = 2026-07-03 onward.
- Source files: `PID_TUNING_EARLY_CONTACT_CENSUS_20260930.csv` (668 rows, one per flight log) and `PID_TUNING_LATE_CONTACT_CENSUS_20260930.csv` (411 rows, one per flight). Both are beside this script; `pid_baseline_tuning_record.py` rebuilds this file from them.
