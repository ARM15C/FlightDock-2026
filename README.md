# FlightDock-2026

Reproduction package for the FlightDock study: a learned residual correction (RRL) layered on a PID formation
controller, docking a QuadPlane to the flying-boom tip of a DHC-2 in ArduPilot SITL, benchmarked over 600 legs per
arm at two turbulence levels. This repository holds the code, the two benchmarked policies, the benchmark design, the
results of record, and the scripts that rebuild every results table and figure. The flight data (twelve corpora,
about 155 GB) are in a private S3 bucket, `s3://flightdock-reproduction-2026/`; see "Flight data" below.

## Layout

| folder | contents |
|---|---|
| `src/` | Python source at the benchmark pin: Gymnasium environment, reward, residual overlay, curriculum, training entry point |
| `tests/` | runtime modules: the PID formation controller, control laws, state machine and the SITL launch helpers (the folder name is historical) |
| `scripts/`, `tools/` | training launch and campaign scripts, gate-check and analysis tools |
| `config/` | vehicle parameter and MAVProxy files |
| `policies/` | the two benchmarked checkpoints, `FlightDock-S1` and `FlightDock-A2`, each with its VecNormalize file, `SHAS.txt` and `ENVELOPE.json` |
| `benchmark/` | the seed-cell tables and slot manifests of the 600-leg benchmark and the launch chain as flown (`launch_as_flown/`) |
| `scoring/` | the scoring instruments and tools that turned the flight data into the board of record |
| `results/` | results of record: per-leg board, readouts, comparative tables, tick-level summaries, derived tables, the nine figures |
| `analysis/` | the scripts that produce the results tables and figures, with their outputs |
| `training_record/` | training lineage, curriculum ladder, the record of the two benchmarked checkpoints |
| `pid_tuning/` | the PID-baseline tuning census files and the tuning-record table |
| `analysis_plans/` | the five signed analysis plans (hypotheses, statistics, decision rules) |
| `requirements-frozen.txt` | the frozen Python environment |
| `MANIFEST.sha256` | SHA-256 of every file in the repository (`sha256sum -c MANIFEST.sha256`) |

## The six arms

| arm | what flew |
|---|---|
| FlightDock-S1O | checkpoint S1 with the 0.25 m vertical setpoint offset (the policy of record) |
| FlightDock-S1 | checkpoint S1, no offset |
| FlightDock-A2O | checkpoint A2 with the offset |
| FlightDock-A2 | checkpoint A2, no offset |
| PIDO | the PID-baseline with the offset (comparator) |
| PID-baseline | the PID formation controller alone (comparator) |

Checkpoint hashes (SHA-256, also in `policies/*/SHAS.txt`):

| file | S1 | A2 |
|---|---|---|
| `model_final.zip` | `34c7f1161a84c0665000dd3960232fddba6690483cb93632e365e7b45449fb1c` | `3d1cf6cac098ef3e53fd5c0ed759d14be58448b242297bde18b68f1f7801d442` |
| `vecnormalize_final.pkl` | `d8dacc4c82e58b542e05a8bff1338f11adbb5f8b9dc207359a34ccb7a04a747c` | `eecebfee2d0dc8894c4f00d8f8b53066b114e5db5a2c4bb8cd5415220177de8f` |

`ENVELOPE.json` in each policy folder holds the residual-bound settings the checkpoint was benchmarked with; they equal the
defaults in `src/drl/config.py` except the altitude floor, 1.00 m for S1 and 0.75 m for A2, which the exam driver set for the
run. The offset arms (S1O, A2O, PIDO) additionally flew with `VERT_OFFSET_DOWN_M = 0.25`, and PIDO with the learned action
scaled to zero, both set by the exam driver.

Each arm flew one corpus of 600 legs at turbulence σ 0.25 and one at σ 0.35: ten stages (Calm Control and mean winds
0 to 8 m/s in 1 m/s steps) × 30 seeds (block 10775 to 10804) × two DHC-2 trajectories (straight, racetrack). The seed
tables are in `benchmark/seed_tables/`. Pooled statistics use the nine wind stages (540 legs); Calm Control is reported
on its own.

## Reproducing the tables and figures from the shipped results

Create the environment described in `ENVIRONMENT.md`, then run from the repository root:

```
python analysis/pvalue_bootstrap_20260922/pvalue_bootstrap.py
python analysis/results_tables/results_tables.py
python analysis/results_tables/safety_tables.py
python analysis/results_tables/why_legs_unsuccessful.py
python analysis/results_tables/s1o_vs_pid_by_stage.py
for f in analysis/verify_20260918/figures/*.py; do python "$f"; done
```

Every script reads only files in this repository, cross-checks its numbers against the results of record, and refuses
on any drift. The tables are written to `analysis/results_tables/RESULTS_TABLES.md`, `SAFETY_TABLES.md` and
`S1O_VS_PID_BY_STAGE.md` (with JSON twins); the figures to `results/derived/figures/`. The nine figures are:

- H1 Corridor p95 Margin; H2 Pooled-Wind Margin; H3 Pooled-Wind Decrease
- H1 Corridor p95 by Arm, Truth and Perception; Approach Funnel by Arm
- H1, H2 and H3 by Trajectory; Docking-Envelope Captures by Wind Stage

## Re-scoring from the flight data

1. Download the twelve corpora from the bucket into one folder, keeping the layout `<folder>/<corpus>/<cell>/<leg>/`
   (corpus names `S1O_s025`, `S1O_s035`, `S1_s025`, `S1_s035`, `A2O_s025`, `A2O_s035`, `A2_s025`, `A2_s035`,
   `PIDO_s025`, `PIDO_s035`, `PID_s025`, `PID_s035`).
2. Set `FLIGHTDOCK_CORPORA` to that folder (the analysis scripts default to `<repository>/corpora`). No shipped script reads
   the bucket at run time. One tool, `scoring/tools/wind_table.py`, reads each leg's full simulator wind witness
   (`fdm_witness.jsonl`, about 40 MB per leg); the bucket does not carry it, so that tool serves a fresh run of the
   benchmark, which writes the witness. Its outputs for the benchmarked corpora are already under `results/`.
3. Run the scoring chain into a fresh run directory: `scoring/tools/rescore_run.py --run-dir <dir>` (readouts),
   `scoring/tools/build_board.py --run-dir <dir>` (the per-leg board), `scoring/tools/ticklevel_run.py --run-dir <dir>`
   (tick-level safety files). `analysis/verify_20260918/verify_board.py` then checks a board against the shipped one.

The scoring tools and the four digest-pinned instruments under `scoring/instruments/` are shipped exactly as they ran
(their SHA-256 digests are in `results/run_20260918T2208Z/INSTRUMENT_SHAS.json`). They name the paths of the machine
they ran on: the repository at `/home/ubuntu/FlightDock` and the corpora at `/home/ubuntu/flightdock_corpora/<corpus>`.
To run them, place the files there or edit the path constants at the top of each tool; the instruments must stay
byte-identical for the digests to match. The path columns in the shipped data files (for example `WIND_PER_LEG.csv`)
carry the same prefix; replace `/home/ubuntu/flightdock_corpora/` with your corpus folder or the bucket address.

## How the corpora were flown, and re-flying them yourself

The twelve corpora were flown by the scripts under `benchmark/launch_as_flown/`, kept exactly as they ran (comments
edited only; private network addresses replaced by `<box-ip>`). They are the record of the procedure, not a turnkey
tool: every fire script gates on signed preregistration documents that are not part of this package, so run as-is they
refuse. A reader who wants to re-fly uses them as the specification and drives the lane spawner directly.

| corpora | launch folder |
|---|---|
| A2 (σ 0.25 and 0.35) | `bench600_20260902T0310Z` |
| S1O σ 0.25; S1O σ 0.35 | `cs600_20260903T0230Z`; `cs600h3_20260903T0330Z` |
| S1 and A2O (both σ) | `s1abl_20260915T0400Z` |
| PIDO (both σ) | `pido_20260916` |
| PID-baseline (both σ) | `armblad025_20260830T1600Z`, `armblad035_20260830T1600Z` |

The chain, in order: the seed plan writes the seed-cell table and the slot manifest (`armblad*_seed_plan.py`,
`armblad*_manifest.py`; the shipped copies are in `benchmark/seed_tables/`); a fire script (`*_fire.sh`) checks its gates
and starts the exam driver (`*_overlay_exam.py`, which applies the arm's knobs: the 0.25 m vertical setpoint offset for
the offset arms, a zero action scale for PIDO); the driver starts the wave (`bench600_exam_wave.py`), which deals the
600 rows across the lanes and runs `scripts/curriculum_eval_spawner.py` once per leg; after each leg
`bench600_sigma_hook.py` checks the realized turbulence against the commanded σ. The PID-baseline used its own driver and
wave launcher (`armblad*_driver.sh`, `armblad*_wave_launcher.sh`) around the same spawner. Each file states its public
interface in its header.

### Before you start: time, size and risk

- Time. One leg takes about 25 minutes of one lane. One corpus of 600 legs took about 10 hours at 24 lanes on a
  96-vCPU machine; at 4 lanes it is about 60 hours. The full benchmark is twelve corpora.
- Size. A corpus writes about 40 GB of flight records (the scoring chain reads a 12 to 14 GB subset of each). The
  simulator build, the Python environment and the policies are small by comparison.
- What runs on your machine. These scripts start and stop processes (ArduPilot SITL instances, MAVProxy, the physics
  backend, Python workers), open local network ports, write under the output folder you name, and stop the processes
  they started when they exit. Run them on a machine with no other ArduPilot or simulator work in progress.
- Warranty and liability. This software is provided as is, without warranty of any kind. You run it on your own
  equipment at your own risk; the authors accept no liability for damage, data loss, cost or downtime arising from its
  use. See `LICENSE` (Apache License 2.0, sections 7 and 8).

### Setting up the evaluation environment yourself

1. Build the simulator: the ArduPilot fork at the pinned commit (see "Simulator" and `ENVIRONMENT.md`).
2. Create the Python environment from `requirements-frozen.txt` (Python 3.12).
3. Place this repository at `/home/ubuntu/FlightDock`, the path the flown scripts name, or edit the path constants at
   the top of the scripts you use.
4. Choose the arm: the policy folder under `policies/` (the PID-baseline and PIDO use no policy), the σ (0.25 or 0.35)
   and the knobs the arm's fire script declares.
5. Choose the seed table for that σ from `benchmark/seed_tables/` and read the lane and leg arguments from the matching
   driver script; drive `scripts/curriculum_eval_spawner.py` with them (its header lists the arguments and environment
   variables).
6. Fly a handful of legs first and confirm the leg folders hold the twelve per-leg files listed under "Flight data"
   and that the σ gate passes; only then start a full corpus.
7. Score the result with the scoring chain under "Re-scoring from the flight data".

## Flight data

The bucket holds, for every leg of the twelve corpora, the twelve per-leg files the scoring chain reads (results,
metrics, σ gate, leg metadata, safety summary, manifests, run summaries, success metrics, transition log, paired
channels, flight state). Layout: `s3://flightdock-reproduction-2026/corpora/<corpus>/<cell>/<leg>/...`, the same tree
as the results files name. A SHA-256 manifest of every object is at the bucket root. The bucket is private; request
access from the repository owner.

## Training and tuning records

Training ran through `scripts/run_drl_training.sh` (curriculum and settings in `src/drl/curriculum.py` and
`src/drl/config.py`). The lineage of the two checkpoints and the curriculum ladder are in
`training_record/`; the PID-baseline tuning record and its census files are in `pid_tuning/`.

## Simulator

The flights ran on a fork of ArduPilot: `https://github.com/ARM15C/ardupilot-flightdock`, branch
`fd-tecs-speed-hook-v2`, commit `678ed4e0e1666169e91623d8398e710d5986c601` (upstream ArduPilot `d901078e29`, ArduPlane
4.7.0-dev, plus ten FlightDock commits). Build and version details are in `ENVIRONMENT.md`.

## Notes on the code

The code under `src/`, `tests/`, `scripts/`, `tools/` and `config/` is the code that flew, limited to what training, the
benchmark, scoring and analysis use; only comments were edited (working notes removed, American spelling). The analysis
scripts under `analysis/` were edited to run from repository-relative paths and to drop reads of files that are not shipped;
their outputs were regenerated from this repository and are identical, cell for cell, to the originals.

## License

Apache License 2.0; see `LICENSE`.
