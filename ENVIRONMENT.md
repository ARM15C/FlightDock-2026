# Environment

## Machines of record

| role | machine |
|---|---|
| Development and analysis | Windows 11 laptop (16 cores, 110 GB memory), WSL2 running Ubuntu 22.04.5 LTS |
| Training | AWS EC2 c7i.4xlarge (16 vCPUs), Ubuntu Server 24.04 LTS |
| Benchmark (600-leg corpora) | AWS EC2 c7i.24xlarge (96 vCPUs, 24 simulators in parallel), Ubuntu Server 24.04 LTS |

## Python

Python 3.12. Create a virtual environment and install the frozen set:

```
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements-frozen.txt
```

Key packages: stable_baselines3 2.3.2, gymnasium 0.29.1, torch 2.2.0+cpu (CPU only), pymavlink 2.4.49,
MAVProxy 1.8.74, numpy 1.26.4, pandas 3.0.1, scipy 1.17.1, statsmodels 0.15.0, matplotlib 3.10.8. The analysis
scripts need only numpy, pandas, scipy, statsmodels and matplotlib; the training and benchmark scripts need the full set
and the simulator.

## Simulator

ArduPilot fork `https://github.com/ARM15C/ardupilot-flightdock`, branch `fd-tecs-speed-hook-v2`, commit
`678ed4e0e1666169e91623d8398e710d5986c601` (ArduPlane 4.7.0-dev lineage). Build SITL ArduPlane with the standard
ArduPilot toolchain (`./waf configure --board sitl && ./waf plane`). Vehicle parameters are in `config/`;
MAVProxy startup scripts in `config/*.scr`. The aircraft dynamics are supplied through ArduPilot's JSON SITL backend
by `tools/json_fdm/` (`SIMPLANE_PORT_SPEC.md` gives the port and message contract).

## Environment variables used by the analysis and scoring layer

| variable | meaning | default |
|---|---|---|
| `FLIGHTDOCK_CORPORA` | folder holding the twelve corpora (`<folder>/<corpus>/<cell>/<leg>/`) | `<repository>/corpora` |
| `FLIGHTDOCK_BUCKET` | address of the flight-data bucket, kept by `scoring/tools/wind_table.py` for a streaming path it does not use; no shipped script reads the bucket at run time | `s3://flightdock-reproduction-2026/` |

## Sizes and times

| item | measured |
|---|---|
| Repository | about 70 MB |
| Flight data in the bucket (twelve per-leg files per leg, 7,200 legs) | about 155 GB |
| Flying one corpus (600 legs, simulation) on the benchmark machine, 24 lanes on 96 vCPUs | about 10 hours (about 25 lane-minutes per leg; about 60 hours at 4 lanes) |
| Re-scoring one corpus from its recorded legs (no simulation) on the development machine | 40 to 45 minutes each; the twelve ran in parallel on a 16-core laptop with 110 GB of memory |
| Rebuilding every table and figure from the shipped results | a few minutes |
