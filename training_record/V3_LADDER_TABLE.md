# V3 CURRICULUM LADDER — GENERATED, DO NOT HAND-EDIT

Source of truth: `src/drl/config.py` (`\_V3\_LADDER\_SPEC` and the tables derived
from it). Regenerate with `python tools/render\_ladder\_table.py --write`; verify
with `--check` (rc=1 when stale). `tests/unit/test\_ladder\_doc\_in\_sync.py` fails
the suite if this file drifts from the code.

<!-- BEGIN GENERATED: v3-ladder -->

|Stage|Label|Wind m/s|σ|Nominal|Cap|Gate|Anchor|
|-|-|-|-|-|-|-|-|
|0|S0|0.0|—|60,000|120,000|anchor30|**ANCHOR**|
|1|W0|0.0|0.10|36,000|96,000|mid12||
|2|W1|0.5|0.10|36,000|96,000|mid12||
|3|W2|1.0|0.10|36,000|96,000|mid12||
|4|W3|1.5|0.10|36,000|96,000|mid12||
|5|W4|2.0|0.10|36,000|96,000|mid12||
|6|W5|2.5|0.10|36,000|96,000|mid12||
|7|W6|3.0|0.10|36,000|96,000|mid12||
|8|W7|3.5|0.10|36,000|96,000|mid12||
|9|W8|4.0|0.10|36,000|96,000|anchor30|**ANCHOR**|
|10|W9|4.5|0.10|48,000|96,000|mid12||
|11|W10|5.0|0.10|48,000|96,000|mid12||
|12|W11|5.5|0.10|48,000|96,000|mid12||
|13|W12|6.0|0.10|48,000|96,000|mid12||
|14|D1a|6.0|0.15|48,000|96,000|mid12||
|15|D1|6.0|0.20|48,000|96,000|mid12||
|16|W13|6.5|0.20|48,000|96,000|mid12||
|17|W14|7.0|0.20|48,000|96,000|mid12||
|18|W15|7.5|0.20|48,000|96,000|mid12||
|19|W16|8.0|0.20|48,000|96,000|mid12||
|20|D2a|8.0|0.25|48,000|96,000|mid12||
|21|D2b|8.0|0.30|48,000|96,000|mid12||
|22|C1|8.0|0.30|160,000|320,000|anchor30|**ANCHOR**|
|23|D1b|6.0|0.25|48,000|72,000|mid12||
|24|R13|6.5|0.25|48,000|72,000|mid12||
|25|R14|7.0|0.25|48,000|72,000|mid12||
|26|R15|7.5|0.25|48,000|72,000|mid12||
|27|L4|4.0|0.25|48,000|72,000|mid12||
|28|L2|2.0|0.25|48,000|72,000|mid12||
|29|L0|0.0|0.25|48,000|72,000|mid12||
|30|P1a|6.0|0.15|48,000|72,000|mid12||
|31|P1|6.0|0.20|48,000|72,000|mid12||
|32|L3|3.0|0.25|48,000|72,000|mid12||
|33|P2a|8.0|0.25|48,000|72,000|mid12||
|34|A2A0|0.0|—|48,000|72,000|mid12||
|35|A2A8|8.0|—|48,000|72,000|mid12||
|36|A2A10|8.0|0.10|48,000|72,000|mid12||
|37|A2A15|8.0|0.15|48,000|72,000|mid12||
|38|A2A20|8.0|0.20|48,000|72,000|mid12||
|39|A2A25|8.0|0.25|48,000|72,000|mid12||

* Rungs: **40** (indices 0–39) · anchors **(0, 9, 22)** · C1 index **22**
* Learner pool (BINDING + INCLUSIVE): **1,796,000** transitions
* Nominal sum **1,936,000** · cap sum **3,680,000** (cap sum is DIAGNOSTIC ONLY — the pool is the single binding constraint)

<!-- END GENERATED: v3-ladder -->

