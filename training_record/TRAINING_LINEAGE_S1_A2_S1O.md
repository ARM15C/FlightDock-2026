# Table A-23. Training Lineage: S1, A2, and S1O



Part 1. Shared Path (S1 and A2)

|Stages|Wind(m/s)|σ|Timesteps|
|-|-|-|-|
|0|0.0|0|154,923|
|1|0.0|0.10|48,050|
|2 to 3|0.5 to 1.0|0.10|72,000|
|4|1.5|0.10|74,785|
|5|2.0|0.10|77,547|
|6 to 9|2.5 to 4.0|0.10|168,411|
|9|4.0|0.10|56,207|
|10 to 12|4.5 to 5.5|0.10|210,426|
|13|6.0|0.10|97,269|
|Subtotal 0–13|0 to 6.0|0 to 0.10|959,618|
|14|6.0|0.15|50,217|
|Shared snapshot|—|—|1,009,835|

Part 2. Legs After the Shared Snapshot

|Policy|Leg|Wind (m/s)|σ|Timesteps|Floor (m)|Buffer and Critic|
|-|-|-|-|-|-|-|
|S1|Leg 1|6.0|0.15|96,000|1.00|New|
|S1|Leg 2|6.0|0.20|48,000|1.00|Carried|
|S1|Final|6.0|0.25|46,000|1.00|Carried|
|A2|Leg 1|6.0|0.15|96,000|1.00|New|
|A2|Leg 2|6.0|0.20|48,000|1.00|Carried|
|A2|Final|6.0|0.20|48,000|0.75|New|

Part 3. Additional Continuation Training

|Path|Stages|Wind (m/s)|σ|Timesteps|Checkpoints|
|-|-|-|-|-|-|
|First climb from the shared snapshot|15 to 20|6.0 to 8.0|0.20 to 0.25|333,609|gate\_W16, gate\_D2a|
|A2, strong-wind continuation|23, 26, 33|6.0, 7.5, 8.0|0.25|138,000|66565ea7, 5a167282, 56dd4146|
|A2, calm then 8 m/s|34, 35|0.0, 8.0|off|46,000|fd3a338a|
|S1, continuation|26|7.5|0.25|46,000|afc1f037|

Part 4. Totals and Settings

|Policy|Timesteps After Snapshot|Timesteps Total|Max Wind (m/s)|Max σ|Ramp Weight|Climb-Rate Weight|Training Seed|
|-|-|-|-|-|-|-|-|
|S1|190,000|1,199,835|7.5|0.25|0.40|1.0|42722|
|A2|192,000|1,201,835|8.0|0.25|0.20|0.5|42722|

Part 5. S1O Offset Pilot Sweep

|Offset(m)|Legs|Held-Height Δ vs. Two Controls(m)|p95 Peak Upward,≤ 3 m(m)|Stages p95 > 2.00 m|> 2.00 m + Overtake Legs|Captures|Outcome|
|-|-|-|-|-|-|-|-|
|0|180|-|2.063|4/5|12(6.7%) + 3(1.7%)|132 of 180,73.3%|control|
|0.10|90|-0.08 / -0.11|1.967|1/5|4(4.4%) + 2(2.2%)|59 of 90,65.6%|not selected|
|0.25|90|-0.20 / -0.23|1.816|0/5|2(2.2%) + 1(1.1%)|66 of 90,73.3%|selected|
|0.50|90|-0.26 / -0.32|1.760|1/5|1(1.1%) + 1(1.1%)|55 of 90,61.1%|discarded|

Note. S1 and A2 share one training path to stage 14 and are parallel lineages after it, not parent and child. S1O and A2O are the S1 and A2 checkpoints flown with the 0.25 m offset; neither was retrained, and no offset was present in training. Floor = the floor of the bound on the altitude axis. Buffer and Critic: New = a fresh replay buffer and a reinitialized critic for that leg. Ramp and Climb-Rate Weight = the ceiling-shaping weights. Part 4 lists the training that reached 8.0 m/s mean wind. It is part of the campaign but Part 5: S1 checkpoint, σ = 0.25, a 90-leg table on seeds 10553–10594, separate from the benchmark block; control = no offset, flown twice; Held-Height Δ = change in the median held height against each of the two control flights; > 2.00 m + Overtake Legs = legs (share of the row’s legs); scored within 3.0 m of the boom-tip, the corridor in use at the time, so its p95 is not comparable with Chapter 4. The rule and the three values were registered before the sweep flew; 0.25 m was fixed before the benchmark.

