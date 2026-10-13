# Gust random-draw stream census (outcome-blind), EDA-20260918

Legs read: 7200; row errors: 0; corpora: A2O_s025, A2O_s035, A2_s025, A2_s035, PIDO_s025, PIDO_s035, PID_s025, PID_s035, S1O_s025, S1O_s035, S1_s025, S1_s035

## Per corpus

```
           legs  flown_ep_not_0  flown_folder_found  no_accepted_attempt  reroll_legs
corpus                                                                               
A2O_s025    600               0                 600                    0           16
A2O_s035    600               0                 600                    0           60
A2_s025     600               0                 600                    0           22
A2_s035     600               2                 600                    0           52
PIDO_s025   600               2                 600                    0           18
PIDO_s035   600               1                 600                    0           47
PID_s025    600               0                 600                    0           21
PID_s035    600               0                 600                    0           39
S1O_s025    600               0                 600                    0           16
S1O_s035    600               0                 600                    0           51
S1_s025     600               0                 600                    0           23
S1_s035     600               0                 600                    0           53
```

## sigma s025: 540 windy keys, 479 with every arm on one stream, 61 with at least one arm on another

```
cause
timebase retry (flew episode 1)                             2
warmup re-roll: flew attempt 1, majority flew attempt 2    43
warmup re-roll: flew attempt 1, majority flew attempt 3     3
warmup re-roll: flew attempt 2, majority flew attempt 1    56
warmup re-roll: flew attempt 3, majority flew attempt 1     4
```

Odd legs by arm:
```
odd_arm
A2       9
A2O     15
PID     20
PIDO    18
S1      23
S1O     23
```

## sigma s035: 540 windy keys, 421 with every arm on one stream, 119 with at least one arm on another

```
cause
timebase retry (flew episode 1)                              3
warmup re-roll: flew attempt 1, majority flew attempt 2     58
warmup re-roll: flew attempt 1, majority flew attempt 3      2
warmup re-roll: flew attempt 2, majority flew attempt 1    123
warmup re-roll: flew attempt 3, majority flew attempt 1     10
warmup re-roll: flew attempt 3, majority flew attempt 2      2
warmup re-roll: flew attempt 4, majority flew attempt 1      1
warmup re-roll: flew attempt 4, majority flew attempt 2      2
```

Odd legs by arm:
```
odd_arm
A2      22
A2O     50
PID     24
PIDO    31
S1      38
S1O     36
```
