#!/usr/bin/env python3
"""Why legs were unsuccessful, with shares and intervals, and the the document text built on the operator's
earlier paragraphs (the funnel table, the funnel figure, the one-reason-per-leg table).
Classes are the classification of results_tables.py (function classify), reproduced here from the board; REFUSES unless every
count equals the machine table T11 (results_tables/RESULTS_NUMBERS.json). Shares are of the 540 wind-stage legs per arm and turbulence level;
intervals are the two-sided 95% seed-cluster bootstrap (30 seeds resampled whole, the draws of p95_boot)."""
import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import ast, json, re, sys, pathlib, numpy as np, pandas as pd
sys.path.insert(0, (_REPO + '/scoring/tools')); import p95_boot as PB
P = pathlib.Path((_REPO + '/analysis'))
B = pd.read_csv((_REPO + '/results/run_20260918T2208Z/board/BOARD_W10.csv'), low_memory=False)
J = json.load(open(P/'results_tables/RESULTS_NUMBERS.json'))
def refuse(m): raise SystemExit('REFUSE (whylegs): ' + m)
TAG = {'CSV025': ('S1O', 0.25), 'S1A25': ('S1', 0.25), 'A2V25': ('A2O', 0.25), 'A2P025': ('A2', 0.25), 'PIDO25': ('PIDO', 0.25), 'ARMB025': ('PID', 0.25),
       'CSV035': ('S1O', 0.35), 'S1A35': ('S1', 0.35), 'A2V35': ('A2O', 0.35), 'A2P035': ('A2', 0.35), 'PIDO35': ('PIDO', 0.35), 'ARMB035': ('PID', 0.35)}
tr = lambda c: B[c].astype(str).str.lower().eq('true')
B['ok'] = tr('passed'); B['ot'] = tr('real_overtake'); B = B[B.wind_label != 'S0'].copy()
B['arm'] = [TAG[c][0] for c in B.corpus]; B['sig'] = [TAG[c][1] for c in B.corpus]
KEYS = ['never', 'pastcg', 'over', 'noentry', 'dwell']
MACHINE = {'never': 'never came within 10 m', 'pastcg': 'overtook the DHC-2 in close formation', 'over': 'climbed above the 2.00 m bar (exclusion-zone violation)',
           'noentry': 'came within 10 m, never inside the envelope on both channels', 'dwell': 'reached the envelope, held it under 3.0 s'}
def classify(r):
    if r.ok: return 'capture'
    reasons = ast.literal_eval(r.fail_reasons)
    if pd.isna(r.win): return 'never'
    if r.ot: return 'pastcg'
    if any(x.startswith('truth_ceiling_bust') for x in reasons): return 'over'
    if any(x.startswith('intersection_dwell_fail:best=0.00s') for x in reasons): return 'noentry'
    if any(x.startswith('intersection_dwell_fail') for x in reasons): return 'dwell'
    refuse('unclassified leg %s %s: %s' % (r.corpus, r.leg, reasons))
B['cls'] = B.apply(classify, axis=1)
SEEDS = np.arange(10775, 10805); PICKS = PB._picks(30); N = {}
ref = {(r['σ'], r['arm']): r for r in J['T11']}
for (arm, sig), g in B.groupby(['arm', 'sig']):
    if len(g) != 540 or sorted(g.seed.unique()) != list(SEEDS): refuse('%s σ %s is not 540 legs on the seed block' % (arm, sig))
    d = {}
    for k in KEYS + ['unsuccessful']:
        flag = (~g.ok) if k == 'unsuccessful' else (g.cls == k)
        ks = flag.groupby(g.seed).sum().reindex(SEEDS).to_numpy(float); ns = flag.groupby(g.seed).size().reindex(SEEDS).to_numpy(float)
        v = 100.0 * ks[PICKS].sum(axis=1) / ns[PICKS].sum(axis=1)
        d[k] = {'k': int(flag.sum()), 'pct': 100.0 * float(flag.mean()), 'lo': float(np.percentile(v, 2.5)), 'hi': float(np.percentile(v, 97.5))}
        want = ref[(sig, arm)]['unsuccessful legs (of 540)'] if k == 'unsuccessful' else ref[(sig, arm)][MACHINE[k]]
        if d[k]['k'] != int(want): refuse('%s σ %s class %s: %d here, %s in the machine table' % (arm, sig, k, d[k]['k'], want))
    N['%s_%s' % (arm, sig)] = d
json.dump(N, open(P/'results_tables/WHY_LEGS_UNSUCCESSFUL.json', 'w'), indent=1)
print('wrote WHY_LEGS_UNSUCCESSFUL.json')
