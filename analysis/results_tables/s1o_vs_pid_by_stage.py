#!/usr/bin/env python3
"""S1O against the PID-baseline at each stage, both turbulence levels.
Paired within seed on (stage, seed, trajectory); two-sided 95% seed-cluster bootstrap, 30 seeds resampled whole, the draws of record.
REFUSES unless the stage counts reproduce DOCKING_ENVELOPE_CAPTURES_BY_WIND_STAGE_SIGMA_*.csv and the pooled row reproduces the H2 statistic."""
import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import sys, json, re, pathlib, numpy as np, pandas as pd
sys.path.insert(0, (_REPO + '/scoring/tools')); import p95_boot as PB
P = pathlib.Path((_REPO + '/analysis'))
BOARD = (_REPO + '/results/run_20260918T2208Z/board/BOARD_W10.csv')
FIG = (_REPO + '/results/derived/figures/DOCKING_ENVELOPE_CAPTURES_BY_WIND_STAGE_SIGMA_%s.csv')
B = pd.read_csv(BOARD, low_memory=False); B['cap'] = B.passed.astype(str).str.lower().eq('true').astype(int)
J = json.load(open(P/'results_tables/RESULTS_NUMBERS.json'))
TAG = {0.25: ('CSV025', 'ARMB025'), 0.35: ('CSV035', 'ARMB035')}
STAGES = [('S0', 'Calm Control')] + [('W%d' % (2*k), '%d m/s' % k) for k in range(9)]
PICKS = PB._picks(30)
def refuse(m): raise SystemExit('REFUSE: ' + m)
rows, N = [], {}
for sig in (0.25, 0.35):
    a, b = (B[B.corpus == t] for t in TAG[sig]); ref = pd.read_csv(FIG % ('025' if sig == 0.25 else '035'))
    m = a[['wind_label', 'seed', 'trajectory', 'cap']].merge(b[['wind_label', 'seed', 'trajectory', 'cap']], on=['wind_label', 'seed', 'trajectory'], suffixes=('_s', '_p'))
    if len(m) != 600: refuse('pairs != 600 at σ %s' % sig)
    m['d'] = m.cap_s - m.cap_p; seeds = np.sort(m.seed.unique())
    def stat(g):
        d = g.groupby('seed').d.sum().reindex(seeds, fill_value=0).to_numpy(float); n = g.groupby('seed').d.size().reindex(seeds, fill_value=0).to_numpy(float)
        v = 100.0*d[PICKS].sum(axis=1)/n[PICKS].sum(axis=1)
        return 100.0*d.sum()/n.sum(), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    for lab, name in STAGES + [('POOL', 'Nine wind stages pooled')]:
        g = m[m.wind_label != 'S0'] if lab == 'POOL' else m[m.wind_label == lab]
        ks, kp, n = int(g.cap_s.sum()), int(g.cap_p.sum()), len(g); d, lo, hi = stat(g)
        if lab != 'POOL':
            for arm, k in (('S1O', ks), ('PID', kp)):
                cell = ref[(ref.stage == name) & (ref.arm == arm)]['docking-envelope captures'].iloc[0]
                if int(re.match(r'(\d+) of', cell).group(1)) != k: refuse('%s %s σ %s does not reproduce the by-stage csv' % (arm, name, sig))
        elif sig == 0.25 and (abs(round(d, 2) - J['H2_S1O']['diff']) > 1e-9 or abs(round(lo, 2) - J['H2_S1O']['lo']) > 1e-9 or abs(round(hi, 2) - J['H2_S1O']['hi']) > 1e-9):
            refuse('pooled σ 0.25 row does not reproduce the H2 statistic of record')
        rows.append({'σ': sig, 'stage': name, 'S1O captures': '%d of %d' % (ks, n), 'PID-baseline captures': '%d of %d' % (kp, n),
                     'same-seed difference, S1O minus PID-baseline [95% interval] (pp)': '%+.1f [%+.1f, %+.1f]' % (d, lo, hi), 'interval excludes zero': 'yes' if (lo > 0 or hi < 0) else 'no'})
        N['%s|%s' % (sig, name)] = {'ks': ks, 'kp': kp, 'n': int(n), 'd': float(d), 'lo': lo, 'hi': hi}
    w = [N['%s|%d m/s' % (sig, k)] for k in range(9)]
    N['%s|summary' % sig] = {'stages_positive': int(sum(x['d'] > 0 for x in w)), 'stages_interval_above_zero': int(sum(x['lo'] > 0 for x in w)), 'min_d': float(min(x['d'] for x in w)), 'max_d': float(max(x['d'] for x in w))}
d = pd.DataFrame(rows); h = '| ' + ' | '.join(d.columns) + ' |'
out = ['# S1O against the PID-baseline by stage, both turbulence levels (amplifying; decides no hypothesis)', '', h, '|' + '---|'*len(d.columns)] + ['| ' + ' | '.join(str(v) for v in r) + ' |' for r in d.itertuples(index=False)]
out += ['', 'Note. 60 same-seed pairs per stage (30 seeds, two trajectories), 540 in the pool. Difference: S1O captures minus PID-baseline captures on the same (stage, seed, trajectory) legs. Interval: two-sided 95% seed-cluster bootstrap (30 seeds resampled whole, 10,000 draws, fixed seed). A 60-pair stage carries a wide interval; the pooled row at σ = 0.25 is the H2 statistic and the only graded row. The σ = 0.35 rows are not graded. Calm Control (no wind, no turbulence) enters no pool.', '', 'Summary: ' + json.dumps({k: v for k, v in N.items() if k.endswith('summary')})]
(P/'results_tables/S1O_VS_PID_BY_STAGE.md').write_text('\n'.join(out), encoding='utf-8'); json.dump(N, open(P/'results_tables/S1O_VS_PID_BY_STAGE.json', 'w'), indent=1)
print('\n'.join(out[4:]))
