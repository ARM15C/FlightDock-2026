#!/usr/bin/env python3
"""the document safety set: guard fires, false and missed aborts (tolerance 1.0 s of record,
four tolerances in the document), recovery after a breakaway, vertical and longitudinal overshoot, contact-plane ticks.
Reproduction only: every number is read from the tick-level artifacts of record and cross-checked (REFUSE on drift).
Nine wind stages, 540 legs per arm and turbulence level; the Calm Control stage (S0) is excluded as in every pooled table.
Intervals: two-sided 95% seed-cluster bootstrap, 30 seeds resampled whole, 10,000 draws, the draws of record (p95_boot)."""
import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import sys, json, pathlib, numpy as np, pandas as pd
sys.path.insert(0, (_REPO + '/scoring/tools')); import p95_boot as PB
P = pathlib.Path((_REPO + '/analysis'))
RUN = pathlib.Path((_REPO + '/results/run_20260918T2208Z'))
T = RUN/'ticklevel'
SRC = {'legs': T/'ABORT_PER_LEG.csv', 'events': T/'ABORT_EVENTS.csv', 'recovery': T/'RECOVERY_SUMMARY.csv',
       'board': RUN/'board'/'BOARD_W10.csv', 'json': P/'results_tables/RESULTS_NUMBERS.json'}
TAGS = {'CSV':'S1O','S1A':'S1','A2V':'A2O','A2P':'A2','PIDO':'PIDO','ARMB':'PID'}
ORDER = ['S1O','S1','A2O','A2','PIDO','PID']
BAR, CONTACT, CG_AHEAD_OF_TIP = 2.00, 3.42, 4.31
TOLS = ['0.25','0.5','1.0','2.0']; TOL = '1.0'
def refuse(m): raise SystemExit('REFUSE: '+m)
def arm_sig(tag):
    for k,v in TAGS.items():
        if tag.startswith(k):
            s = tag[len(k):]; return v, {'025':0.25,'25':0.25,'035':0.35,'35':0.35}[s]
    refuse('unknown tag '+tag)
def add(df):
    a = df.tag.map(arm_sig); df = df.copy(); df['arm'] = [x[0] for x in a]; df['sig'] = [x[1] for x in a]
    if 'seed' not in df: df['seed'] = df.leg.str.split('_').str[1].astype(int)
    if 'wind_label' not in df: df['wind_label'] = df.leg.str.split('_').str[0]
    return df[df.wind_label != 'S0']
L = add(pd.read_csv(SRC['legs'])); E = add(pd.read_csv(SRC['events'], low_memory=False)); R = add(pd.read_csv(SRC['recovery'], low_memory=False))
B = pd.read_csv(SRC['board'], low_memory=False); B['tag'] = B.corpus; B = add(B)
J = json.load(open(SRC['json']))
SEEDS = np.sort(L.seed.unique())
if len(SEEDS) != 30: refuse('seed block is not 30 seeds')
PICKS = PB._picks(30)
def per_seed(df, col):
    return df.groupby('seed')[col].sum().reindex(SEEDS, fill_value=0).to_numpy(float)
def ratio_ci(num, den):
    """100*sum(num)/sum(den) with the seed-cluster interval; None when the denominator is zero."""
    if den.sum() == 0: return None
    n, d = num[PICKS].sum(axis=1), den[PICKS].sum(axis=1); ok = d > 0
    v = 100.0*n[ok]/d[ok]; return 100.0*num.sum()/den.sum(), float(np.percentile(v,2.5)), float(np.percentile(v,97.5))
def fr(x): return 'n/a (0 events)' if x is None else f'{x[0]:.1f}% [{x[1]:.1f}, {x[2]:.1f}]'
rows7, rows8, rowsT, rowsR, rowsG, N, STAGE = [], [], [], [], [], {}, {}
TRUE = lambda s: s.astype(str).eq('True')
# SCOPE: guard fires, breakaways and passes of the CG line are counted ONLY inside the terminal approach corridor
# (truth range <= 10 m of the boom-tip at the event). The all-state totals are still reproduced first as a check on the files.
Eall = pd.read_csv(SRC['events'], low_memory=False); Eall = Eall.assign(arm=[arm_sig(x)[0] for x in Eall.tag], sig=[arm_sig(x)[1] for x in Eall.tag])
Eall['inc'] = TRUE(Eall.inside_10m); Eall['false1'] = TRUE(Eall['is_false_tol' + TOL])
for (a_, s_, w_), g_ in Eall[Eall.inc].groupby(['arm', 'sig', 'wind_label']):
    STAGE['%s|%s|%s' % (a_, s_, w_)] = {'alt_fires': int((g_.kind == 'ceiling_fire').sum()), 'ovt_fires': int((g_.kind == 'overtake_fire').sum())}
E['inc'] = TRUE(E.inside_10m); E['false1'] = TRUE(E['is_false_tol' + TOL]); E['one'] = 1
R = R.merge(E[E.kind == 'breakaway'][['tag', 'leg', 'event_idx', 'inc']], on=['tag', 'leg', 'event_idx'], how='left', validate='one_to_one')
if R.inc.isna().any(): refuse('a recovery row has no breakaway event')
for sig in (0.25, 0.35):
    for arm in ORDER:
        l = L[(L.arm==arm)&(L.sig==sig)]; e = E[(E.arm==arm)&(E.sig==sig)]; r = R[(R.arm==arm)&(R.sig==sig)]; b = B[(B.arm==arm)&(B.sig==sig)]
        if len(l) != 540 or len(b) != 540: refuse(f'{arm} σ {sig}: {len(l)} tick-level legs, {len(b)} board legs, expected 540')
        g = J[f'GUARD_{sig}_{arm}']
        cf, of_, bk = e[e.kind=='ceiling_fire'], e[e.kind=='overtake_fire'], e[e.kind=='breakaway']
        # all-state totals reproduce the files of record
        if int(l.vertical_fires.sum()) != g['alt_fires'] or int(l.longitudinal_fires.sum()) != g['ovt_fires'] or int((l.n_breakaways>0).sum()) != g['brk_legs']:
            refuse(f'{arm} σ {sig}: all-state fires/breakaway legs do not reproduce the guard table of record')
        if len(cf) != int(l.vertical_fires.sum()) or len(of_) != int(l.longitudinal_fires.sum()) or len(bk) != int(l.n_breakaways.sum()) or len(bk) != len(r):
            refuse(f'{arm} σ {sig}: per-leg, per-event and recovery files disagree on fires or breakaways')
        if int(cf.false1.sum()) != int(l[f'vertical_false_tol{TOL}'].sum()) or int(of_.false1.sum()) != int(l[f'longitudinal_false_tol{TOL}'].sum()):
            refuse(f'{arm} σ {sig}: per-event false-abort flags do not sum to the per-leg totals')
        allst = dict(alt=len(cf), alt_false=int(cf.false1.sum()), ovt=len(of_), ovt_false=int(of_.false1.sum()), brk=len(bk))
        # corridor only from here
        cf, of_, bk, r = cf[cf.inc], of_[of_.inc], bk[bk.inc], r[r.inc.astype(bool)]
        vf, of, nb = len(cf), len(of_), len(bk); vfa, lfa = int(cf.false1.sum()), int(of_.false1.sum())
        vmi, lmi = int(l[f'vertical_missed_tol{TOL}'].sum()), int(l[f'longitudinal_missed_tol{TOL}'].sum())
        ve, le = int(l.vertical_truth_events.sum()), int(l.longitudinal_truth_events.sum())
        ps = lambda df, col: df.groupby('seed')[col].sum().reindex(SEEDS, fill_value=0).to_numpy(float)
        ci_vf = ratio_ci(ps(cf,'false1'), ps(cf,'one')); ci_lf = ratio_ci(ps(of_,'false1'), ps(of_,'one'))
        ci_lm = ratio_ci(per_seed(l,f'longitudinal_missed_tol{TOL}'), per_seed(l,'longitudinal_truth_events'))
        rows7.append({'σ': sig, 'arm': arm, 'altitude-guard fires (legs)': f"{vf} ({cf.leg.nunique()})", 'false fires': f'{vfa} of {vf}', 'false-abort rate [95% interval]': fr(ci_vf),
                      'missed': f'{vmi} of {ve}', 'overtake-guard fires (legs)': f"{of} ({of_.leg.nunique()})", 'false fires ': f'{lfa} of {of}', 'false-abort rate [95% interval] ': fr(ci_lf),
                      'missed ': f'{lmi} of {le}', 'missed-abort rate [95% interval]': fr(ci_lm)})
        r = r.assign(ub=TRUE(r.under_bar_at_exit), eg=TRUE(r.behind_meets_exit_gate), dm=TRUE(r.dwell_meets_min), one=1)
        ci_ub = ratio_ci(per_seed(r,'ub'), per_seed(r,'one')); ci_eg = ratio_ci(per_seed(r,'eg'), per_seed(r,'one'))
        rc = r[r.trigger_reason=='guard_altitude_ceiling']
        if len(rc) != vf: refuse(f'{arm} σ {sig}: corridor ceiling-triggered breakaways {len(rc)} != corridor altitude-guard fires {vf}')
        pk = rc.up_max_in_breakaway_m
        ru = e[(e.kind=='truth_overtake_run')&TRUE(e.truth_eligible)&e.inc]
        past = ru[ru.max_ahead_m > CG_AHEAD_OF_TIP]; kept = ru[ru.max_ahead_m <= CG_AHEAD_OF_TIP]
        # capture with and without a corridor breakaway (the bootstrap of ch4_numbers.boot_brk, on the corridor flag)
        blegs = set(bk.leg); bb = b.assign(ok=b.passed.astype(str).str.lower().eq('true'), brk=b.leg.isin(blegs))
        sd = bb.seed.to_numpy(); ok = bb.ok.to_numpy(bool); br = bb.brk.to_numpy(bool)
        kb = np.array([(ok&br)[sd==s].sum() for s in SEEDS], float); nbs = np.array([br[sd==s].sum() for s in SEEDS], float)
        kn = np.array([(ok&~br)[sd==s].sum() for s in SEEDS], float); nn = np.array([(~br)[sd==s].sum() for s in SEEDS], float)
        with np.errstate(invalid='ignore', divide='ignore'):
            v = 100.0*(kb[PICKS].sum(axis=1)/nbs[PICKS].sum(axis=1) - kn[PICKS].sum(axis=1)/nn[PICKS].sum(axis=1))
        rb, rn = 100.0*ok[br].mean(), 100.0*ok[~br].mean(); dlo, dhi = float(np.nanpercentile(v,2.5)), float(np.nanpercentile(v,97.5))
        vv = v[~np.isnan(v)]; p2 = float(min(1.0, 2.0*min(float((vv<=0).mean()), float((vv>=0).mean()))))   # two-sided bootstrap p, the rule of the ablation tables
        hmed = float((cf.up_m_at_event).median()) if vf else None
        rows8.append({'σ': sig, 'arm': arm, 'breakaways (legs)': f"{nb} ({len(blegs)})",
                      'exits under 2.00 m [95% interval]': f"{int(r.ub.sum())} of {nb}, {fr(ci_ub)}", 'exits at ≥ 15 m behind [95% interval]': f"{int(r.eg.sum())} of {nb}, {fr(ci_eg)}",
                      'peak height in an altitude-guard breakaway, median / p95 / max (m)': ('none' if len(pk)==0 else f'{pk.median():.2f} / {np.percentile(pk,95):.2f} / {pk.max():.2f}'),
                      'ahead of the tip on eligible corridor runs, p95 / max (m)': f'{np.percentile(kept.max_ahead_m,95):.2f} / {kept.max_ahead_m.max():.2f} (n = {len(kept)})',
                      'corridor runs past the CG line': len(past), 'legs over 3.42 m in the corridor': int((l.peak_up_m_in_window>CONTACT).sum()),
                      'capture change with a breakaway [95% interval] (pp)': f"{rb-rn:+.1f} [{dlo:+.1f}, {dhi:+.1f}]"})
        rowsG.append({'σ': sig, 'arm': arm, 'legs with an altitude-guard fire': cf.leg.nunique(), 'legs with an overtake-guard fire': of_.leg.nunique(), 'legs with a breakaway': len(blegs),
                      'captures, legs with a breakaway (%)': f'{rb:.1f}', 'captures, legs without (%)': f'{rn:.1f}', 'difference (pp) [95% interval]': f"{rb-rn:+.1f} [{dlo:+.1f}, {dhi:+.1f}]",
                      'truth height at an altitude-guard fire, median (m)': ('none' if hmed is None else f'{hmed:.2f}')})
        rowsR.append({'σ': sig, 'arm': arm, 'breakaways': nb, 'triggered by the altitude guard': len(rc), 'triggered by the overtake guard': int((r.trigger_reason=='guard_hard_overtake').sum()),
                      'dwell ≥ 2.0 s': int(r.dm.sum()), 'exit ≥ 15 m behind': int(r.eg.sum()), 'under 2.00 m at exit': int(r.ub.sum()),
                      'altitude-guard breakaways peaking above 3.42 m': int((pk>CONTACT).sum())})
        N[f'{arm}_{sig}'] = {'alt_fires': vf, 'alt_legs': int(cf.leg.nunique()), 'alt_false': vfa, 'alt_elig': ve, 'alt_missed': vmi, 'ovt_fires': of, 'ovt_legs': int(of_.leg.nunique()), 'ovt_false': lfa, 'ovt_elig': le, 'ovt_missed': lmi,
                             'breakaways': nb, 'brk_legs': len(blegs), 'under_bar': int(r.ub.sum()), 'exit_gate': int(r.eg.sum()), 'pk_max': (None if len(pk)==0 else float(pk.max())), 'pk_p95': (None if len(pk)==0 else float(np.percentile(pk,95))),
                             'ahead_p95': float(np.percentile(kept.max_ahead_m,95)), 'ahead_max': float(kept.max_ahead_m.max()), 'runs_past_cg': len(past), 'past_cg_legs': list(past.leg),
                             'contact_corridor': int((l.peak_up_m_in_window>CONTACT).sum()), 'cap_brk': rb, 'cap_nobrk': rn, 'diff': rb-rn, 'lo': dlo, 'hi': dhi, 'p': p2, 'fire_height_med': hmed, 'all_states': allst}
# survey of 2026-10-02 must reproduce
chk = N['S1O_0.25']; ca = chk['all_states']
if (ca['alt'],ca['alt_false'],chk['alt_elig'],chk['alt_missed'],ca['ovt'],ca['ovt_false'],chk['ovt_elig'],chk['ovt_missed'],ca['brk']) != (24,4,3,1,171,114,172,140,195):
    refuse('S1O σ 0.25 does not reproduce the 2026-10-02 survey row')
if sum(v['contact_corridor'] for v in N.values()) != 0: refuse('a leg crossed 3.42 m inside the corridor: the chapter sentence does not stand')
def md(rows):
    d = pd.DataFrame(rows); h = '| ' + ' | '.join(str(c).strip() for c in d.columns) + ' |'
    return '\n'.join([h, '|' + '---|'*len(d.columns)] + ['| ' + ' | '.join('' if v is None else str(v) for v in r) + ' |' for r in d.itertuples(index=False)])
SCOPE = 'Counted inside the terminal approach corridor only: the truth range to the boom-tip was ≤ 10 m when the guard fired or the breakaway began. '
out = ['# Safety tables, corridor only (nine wind stages, 540 legs per arm and σ; tolerance of record 1.0 s)', '',
 '## S1. Guard fires and false and missed aborts (tolerance 1.0 s)', '', md(rows7), '',
 'Note. ' + SCOPE + 'A false abort is a fire with no truth-channel crossing of that guard\'s line within 1.0 s. A missed abort is a truth-channel crossing inside the corridor, with the guard armed, that drew no fire within 1.0 s. Intervals: two-sided 95% seed-cluster bootstrap.', '',
 '## S2. Recovery after a breakaway and stopping distance', '', md(rows8), '',
 'Note. ' + SCOPE + 'Capture change: captures on legs with a corridor breakaway minus legs without, seed-cluster bootstrap.', '',
 '## S3. Guard record and breakaway cost', '', md(rowsG), '',
 'Note. ' + SCOPE, '']
json.dump(STAGE, open(P/'results_tables/SAFETY_BY_STAGE.json', 'w'), indent=0)
(P/'results_tables/SAFETY_TABLES.md').write_text('\n'.join(out), encoding='utf-8')
json.dump(N, open(P/'results_tables/SAFETY_TABLES.json','w'), indent=1)
print('wrote results_tables/SAFETY_TABLES.md and .json; 12 corpora; checks passed')
