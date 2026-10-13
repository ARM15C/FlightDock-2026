#!/usr/bin/env python3
"""Results numbers of record, computed from files.
Writes results_tables/RESULTS_TABLES.md (every table with its source) and results_tables/RESULTS_NUMBERS.json (values the text uses).
Read-only on every source. Refuses if a cross-check fails."""
import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import ast, json, re, sys, pathlib, datetime
import numpy as np, pandas as pd
sys.path.insert(0, (_REPO + '/scoring/tools')); import p95_boot as PB   # seed-cluster bootstrap draws of record
import hypothesis_verdicts as V; H1_BAR_M, CONTACT_M = float(V.BAR), float(V.CONTACT)   # 2.00 m guard trip / H1 bar; 3.42 m airframe-contact plane (verdict tool, geometry of record)

P = pathlib.Path((_REPO + '/analysis'))
R = pathlib.Path((_REPO + '/results'))
O = R/'derived'
B18 = R/'run_20260918T2208Z'; B20 = R/'run_20260920T1939Z'
SRC = {}
def src(tag, path): SRC[tag] = str(path); return path
ARM = {'A2P':'A2','A2V':'A2O','S1A':'S1','CSV':'S1O','ARMB':'PID','PIDO':'PIDO'}
def arm_of(corpus):
    for k in sorted(ARM, key=len, reverse=True):
        if str(corpus).startswith(k): return ARM[k]
    raise KeyError(corpus)
def sig_of(corpus): return 0.25 if corpus.endswith('25') else 0.35
LONG = {'A2':'FlightDock-A2','A2O':'FlightDock-A2O','S1':'FlightDock-S1','S1O':'FlightDock-S1O','PID':'PID-baseline','PIDO':'PIDO'}
ORDER = ['S1O','S1','A2O','A2','PIDO','PID']
CONFIG = {'S1O':'S1 with the 0.25 m offset','S1':'S1, no offset','A2O':'A2 with the 0.25 m offset','A2':'A2, no offset','PIDO':'PID-baseline with the 0.25 m offset','PID':'PID-baseline, no offset'}   
def md(df): return '| ' + ' | '.join(df.columns) + ' |\n|' + '---|'*len(df.columns) + '\n' + '\n'.join('| ' + ' | '.join('' if (isinstance(v,float) and np.isnan(v)) else str(v) for v in r) + ' |' for r in df.itertuples(index=False))
out = []; J = {}
def emit(tid, title, df, source, note=None):
    out.append(f"### {tid}. {title}\n\n{md(df)}\n" + (f"\nNote. {note}\n" if note else ''))

# ---------- board ----------
bd = pd.read_csv(src('board', B18/'board/BOARD_W10.csv'), low_memory=False)
assert len(bd)==7200 and bd.corpus.nunique()==12
for c in ('gradable','clean','passed','wind_rung','sw','exceeds_2m','real_overtake'):
    bd[c] = bd[c].astype(str).str.lower().eq('true')
bd['arm'] = bd.corpus.map(arm_of); bd['sig'] = bd.corpus.map(sig_of)
bd['stage'] = bd.cell.str.replace(r'^B\d\d-','',regex=True)
# T1 corpus accounting
rows=[]
for (sig,arm),g in bd.groupby(['sig','arm']):
    rows.append({'σ':sig,'arm':arm,'planned':600,'flown':len(g),'gradable':int(g.gradable.sum()),'clean':int(g.clean.sum()),
                 'fault-flagged, kept':int((g.fault_class=='FAULT').sum()),'quarantined':600-int(g.gradable.sum()),
                 'worst-stage attrition':f"{int(g.groupby('stage').gradable.apply(lambda s: int((~s).sum())).max())} of 60 (bar: more than 9)",'σ assertion failed on one axis, legs kept':int((g.sigma_gate_state=='ASSERT_FAIL').sum()),
                 'seeds':int(g.seed.nunique()),'seed block':f"{g.seed.min()}-{g.seed.max()}"})
T1 = pd.DataFrame(rows); T1['o']=T1.arm.map(ORDER.index); T1=T1.sort_values(['σ','o']).drop(columns='o')
assert (T1.gradable==600).all() and (T1.quarantined==0).all() and (T1.flown==600).all() and (T1.seeds==30).all() and (T1['seed block']=='10775-10804').all()
T1e = T1.drop(columns=['planned','flown','gradable','quarantined','seeds','seed block']).rename(columns={'clean':'clean legs of 600'})   # package 10: the constant columns are one sentence in the note
J['legs_total']=int(len(bd)); J['fault_legs']=int((bd.fault_class=='FAULT').sum()); J['sigma_assert_fail']=int((bd.sigma_gate_state=='ASSERT_FAIL').sum())
fl = bd[bd.fault_class=='FAULT'][['corpus','leg']]; J['fault_legs_list']=[f"{arm_of(c)} σ {sig_of(c)} {l}" for c,l in fl.itertuples(index=False)]
emit('T1','Benchmark corpus accounting, six arms, two turbulence levels (600 legs each; 10 stages × 30 seeds × 2 trajectories)',T1e,SRC['board'],
     'Every arm flew 600 of 600 legs on the same 30 seeds (block 10775-10804); every leg was gradable and none was quarantined. Attrition bar: the document, a stage is INCONCLUSIVE when more than 15% of its 60 legs (more than nine) fail. No stage lost a leg. The σ assertion checks the realized gust intensity on each axis against the commanded value; legs that failed it on one axis stay in every denominator.')
# T2 realized sigma per corpus (median of the per-leg per-axis realized sigma, windy stages)
def ned(s):
    d = ast.literal_eval(s); return d.get('ned')
def band(s):
    return ast.literal_eval(s).get('band')
# the gate's band is read from the legs, never typed: one value per commanded sigma, and the ratio band / sigma must be the same at both
_b = bd[bd.stage!='S0'].assign(band=bd.sigma_gate.map(band)).groupby('sig').band.agg(['min','max','count'])
if not (_b['min']==_b['max']).all(): raise SystemExit('REFUSE: more than one band value at a commanded sigma')
BAND = {float(k): float(v) for k, v in _b['min'].items()}
_r = {k: round(v / k, 2) for k, v in BAND.items()}
if len(set(_r.values())) != 1: raise SystemExit('REFUSE: band / sigma differs between sigmas %s' % _r)
BAND_RATIO = list(_r.values())[0]
BAND_LABEL = 'band ' + ' and '.join('±%.2f m/s at σ %.2f' % (BAND[k], k) for k in sorted(BAND)) + ' (the gate\'s own band, %.2f σ)' % BAND_RATIO
J['sigma_band']=BAND; J['sigma_band_ratio']=BAND_RATIO
rows=[]
for (sig,arm),g in bd[bd.stage!='S0'].groupby(['sig','arm']):
    arr = np.array([n for n in g.sigma_gate.map(ned) if n is not None], dtype=float)
    rows.append({'σ commanded':sig,'arm':arm,'legs with a measured σ':len(arr),'realized σ, median, north / east / down (m/s)':f"{np.median(arr[:,0]):.3f} / {np.median(arr[:,1]):.3f} / {np.median(arr[:,2]):.3f}",
                 'legs outside the band on one axis':int((g.sigma_gate_state=='ASSERT_FAIL').sum())})
T2=pd.DataFrame(rows); T2['o']=T2.arm.map(ORDER.index); T2=T2.sort_values(['σ commanded','o']).drop(columns='o')
emit('T2','Realized turbulence intensity by arm and commanded σ, nine wind stages (per-leg values from the σ gate; ' + BAND_LABEL + ')',T2,SRC['board'],
     'Each leg\'s σ gate records the realized gust standard deviation on the three axes from the FDM\'s gust record. The commanded mean wind matched the manifest on every leg (wind_check column).')
# ---------- hypotheses ----------
pv = pd.read_csv(src('pv', P/'pvalue_bootstrap_20260922/PVALUE_BOOTSTRAP_20260922.csv'))
pv['arm']=pv.arm.replace({'PID-baseline':'PID'})
abl = pd.read_csv(src('abl', B20/'verdicts/ABLATION_TABLE.csv')); abl['arm']=abl.corpus.map(arm_of)
h1tp = pd.read_csv(src('h1tp', O/'HYPOTHESIS_VERDICTS_H1_TRUTH_AND_PERCEPTION.csv'))
h1tp['arm']=h1tp.arm.str.replace(r' \(.*\)','',regex=True).replace({'FlightDock-A2':'A2','FlightDock-A2O':'A2O','FlightDock-S1':'S1','FlightDock-S1O':'S1O','PID-baseline':'PID'})
h2a = pd.read_csv(src('h2a', P/'whatif_20260920/H2_ALTERNATIVES_v4_20260920.csv')); h3a = pd.read_csv(src('h3a', P/'whatif_20260920/H3_ALTERNATIVES_v4_20260920.csv'))
def short(n): return {'FlightDock-A2 (no offset)':'A2','FlightDock-A2O (+0.25 m offset)':'A2O','FlightDock-S1 (no offset)':'S1','FlightDock-S1O (+0.25 m offset)':'S1O','PID-baseline':'PID','PIDO (PID-baseline, +0.25 m offset)':'PIDO'}[n]
h2a['arm']=h2a.arm.map(short); h3a['arm']=h3a.arm.map(short)
def verdict(h, r):
    if h=='H1': return 'H1 met' if r.ci95_hi < r.bar else 'H1 not met'
    if h=='H2': return 'H2 met' if r.ci95_lo > r.bar else 'H2 not met'
    return 'H3 met' if r.ci95_hi < r.bar else 'H3 not met'
# T3 H1
rows=[]
for arm in ORDER:
    r = pv[(pv.hypothesis=='H1')&(pv.arm==arm)].iloc[0]; a = abl[(abl.arm==arm)&(abl.sigma==0.25)].iloc[0]; t = h1tp[h1tp.arm==arm].iloc[0]
    assert abs(a.p95_corridor_m - r.value) < 5e-4
    w = bd[(bd.sig==0.25)&bd.wind_rung&(bd.arm==arm)].win.dropna()   # per-leg corridor peaks, truth channel
    if len(w)!=int(a.entered) or int((w>H1_BAR_M).sum())!=int(a.legs_over_bar) or abs(float(np.percentile(w,95))-float(r.value))>5e-4: raise SystemExit(f'REFUSE: H1 corridor legs for {arm} do not reproduce the ablation summary (n, legs over the bar, p95)')
    worst=float(w.max())
    if worst>=CONTACT_M: raise SystemExit(f'REFUSE: {arm} has a corridor peak at or above the contact plane; the clearance columns assume none')
    over_pct=100*int(a.legs_over_bar)/int(a.entered)
    rows.append({'arm':arm,'configuration':CONFIG[arm],'legs in the corridor (n)':int(a.entered),'95th-percentile peak height above the boom-tip (m)':f"{r.value:.3f}",
                 'legs over 2.00 m (% of n)':f"{int(a.legs_over_bar)} ({over_pct:.1f}%)",'95% interval (m)':f"[{r.ci95_lo:.3f}, {r.ci95_hi:.3f}]",
                 'clearance below the contact plane at the p95 (m)':f"{CONTACT_M-float(r.value):.2f}",'clearance below the contact plane, worst leg (m)':f"{CONTACT_M-worst:.2f}",
                 'p':f"{r.boot_p_one_sided:.4f}",'verdict':verdict('H1',r) if bool(r.judged) else 'comparator (not graded)'})
    J[f'H1_{arm}']={'p95':float(r.value),'lo':float(r.ci95_lo),'hi':float(r.ci95_hi),'p':float(r.boot_p_one_sided),'null_resamples':int(round(float(r.boot_p_one_sided)*10000)),'entered':int(a.entered),'over':int(a.legs_over_bar),'over_pct':over_pct,'stages_over':int(a.stages_over),'worst':worst,'clear_p95':CONTACT_M-float(r.value),'clear_worst':CONTACT_M-worst,'verdict':rows[-1]['verdict'],'perc_p95':float(str(t['corridor p95, perception (m) - not graded']).split()[0]),'perc_over':int(t['legs over 2.00 m, perception'])}
T3=pd.DataFrame(rows)
# contact-plane legs in corridor: from SAFETY_BY_STAGE (whole-leg) and the board win > 3.42
cp = bd[(bd.sig==0.25)&bd.wind_rung&(bd.win>3.42)].groupby('arm').size().to_dict()
if any(int(cp.get(a,0)) for a in ORDER): raise SystemExit('REFUSE: a corridor peak above the 3.42 m contact plane exists; the clearance columns assume none')   # was a table column; now a check
emit('T3','H1 statistic by arm: corridor p95 of the peak upward deviation above the boom-tip, truth channel, σ = 0.25, nine wind stages pooled, with the clearance that remained to the 3.42 m contact plane',T3,
     f"{SRC['pv']} (value, interval, p); {SRC['abl']} (entered, legs over the bar); {SRC['board']} (per-leg corridor peaks: the worst leg and the two clearance columns)",
     'H1 pool = legs that entered the terminal approach corridor (within 10 m of the boom-tip); the legs-over-2.00 m share is of that pool. Bar: upper bound of the interval below 2.00 m (the table “Hypothesis Tests”). The contact plane is where the QuadPlane\'s top would touch the DHC-2\'s belly, 3.42 m above the boom-tip; clearance below it = 3.42 m minus the p95, and 3.42 m minus the highest corridor peak among the arm\'s legs; both include the 1.42 m the guard keeps in reserve. Interval: two-sided 95% seed-cluster bootstrap (30 seeds resampled whole, 10,000 draws, fixed seed). The one-sided bootstrap p is the share of the 10,000 whole-seed resamples whose p95 fell at or above the bar.')
# T20 perception beside truth (the table “Truth vs. Perception: H1 Statistic by Arm”): the H1 statistic and the legs over the bar on both channels, σ 0.25,
# per-leg peaks from PERCEPTION_PEAKS_PER_LEG.csv (truth == the board's win), seed-cluster bootstrap on both p95s; refuses unless every
# cell reproduces HYPOTHESIS_VERDICTS_H1_TRUTH_AND_PERCEPTION.csv and figures/H1_CORRIDOR_P95_MARGIN.csv.
SEED_BLOCK = list(range(10775, 10805))   # the seed block of record (also defined at the ablation block, same value)
ppl = pd.read_csv(src('ppl', O/'PERCEPTION_PEAKS_PER_LEG.csv'))
h1m = pd.read_csv(src('h1m', O/'figures/H1_CORRIDOR_P95_MARGIN.csv'))
def p95_ci(df, col):
    """p95 of a per-leg column and its two-sided 95% seed-cluster bootstrap interval, whole seeds, PB's draws (the H1 method of record)."""
    g = df[['seed', col]].rename(columns={col:'win'}); labels = np.sort(g.seed.unique())
    if list(labels)!=SEED_BLOCK: raise SystemExit('REFUSE: perception cell is not on the seed block')
    grp = PB._groups(g, 'win', labels); picks = PB._picks(len(labels))
    vals = np.array([np.percentile(np.concatenate([grp[i] for i in row]), 95) for row in picks])
    return float(np.percentile(g.win.dropna().to_numpy(float), 95)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
def _trip(s): return tuple(float(v) for v in re.findall(r'-?\d+\.\d+', str(s)))
rows=[]; n20=0
for arm in ORDER:
    x = bd[(bd.sig==0.25)&bd.wind_rung&(bd.arm==arm)][['corpus','leg','seed','win']].merge(ppl[['corpus','leg','truth_corridor','perception_corridor']], on=['corpus','leg'], how='inner')
    if len(x)!=540 or not ((x.win.isna()==x.truth_corridor.isna()).all() and float((x.win-x.truth_corridor).abs().max())<=1e-4): raise SystemExit(f'REFUSE: the per-leg perception file does not join {arm} one to one, or its truth peak differs from the board')
    if (x.perception_corridor.isna()!=x.win.isna()).any(): raise SystemExit(f'REFUSE: {arm} perception peaks are missing on corridor legs')
    t = p95_ci(x, 'win'); p = p95_ci(x, 'perception_corridor')
    k_t, k_p = int((x.win>H1_BAR_M).sum()), int((x.perception_corridor>H1_BAR_M).sum())
    h = h1tp[h1tp.arm==arm].iloc[0]; f = h1m[h1m.arm==arm].iloc[0]
    r3 = lambda v: tuple(round(q,3) for q in v)
    ok = r3(t)==_trip(h['corridor p95, truth (m)']) and r3(p)==_trip(h['corridor p95, perception (m) - not graded']) and k_t==int(h['legs over 2.00 m, truth']) and k_p==int(h['legs over 2.00 m, perception'])
    ok = ok and round(p[0],3)==float(f['corridor p95, perception (m), not graded']) and r3(p)[1:]==_trip(f['two-sided 95% seed-cluster bootstrap interval, perception (m)']) and round(t[0],3)==float(f['corridor p95, truth (m)'])
    if not ok: raise SystemExit(f'REFUSE: perception-beside-truth cells for {arm} do not reproduce HYPOTHESIS_VERDICTS_H1_TRUTH_AND_PERCEPTION.csv / H1_CORRIDOR_P95_MARGIN.csv')
    n20+=1
    rows.append({'arm':arm,'legs in the corridor (n)':int(x.win.notna().sum()),'truth: 95th-percentile peak height (m) [95% interval]':f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]",
                 'perception: 95th-percentile peak height (m) [95% interval]':f"{p[0]:.3f} [{p[1]:.3f}, {p[2]:.3f}]",'perception minus truth at the p95 (m)':f"{p[0]-t[0]:+.3f}",
                 'legs over 2.00 m, truth':k_t,'legs over 2.00 m, perception':k_p,'verdict (truth)':J[f'H1_{arm}']['verdict']})
    J[f'PERC_{arm}']={'p95':p[0],'lo':p[1],'hi':p[2],'over':k_p}
T20=pd.DataFrame(rows)
emit('T4','Perception beside truth: the H1 statistic and the legs over the 2.00 m bar on both channels, σ = 0.25, nine wind stages pooled (perception not graded)',T20,
     f"{SRC['ppl']} (per-leg peaks on both channels) joined to {SRC['board']}; every cell reproduces {SRC['h1tp']} and {SRC['h1m']}",
     'Truth = the simulation state; perception = what the controller and the altitude guard read (the GPS-direct height). H1 is tested on truth. Both p95s carry the two-sided 95% seed-cluster bootstrap (30 seeds resampled whole, 10,000 draws, fixed seed). The verdict column is the truth verdict of the H1 table.')
print(f'perception check: {n20} arms reproduce HYPOTHESIS_VERDICTS_H1_TRUTH_AND_PERCEPTION.csv / H1_CORRIDOR_P95_MARGIN.csv')
# T4 H2
rows=[]
for arm in [a for a in ORDER if a!='PID']:
    r = pv[(pv.hypothesis=='H2')&(pv.arm==arm)].iloc[0]; n9 = h2a[(h2a.sigma==0.25)&(h2a.pool=='nine wind stages')&(h2a.arm==arm)].iloc[0]
    assert abs(n9.diff_pp - r.value) < 0.01
    rows.append({'arm':arm,'captures of 540, arm':f"{int(n9.arm_captures)} ({n9.arm_rate:.1f}%)",'captures of 540, PID-baseline':f"{int(n9.pid_captures)} ({n9.pid_rate:.1f}%)",
                 'difference on the same flights (pp)':f"{r.value:+.2f}",'95% interval (pp)':f"[{r.ci95_lo:+.2f}, {r.ci95_hi:+.2f}]",'p':f"{r.boot_p_one_sided:.4f}",'verdict':verdict('H2',r) if bool(r.judged) else 'comparator (not graded)'})
    J[f'H2_{arm}']={'diff':float(r.value),'lo':float(r.ci95_lo),'hi':float(r.ci95_hi),'p':float(r.boot_p_one_sided),'k':int(n9.arm_captures),'rate':float(n9.arm_rate),'verdict':rows[-1]['verdict']}
J['PID_k9']=int(n9.pid_captures); J['PID_rate9']=float(n9.pid_rate)
T4=pd.DataFrame(rows)
emit('T5','H2 statistic by arm: same-seed difference in docking-envelope captures against the PID-baseline, σ = 0.25',T4,f"{SRC['pv']}; {SRC['h2a']}",
     'A pair is one (stage, seed, trajectory) flown by both controllers. Bar: lower bound of the nine-stage interval above 0 pp (the table “Hypothesis Tests”). PIDO is a comparator: reported with the same statistic, not graded. A pair is the same stage, seed and trajectory flown by both controllers. Interval: two-sided 95% seed-cluster bootstrap (30 seeds resampled whole, 10,000 draws, fixed seed); p: one-sided, the share of the 10,000 resamples whose difference fell at or below 0 pp.')
# Calm control captures (S0) from the board; intervals = two-sided 95% seed-cluster bootstrap
def _seed_sums(g):
    ok = g.passed.astype(str).str.lower().eq('true').to_numpy(); seed = g.seed.to_numpy(); labels = np.sort(np.unique(seed))
    if len(labels) != 30: raise SystemExit('REFUSE: Calm Control cell does not hold 30 seeds')
    return labels, np.array([ok[seed==x].sum() for x in labels], float), np.array([(seed==x).sum() for x in labels], float), ok
def boot_rate(g):
    _, k, n, ok = _seed_sums(g); picks = PB._picks(len(k)); v = 100.0 * k[picks].sum(axis=1) / n[picks].sum(axis=1)
    return int(ok.sum()), len(ok), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
def boot_change(g1, g2):
    """second minus first, pp, on the same (seed, trajectory) legs; whole seeds resampled with PB's draws."""
    j = g1.set_index(['seed','trajectory']).passed.astype(str).str.lower().eq('true').rename('f1').to_frame().join(
        g2.set_index(['seed','trajectory']).passed.astype(str).str.lower().eq('true').rename('f2'), how='inner')
    if len(j) != 60 or len(g1) != 60 or len(g2) != 60: raise SystemExit('REFUSE: Calm Control flights do not pair 60 legs')
    seed = j.index.get_level_values('seed').to_numpy(); labels = np.sort(np.unique(seed))
    d = np.array([(j.f2[seed==x].sum() - j.f1[seed==x].sum()) for x in labels], float); n = np.array([(seed==x).sum() for x in labels], float)
    picks = PB._picks(len(labels)); v = 100.0 * d[picks].sum(axis=1) / n[picks].sum(axis=1)
    return dict(change=100.0*(j.f2.mean()-j.f1.mean()), lo=float(np.percentile(v,2.5)), hi=float(np.percentile(v,97.5)),
                lost=int((j.f1&~j.f2).sum()), gained=int((~j.f1&j.f2).sum()), same=int((j.f1==j.f2).sum()))
rows=[]; rows11=[]
calm_src = pd.read_csv(src('calm', O/'CALM_STAGE_REPEATABILITY.csv')); calm_src['arm']=calm_src.arm.replace({'PID-baseline (comparator)':'PID'}).str.replace('FlightDock-','')
for arm in ORDER:
    g = bd[(bd.sig==0.25)&(bd.stage=='S0')&(bd.arm==arm)]; g2 = bd[(bd.sig==0.35)&(bd.stage=='S0')&(bd.arm==arm)]
    k,n,lo,hi = boot_rate(g); k2,n2,lo2,hi2 = boot_rate(g2); c = boot_change(g, g2)
    r = calm_src[calm_src.arm==arm].iloc[0]
    if (int(r['first flight (in the sigma 0.25 corpus)'].split(' ')[0]), int(r['second flight (in the sigma 0.35 corpus)'].split(' ')[0]), int(r['legs that flipped']), int(r['success lost']), int(r['success gained']), int(r['legs identical in outcome'])) != (k, k2, c['lost']+c['gained'], c['lost'], c['gained'], c['same']):
        raise SystemExit(f'REFUSE: Calm Control counts for {arm} do not reproduce CALM_STAGE_REPEATABILITY.csv')
    rows.append({'arm':arm,'Calm Control captures, first flight (σ 0.25 ladder)':f"{k} of {n} ({100*k/n:.1f}% [{lo:.1f}, {hi:.1f}])",'Calm Control captures, second flight (σ 0.35 ladder)':f"{k2} of {n2} ({100*k2/n2:.1f}% [{lo2:.1f}, {hi2:.1f}])"})
    rows11.append({'arm':arm,'paired legs':60,'first flight (σ 0.25 ladder)':f"{k} of 60, {100*k/n:.1f}% [{lo:.1f}, {hi:.1f}]",'second flight (σ 0.35 ladder)':f"{k2} of 60, {100*k2/n2:.1f}% [{lo2:.1f}, {hi2:.1f}]",
                   'change, second minus first (pp)':round(c['change'],1),'change: two-sided 95% seed-cluster bootstrap (pp)':f"[{c['lo']:+.1f}, {c['hi']:+.1f}]",
                   'legs that flipped':c['lost']+c['gained'],'success lost':c['lost'],'success gained':c['gained'],'legs identical in outcome':c['same']})
    J[f'S0_{arm}']={'k1':k,'k2':k2,'n':n}
T4b=pd.DataFrame(rows)
# T4b is no longer emitted (package 10): its two columns are columns 3 and 4 of T11; the counts stay in J['S0_*'].
# T4c reference floor: pooled nine-stage capture rate by arm and sigma against the 35% minimum capture rate of reference
FLOOR_PCT = 35.0   # crewed aerial-refueling docking success rate reported by Ren and Quan (2024); a reference floor, reported, not a hypothesis
rows=[]; J['FLOOR']={}
for sig in (0.25,0.35):
    for arm in ORDER:
        g = bd[(bd.sig==sig)&bd.wind_rung&(bd.arm==arm)]
        k,n,lo,hi = boot_rate(g)
        _, ks_, ns_, _ = _seed_sums(g); _v = 100.0*ks_[PB._picks(len(ks_))].sum(axis=1)/ns_[PB._picks(len(ks_))].sum(axis=1); p_floor = float((_v <= FLOOR_PCT).mean())   # one-sided bootstrap p against the floor, the H1-H3 method
        n9r = h3a[(h3a.pool=='nine wind stages')&(h3a.arm==arm)].iloc[0]; ref = int(n9r.captures_s025) if sig==0.25 else int(n9r.captures_s035)
        if n!=540 or k!=ref: raise SystemExit(f'REFUSE: pooled captures for {arm} at sigma {sig} ({k} of {n}) do not reproduce H3_ALTERNATIVES ({ref} of 540)')
        rate=100*k/n
        rows.append({'σ':sig,'arm':arm,'captures of 540':k,'capture rate [95% interval] (%)':f"{rate:.1f} [{lo:.1f}, {hi:.1f}]",'p (rate at or below 35%)':f"{p_floor:.4f}",'against the 35% floor':'above' if lo>FLOOR_PCT else ('below' if hi<FLOOR_PCT else 'straddles')})
        J['FLOOR'][f'{arm}_{sig}']={'k':k,'n':n,'rate':rate,'lo':lo,'hi':hi,'margin':rate-FLOOR_PCT,'above':bool(lo>FLOOR_PCT),'below':bool(hi<FLOOR_PCT),'p':p_floor}
T4c=pd.DataFrame(rows)
emit('T6','Pooled docking-envelope capture rate by arm and turbulence level against the 35% minimum capture rate of reference (reported; decides no hypothesis)',T4c,SRC['board'],
     'Nine wind stages pooled, 540 legs per arm and σ, seed block 10775-10804. The reference floor is the crewed aerial-refueling docking success rate of about 35% reported by Ren and Quan (2024) (the document). Intervals: two-sided 95% seed-cluster bootstrap (30 seeds resampled whole, 10,000 draws). "Against the 35% floor" reads the whole interval: above = the whole interval above 35%, below = the whole interval below, straddles = the interval contains 35%; the floor is a reference, not a hypothesis bar. The p is the one-sided bootstrap p of the H1 to H3 tables applied to the floor: the share of the 10,000 whole-seed resamples whose pooled rate fell at or below 35%. It says how firmly an arm cleared the reference, nothing more; the floor decides no hypothesis, so no α applies and no verdict follows from it.')
J['SIGMA_RISE_PCT']=float(100.0*(0.35/0.25-1.0))   # numpy-free arithmetic in python: the rise in commanded turbulence intensity from sigma 0.25 to 0.35
J['QUADPLANE_MASS_KG']=4.5   # kg, the QuadPlane airframe of record
# T5 H3
rows=[]
for arm in ORDER:
    r = pv[(pv.hypothesis=='H3')&(pv.arm==arm)].iloc[0]; n9 = h3a[(h3a.pool=='nine wind stages')&(h3a.arm==arm)].iloc[0]
    assert abs(n9.decrease_pp - r.value) < 0.01
    rows.append({'arm':arm,'captures of 540, σ 0.25':f"{int(n9.captures_s025)} ({n9.rate_s025:.1f}%)",'captures of 540, σ 0.35':f"{int(n9.captures_s035)} ({n9.rate_s035:.1f}%)",
                 'decrease on the same flights (pp)':f"{r.value:.2f}",'95% interval (pp)':f"[{r.ci95_lo:.2f}, {r.ci95_hi:.2f}]",'p':f"{r.boot_p_one_sided:.4f}",'verdict':verdict('H3',r) if bool(r.judged) else 'comparator (not graded)',
                 'σ 0.35 rate against the 35% floor':('above' if J['FLOOR'][f'{arm}_0.35']['above'] else ('below' if J['FLOOR'][f'{arm}_0.35']['below'] else 'straddles'))})
    J[f'H3_{arm}']={'dec':float(r.value),'lo':float(r.ci95_lo),'hi':float(r.ci95_hi),'p':float(r.boot_p_one_sided),'k25':int(n9.captures_s025),'k35':int(n9.captures_s035),'r25':float(n9.rate_s025),'r35':float(n9.rate_s035),'verdict':rows[-1]['verdict']}
T5=pd.DataFrame(rows)
emit('T7','H3 statistic by arm: same-seed decrease in docking-envelope captures from σ = 0.25 to σ = 0.35, same controller',T5,f"{SRC['pv']}; {SRC['h3a']}",
     'Bar: upper bound of the nine-stage interval below 35 pp (the table “Hypothesis Tests”). The PID-baseline and PIDO are comparators: reported with the same statistic, not graded. Last column: the pooled σ 0.35 capture rate of the reference-floor table read against the 35% minimum capture rate of reference (Ren and Quan, 2024): above = the whole interval above 35%, below = the whole interval below, straddles = the interval contains 35%; a reference, not a verdict. Interval: two-sided 95% seed-cluster bootstrap (30 seeds resampled whole, 10,000 draws, fixed seed); p: one-sided, the share of the 10,000 resamples whose decrease fell at or above 35 pp.')
# T6 clearance at sigma 0.35 (reported)
rows=[]
for arm in ORDER:
    a = abl[(abl.arm==arm)&(abl.sigma==0.35)].iloc[0]
    rows.append({'arm':arm,'legs that entered the corridor':int(a.entered),'95th-percentile peak height above the boom-tip (m)':f"{a.p95_corridor_m:.3f}",'95% interval (m)':a.p95_bootstrap_ci95,'legs over 2.00 m':int(a.legs_over_bar),'wind stages with p95 over 2.00 m (of 9)':int(a.stages_over),
                 'legs over the 3.42 m plane in the corridor':int(bd[(bd.sig==0.35)&bd.wind_rung&(bd.win>3.42)&(bd.arm==arm)].shape[0])})
    J[f'P95_035_{arm}']={'p95':float(a.p95_corridor_m),'over':int(a.legs_over_bar),'stages_over':int(a.stages_over)}
T6=pd.DataFrame(rows)
emit('T8','Vertical clearance at σ = 0.35, nine wind stages (not graded for H1)',T6,f"{SRC['abl']}; {SRC['board']}",'H1 is graded at σ = 0.25 only. These rows describe the σ = 0.35 legs and decide nothing. Interval: two-sided 95% seed-cluster bootstrap (30 seeds resampled whole, 10,000 draws, fixed seed).')
# T7 PTA ablation, factor 1 (the offset), recomputed from the board with the seed-cluster bootstrap on BOTH outcomes (operator 10-01:
# every table carries it; the 09-20 source file's capture intervals were Newcombe). Point estimates, pair counts and the p95 intervals
# must reproduce OFFSET_EFFECT_BY_OUTCOME.csv or the script refuses. The Holm sentence is withdrawn.
SEED_BLOCK = list(range(10775, 10805))
off = pd.read_csv(src('off', P/'offset_effect_20260920/OFFSET_EFFECT_BY_OUTCOME.csv')); off = off[off.pool=='nine wind stages']
def legs(arm, sig, traj=None):
    g = bd[(bd.sig==sig)&bd.wind_rung&(bd.arm==arm)]
    return g if traj is None else g[g.trajectory==traj]
def pair_boot(ga, gb):
    """ga minus gb in pp on the (stage, seed, trajectory) legs both flew; whole seeds resampled with PB's draws (the H2 method of record)."""
    j = ga.set_index('key')[['passed','seed']].join(gb.set_index('key').passed.rename('pb'), how='inner')
    if len(j)!=len(ga) or len(j)!=len(gb): raise SystemExit('REFUSE: the paired join does not cover both pools')
    seed = j.seed.to_numpy(); labels = np.sort(np.unique(seed))
    if list(labels)!=SEED_BLOCK: raise SystemExit('REFUSE: paired keys do not cover the seed block 10775-10804')
    pa, pb_ = j.passed.to_numpy(bool), j.pb.to_numpy(bool)
    d = np.array([(pa[seed==s].sum()-pb_[seed==s].sum()) for s in labels], float); n = np.array([(seed==s).sum() for s in labels], float)
    picks = PB._picks(len(labels)); v = 100.0*d[picks].sum(axis=1)/n[picks].sum(axis=1)
    return dict(n=len(j), ka=int(pa.sum()), kb=int(pb_.sum()), diff=100.0*(pa.sum()-pb_.sum())/len(j), lo=float(np.percentile(v,2.5)), hi=float(np.percentile(v,97.5)), p=p_two(v))
def p_two(v):
    """Two-sided bootstrap p: twice the smaller share of the resamples at or below / at or above zero, capped at 1."""
    v = np.asarray(v, float); return float(min(1.0, 2.0*min(float((v<=0).mean()), float((v>=0).mean()))))
def p95_diff(ga, gb):
    """PB.diff_ci's p95 difference redrawn with PB's labels and draws so the resample vector is available for the p; the redrawn interval must equal PB.diff_ci's (3 dp)."""
    ref = PB.diff_ci(ga, gb, 'win')
    labels = np.sort(np.union1d(ga.seed.unique(), gb.seed.unique())); xa, xb = PB._groups(ga, 'win', labels), PB._groups(gb, 'win', labels)
    vals = np.array([PB.p95(np.concatenate([xa[i] for i in row])) - PB.p95(np.concatenate([xb[i] for i in row])) for row in PB._picks(len(labels))])
    lo, hi = float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))
    if abs(round(lo,3)-round(ref[3],3))>1e-9 or abs(round(hi,3)-round(ref[4],3))>1e-9: raise SystemExit('REFUSE: redrawn p95 difference interval differs from PB.diff_ci')
    return ref + (p_two(vals),)
def klass(lo, hi, up_is_good):
    if lo>0: return 'helped' if up_is_good else 'hurt'
    if hi<0: return 'hurt' if up_is_good else 'helped'
    return 'no clear change'
IVL = 'two-sided 95% seed-cluster bootstrap interval on the change'
FLOORCOL = 'against the 35% floor, without → with'   # the reference floor as a diagnostic beside the paired change
FSENT = (' "Against the 35% floor": each arm\'s pooled capture rate read against the 35% crewed aerial-refueling docking success rate (Ren and Quan, 2024) from the reference-floor table '
         '(above = whole interval above 35%, below = whole interval below, straddles = the interval contains 35%), without the factor → with it; the floor is read per arm and is unpaired, so a change of category is read with the paired change beside it.')
def fcat(arm, sig):
    f = J['FLOOR'][f'{arm}_{sig}']; return 'above' if f['above'] else ('below' if f['below'] else 'straddles')
PSENT = ' p: two-sided bootstrap p, twice the smaller share of the 10,000 whole-seed resamples at or below and at or above zero; the ablation decides no hypothesis, so no α applies.'
def two_rows(ga, gb, lab, wcol, ocol, arms=None):
    """One captures row and one corridor-p95 row for ga minus gb; wcol/ocol name the 'with'/'without' columns."""
    c = pair_boot(ga, gb); p = p95_diff(ga, gb)   # p = (p95 a, p95 b, a-b, lo, hi, two-sided p)
    cap = dict(lab, **{'outcome':'captures (pp)','legs or pairs':c['n'], ocol:f"{100.0*c['kb']/c['n']:.1f}", wcol:f"{100.0*c['ka']/c['n']:.1f}",
                       'change, with minus without':f"{c['diff']:+.2f}", IVL:f"[{c['lo']:+.2f}, {c['hi']:+.2f}]", 'p':f"{c['p']:.4f}", 'reading':klass(c['lo'],c['hi'],True)})
    p95 = dict(lab, **{'outcome':'corridor p95 (m)','legs or pairs':int(ga.win.notna().sum()), ocol:f"{p[1]:.3f}", wcol:f"{p[0]:.3f}",
                       'change, with minus without':f"{p[2]:+.3f}", IVL:f"[{p[3]:+.3f}, {p[4]:+.3f}]", 'p':f"{p[5]:.4f}", 'reading':klass(p[3],p[4],False)})
    cap[FLOORCOL] = (f"{fcat(arms[1], lab['σ'])} → {fcat(arms[0], lab['σ'])}" if arms else ''); p95[FLOORCOL] = ''   # arms = (with, without); pooled rows only
    return cap, p95, c, p
def chk_off(sig, ctl, tname, n95, c, p):
    q = off[(off.sigma==sig)&(off.trajectory==tname)&(off.controller.str.startswith(ctl))]
    qc, qp = q[q.outcome=='captures (pp)'], q[q.outcome=='corridor p95 (m)']
    ok = len(qc)==1 and len(qp)==1
    if ok:
        qc, qp = qc.iloc[0], qp.iloc[0]
        ok = (int(qc.legs_or_pairs)==c['n'] and abs(round(100.0*c['kb']/c['n'],1)-qc.offset_off)<1e-9 and abs(round(100.0*c['ka']/c['n'],1)-qc.offset_on)<1e-9
              and abs(round(c['diff'],2)-qc.change)<1e-9 and int(qp.legs_or_pairs)==n95 and abs(round(p[1],3)-qp.offset_off)<1e-9 and abs(round(p[0],3)-qp.offset_on)<1e-9
              and abs(round(p[2],3)-qp.change)<1e-9 and abs(round(p[3],3)-qp.ci_lo)<1e-9 and abs(round(p[4],3)-qp.ci_hi)<1e-9)
    if not ok: raise SystemExit(f'REFUSE: offset effect {ctl} σ {sig} {tname} does not reproduce OFFSET_EFFECT_BY_OUTCOME.csv')
FAM = [('S1O','S1','FlightDock-S1'),('A2O','A2','FlightDock-A2'),('PIDO','PID','PID-baseline')]
WO, WI = 'without the offset', 'with the offset'
rows=[]; rowsf=[]; n_chk=0
for sig in (0.25, 0.35):
    for on, offa, ctl in FAM:
        for traj in (None, 'straight', 'racetrack'):
            tname = 'all' if traj is None else traj
            ga, gb = legs(on, sig, traj), legs(offa, sig, traj)
            cap, p95, c, p = two_rows(ga, gb, {'σ':sig,'trajectory':tname,'controller':ctl}, WI, WO, arms=((on, offa) if traj is None else None))
            chk_off(sig, ctl, tname, p95['legs or pairs'], c, p); n_chk += 1
            rowsf += [cap, p95]
            if traj is None:
                rows += [{k:v for k,v in cap.items() if k!='trajectory'}, {k:v for k,v in p95.items() if k!='trajectory'}]
def ablation_view(df, drop=()):
    """Display copy (package 10): the change and its interval in one cell, plain labels; the J rows keep the separate keys."""
    v = df.copy()
    v['change, with minus without [95% interval]'] = v['change, with minus without'].astype(str) + ' ' + v[IVL].astype(str)
    v = v.drop(columns=['change, with minus without', IVL] + list(drop)).rename(columns={'legs or pairs':'flights'})
    v['outcome'] = v.outcome.replace({'corridor p95 (m)':'peak height, p95 (m)'})
    tail = ['change, with minus without [95% interval]', 'p', FLOORCOL, 'reading']
    return v[[c for c in v.columns if c not in tail] + tail]
COLS7 = ['outcome','σ','controller','legs or pairs',WO,WI,'change, with minus without',IVL,'p',FLOORCOL,'reading']
T7 = pd.DataFrame(rows)[COLS7]
NOTE7 = ('Change = with the offset minus without, on the same (stage, seed, trajectory) legs; the interval is the two-sided 95% seed-cluster bootstrap '
         '(30 seeds resampled whole, 10,000 draws, fixed seed) on both outcomes. "Reading" is the analyst\'s class rule (helped, hurt, no clear change by '
         'whether the interval excludes zero), not a verdict.' + PSENT + FSENT + ' Amplifying; decides no hypothesis.')
emit('T9','PTA ablation, factor 1: what the 0.25 m vertical setpoint offset changed, controller by controller, nine wind stages (amplifying; decides no hypothesis)',ablation_view(T7),SRC['board'],NOTE7)
J['offset_rows']=rows
# T7b PTA ablation, factor 2 (the learned corrections), the offset held fixed.
# Without the offset: FlightDock-S1 or A2 minus the PID-baseline (at σ 0.25 these ARE the H2 statistics); with the offset: S1O or A2O minus PIDO.
# Point estimates and counts reproduce comparative/ARM_PAIRS.csv; the σ 0.25 without-offset capture intervals reproduce the H2 intervals of record.
ap = pd.read_csv(src('ap', B18/'comparative/ARM_PAIRS.csv'))
def _ivl(s): lo_, hi_ = re.findall(r'-?\d+\.\d+', str(s)); return float(lo_), float(hi_)
def chk_pairs(sig, a, b, c, p):
    na, nb = V.NAME[a], V.NAME[b]
    for pool in ('nine wind stages', 'nine wind stages, terminal-approach-corridor p95 (m)'):
        q = ap[(ap.sigma==sig)&(ap.pool==pool)&(((ap.arm_a==na)&(ap.arm_b==nb))|((ap.arm_a==nb)&(ap.arm_b==na)))]
        if len(q)!=1: raise SystemExit(f'REFUSE: ARM_PAIRS.csv lacks the pair {a}/{b} at σ {sig} ({pool})')
        q = q.iloc[0]; flip = (q.arm_a==nb); dd = -q.diff_pp if flip else q.diff_pp
        if 'p95' in pool:
            lo_, hi_ = _ivl(q.ci95); lo_, hi_ = ((-hi_, -lo_) if flip else (lo_, hi_))
            ok = abs(round(p[2],3)-round(dd,3))<1e-9 and abs(round(p[3],3)-lo_)<1e-9 and abs(round(p[4],3)-hi_)<1e-9
        else:
            ka, kb = (q.k_b, q.k_a) if flip else (q.k_a, q.k_b)
            ok = int(q.n_pairs)==c['n'] and int(ka)==c['ka'] and int(kb)==c['kb'] and abs(round(c['diff'],2)-round(dd,2))<1e-9
        if not ok: raise SystemExit(f'REFUSE: learned-correction effect {a} minus {b} at σ {sig} ({pool}) does not reproduce ARM_PAIRS.csv')
    if sig==0.25 and b=='PID':
        r = pv[(pv.hypothesis=='H2')&(pv.arm==a)].iloc[0]
        if abs(round(c['diff'],2)-r.value)>1e-9 or abs(round(c['lo'],2)-r.ci95_lo)>1e-9 or abs(round(c['hi'],2)-r.ci95_hi)>1e-9:
            raise SystemExit(f'REFUSE: {a} minus PID at σ 0.25 does not reproduce the H2 interval of record')
LEARN = [('without the offset','S1','PID'),('without the offset','A2','PID'),('with the offset','S1O','PIDO'),('with the offset','A2O','PIDO')]
WLO, WLI = 'without the learned corrections', 'with the learned corrections'
rows=[]; J['LEARN']={}
for sig in (0.25, 0.35):
    for state, a, b in LEARN:
        fam = {'S1':'S1','S1O':'S1','A2':'A2','A2O':'A2'}[a]
        cap, p95, c, p = two_rows(legs(a, sig), legs(b, sig), {'σ':sig,'offset state':state,'learned corrections':f'FlightDock-{fam}','arms (with minus without)':f'{a} minus {b}'}, WLI, WLO, arms=(a, b))
        chk_pairs(sig, a, b, c, p); n_chk += 1
        rows += [cap, p95]
        J['LEARN'][f'captures|{sig}|{state.split(" ")[0]}|{fam}'] = dict(change=c['diff'], lo=c['lo'], hi=c['hi'], p=c['p'], reading=cap['reading'], without=100.0*c['kb']/c['n'], with_=100.0*c['ka']/c['n'], n=c['n'], arms=f'{a} minus {b}')
        J['LEARN'][f'p95|{sig}|{state.split(" ")[0]}|{fam}'] = dict(change=p[2], lo=p[3], hi=p[4], p=p[5], reading=p95['reading'], without=p[1], with_=p[0], n=p95['legs or pairs'], arms=f'{a} minus {b}')
T7b = pd.DataFrame(rows)[['outcome','σ','offset state','learned corrections','arms (with minus without)','legs or pairs',WLO,WLI,'change, with minus without',IVL,'p',FLOORCOL,'reading']]
emit('T10','PTA ablation, factor 2: what the learned corrections changed with the offset held fixed, nine wind stages (amplifying; decides no hypothesis)',ablation_view(T7b, drop=('offset state','learned corrections')),f"{SRC['board']}; checked against {SRC['ap']} and {SRC['pv']}",
     'Change = the FlightDock arm minus the PID-baseline in the same offset state (without the offset: S1 or A2 minus the PID-baseline; with the offset: S1O or A2O minus PIDO), '
     'on the same (stage, seed, trajectory) legs; two-sided 95% seed-cluster bootstrap interval on both outcomes (30 seeds resampled whole, 10,000 draws, fixed seed). '
     'The without-offset capture rows at σ 0.25 are the H2 statistics of the H2 table. "Reading" is the analyst\'s class rule (interval excludes zero), not a verdict.' + PSENT + FSENT + ' Amplifying; decides no hypothesis.')
J['learn_rows']=rows
print(f'ablation check: {n_chk} cells reproduce OFFSET_EFFECT_BY_OUTCOME.csv / ARM_PAIRS.csv / the H2 intervals')
# T8 terminal types of unsuccessful legs (classification of the board)
def classify(r):
    reasons = ast.literal_eval(r.fail_reasons); term = ast.literal_eval(r.terminals)
    if np.isnan(r.win): return 'never entered the corridor (10 m)'
    if r.real_overtake: return 'ahead of the DHC-2 (overtake) in close formation'
    if any(x.startswith('truth_ceiling_bust') for x in reasons): return 'entered the corridor, exclusion-zone violation (over 2.00 m)'
    if any(x.startswith('intersection_dwell_fail:best=0.00s') for x in reasons): return 'entered the corridor, never inside the M3b envelope on both channels'
    if any(x.startswith('intersection_dwell_fail') for x in reasons): return 'inside the envelope, dwell under 3.0 s'
    return 'other: ' + ';'.join(reasons)[:40]
fails = bd[(~bd.passed)&bd.wind_rung].copy(); fails['class']=fails.apply(classify,axis=1)
T8 = fails.groupby(['sig','arm','class']).size().unstack('class').fillna(0).astype(int)
T8['unsuccessful legs (of 540)']=T8.sum(axis=1); T8=T8.reset_index(); T8['o']=T8.arm.map(ORDER.index); T8=T8.sort_values(['sig','o']).drop(columns='o').rename(columns={'sig':'σ'})
T8MAP={'never entered the corridor (10 m)':'never came within 10 m','ahead of the DHC-2 (overtake) in close formation':'overtook the DHC-2 in close formation','entered the corridor, exclusion-zone violation (over 2.00 m)':'climbed above the 2.00 m bar (exclusion-zone violation)','entered the corridor, never inside the M3b envelope on both channels':'came within 10 m, never inside the envelope on both channels','inside the envelope, dwell under 3.0 s':'reached the envelope, held it under 3.0 s'}   # package 10: plain headers
if set(T8.columns)-{'σ','arm','unsuccessful legs (of 540)'}!=set(T8MAP): raise SystemExit('REFUSE: an unexpected failure class appeared: %s' % sorted(set(T8.columns)-{'σ','arm','unsuccessful legs (of 540)'}))
T8=T8.rename(columns=T8MAP)[['σ','arm','unsuccessful legs (of 540)']+list(T8MAP.values())]
emit('T11','Why legs were unsuccessful: one class per leg, nine wind stages, six arms',T8,SRC['board'],
     'Classes are assigned in the order listed, so each leg appears once: never within 10 m of the boom-tip; ahead of the DHC-2 in close formation (overtake); exclusion-zone violation above 2.00 m inside the corridor; never inside the M3b envelope on both channels; inside the envelope but the continuous dwell fell short of 3.0 s. The classification is the analyst\'s reading of the per-leg fail reasons on the board, not a scorer output.')
J['T11']=T8.to_dict('records')
# T9 guard record + breakaway cost (package 10, M1): the breakaway cost is recomputed from the board with the seed-cluster bootstrap on the
# difference; the CMH test of BREAKAWAY_COST.csv is superseded (one method everywhere). Counts and rates must reproduce that file or the script refuses.
sbs = pd.read_csv(src('sbs', B20/'comparative/SAFETY_BY_STAGE.csv')); sbs['armS']=sbs.arm.map(short)
brk = pd.read_csv(src('brk', O/'BREAKAWAY_COST.csv')); brk['armS']=brk.arm.replace({'PID-baseline (comparator)':'PID'}).str.replace('FlightDock-','')
bd['breakaway'] = bd.guards.map(lambda s: int(ast.literal_eval(s)['emergency_breakaways'])>0)
def boot_brk(g):
    """capture rate of the legs with a breakaway minus the legs without, same arm and σ; whole seeds resampled with PB's draws."""
    seed=g.seed.to_numpy(); labels=np.sort(np.unique(seed))
    if list(labels)!=SEED_BLOCK or len(g)!=540: raise SystemExit('REFUSE: breakaway cell is not 540 legs on the seed block')
    ok=g.passed.to_numpy(bool); b=g.breakaway.to_numpy(bool)
    kb=np.array([(ok&b)[seed==s].sum() for s in labels],float); nb=np.array([b[seed==s].sum() for s in labels],float)
    kn=np.array([(ok&~b)[seed==s].sum() for s in labels],float); nn=np.array([(~b)[seed==s].sum() for s in labels],float)
    picks=PB._picks(len(labels))
    with np.errstate(invalid='ignore', divide='ignore'):
        v=100.0*(kb[picks].sum(axis=1)/nb[picks].sum(axis=1) - kn[picks].sum(axis=1)/nn[picks].sum(axis=1))
    return dict(nb=int(b.sum()), rb=100.0*ok[b].mean(), rn=100.0*ok[~b].mean(), diff=100.0*(ok[b].mean()-ok[~b].mean()), lo=float(np.nanpercentile(v,2.5)), hi=float(np.nanpercentile(v,97.5)))
rows=[]; n_brk=0
for sig in (0.25,0.35):
    for arm in ORDER:
        g = sbs[(sbs.armS==arm)&(sbs.sigma==sig)&(sbs.stage!='S0 calm control stage')]; b = brk[(brk.armS==arm)&(brk.sigma==sig)].iloc[0]
        assert len(g)==9 and g.n.sum()==540
        c = boot_brk(bd[(bd.sig==sig)&bd.wind_rung&(bd.arm==arm)])
        if c['nb']!=int(b['legs with a breakaway']) or abs(round(c['rb'],1)-float(b['docked, legs with a breakaway (%)']))>1e-9 or abs(round(c['rn'],1)-float(b['docked, legs without (%)']))>1e-9 or abs(round(c['diff'],1)-float(b['difference (pp)']))>1e-9:
            raise SystemExit(f'REFUSE: breakaway cost for {arm} σ {sig} does not reproduce BREAKAWAY_COST.csv')
        n_brk+=1
        rows.append({'σ':sig,'arm':arm,'legs with an altitude-guard fire':int(g['legs with an altitude fire'].sum()),'legs with an overtake-guard fire':int(g['legs with an overtake fire'].sum()),
                     'legs with a breakaway':c['nb'],'captures, legs with a breakaway (%)':f"{c['rb']:.1f}",'captures, legs without (%)':f"{c['rn']:.1f}",'difference (pp) [95% interval]':f"{c['diff']:+.1f} [{c['lo']:+.1f}, {c['hi']:+.1f}]"})
        J[f'GUARD_{sig}_{arm}']={'alt_fires':int(g['altitude-guard fires'].sum()),'ovt_fires':int(g['overtake-guard fires'].sum()),'alt_legs':int(g['legs with an altitude fire'].sum()),'ovt_legs':int(g['legs with an overtake fire'].sum()),'brk_legs':c['nb'],'cap_brk':c['rb'],'cap_nobrk':c['rn'],'diff':c['diff'],'lo':c['lo'],'hi':c['hi']}
print(f'breakaway check: {n_brk} cells reproduce BREAKAWAY_COST.csv')
# T10 held position (nine wind stages + calm)
held = pd.read_csv(src('held', O/'HELD_POSITION_IN_BOX_BY_STAGE.csv')); held['armS']=held.arm.replace({'PID-baseline (comparator)':'PID','PID-baseline':'PID'}).str.replace('FlightDock-','')
hc = [c for c in held.columns]
rows=[]
for sig in (0.25,0.35):
    for arm in ORDER:
        for st in ('calm control stage','nine wind stages'):
            g = held[(held.armS==arm)&(held.sigma==sig)&(held.stage==st)]
            if len(g)!=1: continue
            g=g.iloc[0]
            rows.append({'σ':sig,'arm':arm,'stage':st,'ticks inside the envelope (truth)':int(g[hc[3]]),'held height above the tip, truth median (m)':g[hc[4]],'perception median (m)':g[hc[5]],'ticks held below the tip (%)':g[hc[7]],'applied altitude correction, median':g[hc[8]],'ticks asking up on altitude (%)':g[hc[9]]})
T10=pd.DataFrame(rows)
T10v=T10[T10.stage=='nine wind stages'].drop(columns=['stage','ticks inside the envelope (truth)'])   # package 10: Calm Control rows and the tick count out of the table; J keeps the full rows
emit('T12','Where each controller holds inside the M3b envelope and what it commands there (per tick, truth channel, nine wind stages)',T10v,SRC['held'],'Per-tick medians over every tick inside the envelope; no interval (ticks are not resampled). Negative height = below the boom-tip. The PID-baseline and PIDO apply no learned correction; PIDO\'s altitude change is the offset.')
J['T12']=rows
# T11 calm repeatability: recomputed from the board with the seed-cluster bootstrap (counts cross-checked above against CALM_STAGE_REPEATABILITY.csv, whose Newcombe column is no longer used)
T11full=pd.DataFrame(rows11); T11=T11full.drop(columns=['legs identical in outcome'])   # package 10: 60 minus the flipped legs
emit('T13','Calm Control repeatability: the same arm on the same 60 legs, flown twice',T11,f"{SRC['board']}; counts cross-checked against {SRC['calm']}",'The stage was flown once in each ladder. The change is the second flight minus the first on the same (seed, trajectory) legs; its interval is the two-sided 95% seed-cluster bootstrap of the paired change (30 seeds resampled whole, 10,000 draws, fixed seed).')
J['T13']=T11full.to_dict('records')
# T12 funnel (package 10, M2): recomputed per leg from EXTRA_TL_PER_LEG.csv (the tick-level script's per-leg steps) with the seed-cluster
# bootstrap on every step; APPROACH_FUNNEL.csv's Wilson intervals are superseded. Every count must reproduce that file or the script refuses.
fun = pd.read_csv(src('fun', O/'APPROACH_FUNNEL.csv')); fun['armS']=fun.arm.replace({'PID-baseline (comparator)':'PID','PID-baseline':'PID'}).str.replace('FlightDock-','')
ex = pd.read_csv(src('extl', O/'EXTRA_TL_PER_LEG.csv'))
if ex.dt.round(6).nunique()!=1 or len(ex)!=7200: raise SystemExit('REFUSE: the per-leg funnel file is not one tick period over 7,200 legs')
ex['arm']=ex.tag.map(arm_of); ex['sig']=ex.tag.map(sig_of)
ex = ex.merge(bd[['corpus','leg','seed','wind_rung','passed']], left_on=['tag','leg'], right_on=['corpus','leg'], how='inner')
if len(ex)!=7200: raise SystemExit('REFUSE: per-leg funnel rows do not join the board one to one')
sec = ex.raw_inbox_ticks*ex.dt
# the 1e-9 tolerance on the time steps is the producer's (extra_analyses_ticklevel.funnel): 60 ticks x the exact period is 2.9999999999973 s
STEPS = [('within 10 m of the boom-tip', ex.raw_w10.astype(bool)), ('within 3 m', ex.raw_w3.astype(bool)), ('within 1.5 m', ex.raw_w15.astype(bool)),
         ('inside the envelope', ex.raw_inbox_ticks>0), ('inside for 1 s or more (cumulative)', sec>=1.0-1e-9), ('inside for 3 s or more (cumulative)', sec>=3.0-1e-9), ('capture (flag of record)', ex.passed.astype(bool))]
FUNSRC = {'within 10 m of the boom-tip':'within 10 m of the tip','within 3 m':'within 3 m','within 1.5 m':'within 1.5 m','inside the envelope':'inside the M3b box',
          'inside for 1 s or more (cumulative)':'in the box >= 1 s cumulative','inside for 3 s or more (cumulative)':'in the box >= 3 s cumulative','capture (flag of record)':'M3b success (flag of record)'}
def boot_flag(g, flag):
    seed=g.seed.to_numpy(); labels=np.sort(np.unique(seed))
    if list(labels)!=SEED_BLOCK or len(g)!=540: raise SystemExit('REFUSE: funnel cell is not 540 legs on the seed block')
    f=flag.to_numpy(bool); k=np.array([f[seed==s].sum() for s in labels],float); n=np.array([(seed==s).sum() for s in labels],float)
    picks=PB._picks(len(labels)); v=100.0*k[picks].sum(axis=1)/n[picks].sum(axis=1)
    return int(f.sum()), float(np.percentile(v,2.5)), float(np.percentile(v,97.5))
rows=[]; n_fun=0
for sig in (0.25,0.35):
    for arm in ORDER:
        m=(ex.sig==sig)&ex.wind_rung.astype(bool)&(ex.arm==arm); g=ex[m]; rr={'σ':sig,'arm':arm}
        for name, flag in STEPS:
            k,lo,hi = boot_flag(g, flag[m])
            q = fun[(fun.pool=='nine wind stages')&(fun.sigma==sig)&(fun.armS==arm)&(fun.step==FUNSRC[name])]
            if len(q)!=1 or int(str(q.truth.iloc[0]).split(' of ')[0])!=k: raise SystemExit(f'REFUSE: funnel {arm} σ {sig} {name}: {k} does not reproduce APPROACH_FUNNEL.csv')
            n_fun+=1; rr[name]=f"{k} of 540, {100.0*k/540:.1f}% [{lo:.1f}, {hi:.1f}]"
        rows.append(rr)
T12=pd.DataFrame(rows)
emit('T14','Approach funnel, nine wind stages, truth channel: legs reaching each step (of 540)',T12,f"{SRC['extl']} (per-leg steps) joined to {SRC['board']} (seed, pool, capture flag); counts reproduce {SRC['fun']}, whose Wilson intervals are superseded",
     'Each column is the share of the 540 legs that reached that step at least once; from left to right the counts only fall; time inside the envelope is ticks times the tick period, not necessarily continuous. Intervals: two-sided 95% seed-cluster bootstrap (30 seeds resampled whole, 10,000 draws, fixed seed). The last column is the four-condition capture flag of record.')
print(f'funnel check: {n_fun} cells reproduce APPROACH_FUNNEL.csv')
# T13 straight vs racetrack, recomputed from the board with the seed-cluster bootstrap (operator 10-01: every table carries it; the figure CSV's
# Wilson intervals are superseded here; the figure itself awaits regeneration). Counts and point differences must reproduce the CSV or the script refuses.
svr = pd.read_csv(src('svr', O/'figures/STRAIGHT_VS_RACETRACK_POOLED_ACROSS_NINE_WIND_STAGES.csv')); svr['arm']=svr.arm.replace({'PID-baseline (comparator)':'PID','PID-baseline':'PID'}).str.replace('FlightDock-','')
def _per_seed(g):
    ok = g.passed.to_numpy(bool); seed = g.seed.to_numpy(); labels = np.sort(np.unique(seed))
    if list(labels)!=SEED_BLOCK: raise SystemExit('REFUSE: trajectory cell does not cover the seed block')
    return np.array([ok[seed==s].sum() for s in labels], float), np.array([(seed==s).sum() for s in labels], float)
def boot_sum_ratio(d, n):
    picks = PB._picks(len(d)); v = 100.0*d[picks].sum(axis=1)/n[picks].sum(axis=1)
    return float(np.percentile(v,2.5)), float(np.percentile(v,97.5))
def _cell(s):
    k, n = re.match(r'(\d+) of (\d+)', str(s)).groups(); return int(k), int(n)
rows=[]; J['TRAJ']={}
for sig in (0.25, 0.35):
    ks_pid, n_pid = _per_seed(legs('PID', sig, 'straight')); kr_pid, _ = _per_seed(legs('PID', sig, 'racetrack')); d_pid = ks_pid - kr_pid
    for arm in ORDER:
        gs, gr = legs(arm, sig, 'straight'), legs(arm, sig, 'racetrack')
        ks, ns = _per_seed(gs); kr, nr = _per_seed(gr)
        if not (np.all(ns==nr) and int(ns.sum())==270): raise SystemExit(f'REFUSE: {arm} σ {sig} trajectories are not 270 legs each with equal legs per seed')
        slo, shi = boot_sum_ratio(ks, ns); rlo, rhi = boot_sum_ratio(kr, nr)
        d = ks - kr; dlo, dhi = boot_sum_ratio(d, ns); diff = 100.0*d.sum()/ns.sum()
        dd = d - d_pid; ddiff = 100.0*dd.sum()/ns.sum(); ddlo, ddhi = (0.0, 0.0) if arm=='PID' else boot_sum_ratio(dd, ns)
        q = svr[(svr.sigma==sig)&(svr.arm==arm)]
        if len(q)!=1: raise SystemExit(f'REFUSE: trajectory CSV lacks {arm} σ {sig}')
        q = q.iloc[0]
        if _cell(q.straight)!=(int(ks.sum()),270) or _cell(q.racetrack)!=(int(kr.sum()),270) or abs(float(q['straight minus racetrack (pp)'])-round(diff,1))>1e-9 or abs(float(q["that difference minus the PID-baseline's (pp)"])-round(ddiff,1))>1e-9:
            raise SystemExit(f'REFUSE: trajectory counts or differences for {arm} σ {sig} do not reproduce the CSV')
        rows.append({'σ':sig,'arm':arm,'straight: captures of 270, rate [interval]':f"{int(ks.sum())} of 270, {100.0*ks.sum()/270:.1f}% [{slo:.1f}, {shi:.1f}]",
                     'racetrack: captures of 270, rate [interval]':f"{int(kr.sum())} of 270, {100.0*kr.sum()/270:.1f}% [{rlo:.1f}, {rhi:.1f}]",
                     'straight minus racetrack, within seed (pp) [interval]':f"{diff:+.1f} [{dlo:+.1f}, {dhi:+.1f}]",
                     "that difference minus the PID-baseline's (pp) [interval]":('0.0, the baseline' if arm=='PID' else f"{ddiff:+.1f} [{ddlo:+.1f}, {ddhi:+.1f}]")})
        J['TRAJ'][f'{arm}_{sig}']=dict(ks=int(ks.sum()), kr=int(kr.sum()), diff=diff, lo=dlo, hi=dhi, dpid=ddiff, dlo=ddlo, dhi=ddhi)
T13=pd.DataFrame(rows)
emit('T15','Docking-envelope captures by DHC-2 trajectory, nine wind stages pooled (270 legs per trajectory)',T13,f"{SRC['board']}; counts and point differences checked against {SRC['svr']}",
     'Two-sided 95% seed-cluster bootstrap intervals (30 seeds resampled whole, 10,000 draws, fixed seed); the differences are paired within seed. Amplifying; decides no hypothesis.')
# T14 gust agreement
ga = pd.read_csv(src('ga', B20/'comparative/GUST_AGREEMENT_TABLE.csv'))
ga['L']=ga.left.map(arm_of); ga['Rr']=ga.right.map(arm_of); ga['σ']=ga.left.map(sig_of)
T14 = ga[(ga.Rr=='PID')|(ga.L=='PID')][['σ','L','Rr','n_pairs','agree_share_pct','diff_all_pp','ci95_all','diff_agreeing_pp','ci95_agreeing','n_agreeing']].rename(columns={'L':'arm','Rr':'against','agree_share_pct':'pairs with identical gust streams (%)','diff_all_pp':'capture difference, all pairs (pp)','ci95_all':'95% interval, all pairs','diff_agreeing_pp':'capture difference, identical-stream pairs only (pp)','ci95_agreeing':'95% interval, identical-stream pairs','n_agreeing':'identical-stream pairs'})

def _row14(r):   # package 10: every row reads arm minus the PID-baseline (the source lists PID against PIDO once, as PID minus PIDO); one sign format
    r=r.copy(); flip=(r['arm']=='PID')
    d1,d2=float(r['capture difference, all pairs (pp)']),float(r['capture difference, identical-stream pairs only (pp)'])
    (l1,h1),(l2,h2)=_ivl(r['95% interval, all pairs']),_ivl(r['95% interval, identical-stream pairs'])
    if flip: r['arm']=r['against']; d1,d2,(l1,h1),(l2,h2)=-d1,-d2,(-h1,-l1),(-h2,-l2)
    r['capture difference, all pairs (pp)']=f"{d1:+.2f}"; r['capture difference, identical-stream pairs only (pp)']=f"{d2:+.2f}"
    r['95% interval, all pairs']=f"[{l1:+.1f}, {h1:+.1f}]"; r['95% interval, identical-stream pairs']=f"[{l2:+.1f}, {h2:+.1f}]"
    return r
if not ((T14.against=='PID')|((T14.arm=='PID')&(T14.against=='PIDO'))).all(): raise SystemExit('REFUSE: a gust-agreement row is not against the PID-baseline')
T14=T14.apply(_row14, axis=1).drop(columns='against'); T14['o']=T14.arm.map(ORDER.index); T14=T14.sort_values(['σ','o']).drop(columns='o')
gs = pathlib.Path(src('gs', P/'eda_20260918/GUST_STREAMS_SUMMARY.md')).read_text()
m25 = re.search(r'sigma s025: 540 windy keys, (\d+) with every arm on one stream', gs); m35 = re.search(r'sigma s035: 540 windy keys, (\d+) with every arm on one stream', gs)
J['gust_all6_s025']=int(m25.group(1)); J['gust_all6_s035']=int(m35.group(1))
emit('T16','Same-seed gust-stream agreement against the PID-baseline, and the capture difference with and without the disagreeing pairs',T14,f"{SRC['ga']}; {SRC['gs']}",
     f"Across all six arms, {J['gust_all6_s025']} of 540 wind-stage legs at σ 0.25 and {J['gust_all6_s035']} of 540 at σ 0.35 flew on one and the same gust stream in every arm; the rest had at least one arm re-rolled by a warm-up retry. Intervals in this table are the two-sided 95% seed-cluster bootstrap of the source file. Sign: arm minus the PID-baseline in every row (the source lists the PID-baseline against PIDO once, as PID minus PIDO; that row is shown as PIDO minus the PID-baseline).")
# T15 captures by stage six arms
rows=[]
for sig,fn in ((0.25,'DOCKING_ENVELOPE_CAPTURES_BY_WIND_STAGE_SIGMA_025.csv'),(0.35,'DOCKING_ENVELOPE_CAPTURES_BY_WIND_STAGE_SIGMA_035.csv')):
    c = pd.read_csv(src(f'cap{sig}', O/'figures'/fn)); c['σ']=sig; rows.append(c)
cap = pd.concat(rows); cap['arm']=cap.arm.replace({'PID-baseline (comparator)':'PID','PID-baseline':'PID'}).str.replace('FlightDock-','')
BOOTCOL = 'rate [two-sided 95% seed-cluster bootstrap]'
if BOOTCOL not in cap.columns: raise SystemExit('REFUSE: captures-by-stage CSV lacks the seed-cluster bootstrap column')
def stage_key(s): return -1 if s=='Calm Control' else (99 if str(s).startswith('All') else int(str(s).split(' ')[0]))
cap['cell']=cap['docking-envelope captures'].astype(str)+', '+cap[BOOTCOL].astype(str)
for sig, tid in ((0.25,'T17'),(0.35,'T18')):   # package 10: one table per σ, rows = stage, columns = the six arms
    w = cap[cap['σ']==sig].pivot_table(index='stage', columns='arm', values='cell', aggfunc='first')
    w = w.reindex(sorted(w.index, key=stage_key))[ORDER].reset_index().rename(columns={'stage':'wind stage'})
    if w.isna().any().any() or len(w)!=11: raise SystemExit('REFUSE: the captures-by-stage pivot has a hole or the wrong number of stages')
    emit(tid, f'Docking-envelope captures by wind stage and arm, σ = {sig} (60 legs per stage and arm; the nine-stage pool is the last row)', w, SRC[f'cap{sig}'],
         'Cell: captures of 60 (of 540 for the pool), rate [two-sided 95% seed-cluster bootstrap interval] (30 seeds resampled whole, 10,000 draws). Calm Control (no wind, no turbulence) enters no pool. The paired statistics of record are in the H2 and H3 tables.')
# T16 clearance and guard record by stage, terminal approach corridor only
ev = pd.read_csv(src('events', B18/'ticklevel/ABORT_EVENTS.csv'), low_memory=False); ev['inc'] = ev.inside_10m.astype(str).str.lower().eq('true')
ev['armS'] = ev.tag.map(arm_of); ev['sg'] = ev.tag.map(sig_of); ev['stage'] = np.where(ev.wind_label.eq('S0'), 'S0 calm control stage', ev.wind_ms.astype(int).astype(str) + ' m/s')
pl = pd.read_csv(src('perleg', B18/'ticklevel/ABORT_PER_LEG.csv'), low_memory=False); pl['armS'] = pl.tag.map(arm_of); pl['sg'] = pl.tag.map(sig_of)
pl['stage'] = np.where(pl.wind_label.eq('S0'), 'S0 calm control stage', pl.wind_ms.astype(int).astype(str) + ' m/s')
T16 = sbs[['sigma','armS','stage','n','captured','entered corridor','never reached corridor','legs over 2.00 m in corridor']].rename(columns={'sigma':'σ','armS':'arm'})
if not (T16.n==60).all(): raise SystemExit('REFUSE: a safety-by-stage cell is not 60 legs')
def corridor_counts(sig, arm, stage):
    e = ev[(ev.sg==sig)&(ev.armS==arm)&(ev.stage==stage)&ev.inc]; l = pl[(pl.sg==sig)&(pl.armS==arm)&(pl.stage==stage)]
    if len(l)!=60: raise SystemExit('REFUSE: a per-leg cell is not 60 legs')
    return int((l.peak_up_m_in_window>CONTACT_M).sum()), int((e.kind=='ceiling_fire').sum()), int((e.kind=='overtake_fire').sum())
cc = [corridor_counts(r['σ'], r['arm'], r['stage']) for r in T16.to_dict('records')]
T16['legs over 3.42 m in corridor'] = [c[0] for c in cc]; T16['altitude-guard fires in corridor'] = [c[1] for c in cc]; T16['overtake-guard fires in corridor'] = [c[2] for c in cc]
whole = sbs.set_index(['sigma','armS','stage'])
for r in T16.to_dict('records'):
    w = whole.loc[(r['σ'], r['arm'], r['stage'])]
    if r['altitude-guard fires in corridor'] > int(w['altitude-guard fires']) or r['overtake-guard fires in corridor'] > int(w['overtake-guard fires']): raise SystemExit('REFUSE: corridor fires exceed the whole-leg fires of record')
T16['o']=T16.arm.map(ORDER.index); T16['sk']=T16.stage.map(lambda s: -1 if str(s).startswith('S0') else int(str(s).split(' ')[0])); T16=T16.sort_values(['σ','o','sk']).drop(columns=['o','sk','n'])
for sg, tid in ((0.25,'T19'),(0.35,'T20')):
    emit(tid, f'Clearance and guard record by arm and wind stage, six arms, σ = {sg}, terminal approach corridor only', T16[T16['σ']==sg].drop(columns='σ'), f"{SRC['sbs']}; {SRC['events']}; {SRC['perleg']}",
         'Every cell holds the 60 legs of its stage. Legs over 2.00 m are exclusion-zone violations inside the corridor on the truth channel; a leg over 3.42 m would be airframe contact. Guard fires are counted only when the truth range to the boom-tip was 10 m or less at the fire (the terminal approach corridor), from the fire events of record.')
# every arm consolidated: the in-chapter verdict table
rows=[]; MET={}
for arm in ORDER:
    rr={'arm':arm,'configuration':CONFIG[arm]}; met=[]; judged = arm not in ('PID','PIDO')
    for h in ('H1','H2','H3'):
        if h=='H2' and arm=='PID':
            rr['H2 value [interval]']='0 pp, the baseline comparator'; rr['H2 verdict']='baseline comparator (not graded)'; continue
        r=pv[(pv.hypothesis==h)&(pv.arm==arm)].iloc[0]
        if bool(r.judged)!=judged: raise SystemExit(f'REFUSE: judged flag for {arm} {h} disagrees with the arm list')
        rr[f'{h} value [interval]']=(f"{r.value:.3f} [{r.ci95_lo:.3f}, {r.ci95_hi:.3f}] m" if h=='H1' else f"{r.value:+.2f} [{r.ci95_lo:+.2f}, {r.ci95_hi:+.2f}] pp" if h=='H2' else f"{r.value:.2f} [{r.ci95_lo:.2f}, {r.ci95_hi:.2f}] pp")
        rr[f'{h} verdict']=verdict(h,r) if judged else 'comparator (not graded)'
        if judged and rr[f'{h} verdict']==f'{h} met': met.append(h)
    rr['hypotheses met']=('all three' if len(met)==3 else 'none' if not met else ' and '.join(met) if len(met)==2 else f'{met[0]} only') if judged else 'comparator (not graded)'
    MET[arm]=met if judged else None; rows.append(rr)
rows18=[]
HDR18={'H1':'H1: pooled corridor p95, truth channel, σ 0.25 (m); bar: upper bound < 2.00 m. Value [95% interval], verdict','H2':'H2: same-seed capture difference against the PID-baseline, σ 0.25 (pp); bar: lower bound > 0 pp. Value [95% interval], verdict','H3':'H3: same-seed capture decrease, σ 0.25 to σ 0.35 (pp); bar: upper bound < 35 pp. Value [95% interval], verdict'}   # 2026-10-03: the registered statistic and the bar in the header
for rr in rows:   # package 10: one cell per hypothesis (value, interval and the verdict word); the configuration column is the table “H1: p95 Peak Upward Deviation, Light Turbulence”
    v={'arm':rr['arm']}
    for h in ('H1','H2','H3'):
        val, vd = rr[f'{h} value [interval]'], rr[f'{h} verdict']
        v[HDR18[h]] = ('0 pp, baseline comparator (not graded)' if (h=='H2' and rr['arm']=='PID') else f"{val}, met" if vd==f'{h} met' else f"{val}, not met" if vd==f'{h} not met' else f"{val}, comparator (not graded)")
    v['hypotheses met']=rr['hypotheses met']; rows18.append(v)
T18=pd.DataFrame(rows18)
emit('T21','Hypothesis verdicts by arm: which arm met which hypothesis (the four FlightDock arms are judged; the PID-baseline and PIDO are comparators, not graded)',T18,SRC['pv'],
     'One-sided test at α = 0.025 through the two-sided 95% seed-cluster bootstrap interval (the table “Hypothesis Tests”); the point estimate is reported beside the interval; the one-sided bootstrap p of each cell is in the H1, H2 and H3 tables.')
J['T21']=rows; J['MET']=MET
# write
hdr = f"# Results tables (built {datetime.datetime.now():%Y-%m-%d} by results_tables.py)\n\nEvery cell is computed from the shipped board, readouts and derived files; nothing is typed. Arm order: S1O, S1, A2O, A2, PIDO, PID (each controller with the 0.25 m offset first, then without; the PID-baseline last).\n\n"
(P/'results_tables/RESULTS_TABLES.md').write_text(hdr + '\n'.join(out))
json.dump(J, open(P/'results_tables/RESULTS_NUMBERS.json','w'), indent=1, default=float)
print('wrote results_tables/RESULTS_TABLES.md and results_tables/RESULTS_NUMBERS.json;', len(out), 'tables')
