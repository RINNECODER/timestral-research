import glob, json
import numpy as np, pandas as pd
rng = np.random.default_rng(0)
fee = lambda p: 0.0175 * p * (1 - p)
assert abs(fee(0.40) - 0.0042) < 1e-12 and abs(fee(0.5) - 0.004375) < 1e-12  # hand example
# taker_side=='yes' => taker bought YES (hits ask); 'no' => taker bought NO (sold YES, hits bid)
M = pd.read_parquet('data/all_markets.parquet', columns=['ticker','open_ts','close_ts','result'])
T = pd.concat(pd.read_parquet(f, columns=['ticker','created_time','yes_price','taker_side']) for f in glob.glob('data/trades/chunk_*.parquet'))
C = pd.concat(pd.read_parquet(f) for f in glob.glob('data/hist_candles/c_*.parquet')).drop_duplicates(['ticker','ts'])
M = M[M.ticker.isin(set(T.ticker)) & M.ticker.isin(set(C.ticker))]
tr = {k: (g.created_time.values, g.yes_price.values, (g.taker_side.values == 'yes')) for k, g in T.groupby('ticker')}
cd = {k: g.set_index('ts')[['bid','ask']] for k, g in C.groupby('ticker')}
rows = []  # market, k, variant, filled-pnl-sum, nfills, wins, quote-sum
def fill(t, buyer_yes, px, close, strict, side):
    # side 'A': need taker NO trade price <= (<) q ; 'B': taker YES trade price >= (>) q
    ct, yp, ty = t
    m = ct > px[0]
    assert (not m.any()) or ct[m].min() > px[0]  # only trades after quote time
    m &= ct <= close
    q = px[1]
    if side == 'A': m &= ~ty & ((yp < q) if strict else (yp <= q))
    else: m &= ty & ((yp > q) if strict else (yp >= q))
    return m.any()
for r in M.itertuples():
    if r.ticker not in cd or r.ticker not in tr: continue
    c = cd[r.ticker]
    for k in (1, 5, 10):
        t = r.open_ts + 60 * k
        if t not in c.index: continue
        b, a = c.loc[t, 'bid'], c.loc[t, 'ask']
        if not (0 < b < a < 1): continue
        res = r.result
        spread2 = round((a - b) * 100) >= 2
        for var, strict in (('opt', False), ('pess', True)):
            for name, side, q in (('A', 'A', b), ('B', 'B', a), ('C_A', 'A', b + .01 if spread2 else None), ('C_B', 'B', a - .01 if spread2 else None)):
                if q is None: continue
                f = fill(tr[r.ticker], 0, (t, q), r.close_ts, strict, side)
                if side == 'A': p, w, qp = res - q - fee(q), res, q
                else: p, w, qp = (1 - res) - (1 - q) - fee(q), 1 - res, 1 - q
                rows.append((r.ticker, r.open_ts, k, var, name, int(f), p if f else 0., w if f else 0, qp if f else 0.))
R = pd.DataFrame(rows, columns=['tk','open_ts','k','var','v','f','pnl','win','qp'])
# D = A+B joined, per market
D = R[R.v.isin(['A','B'])].groupby(['tk','open_ts','k','var'], as_index=False)[['f','pnl','win','qp']].sum(); D['v'] = 'D'
R = pd.concat([R, D])
mid = M.open_ts.median()
def stat(g):
    gm = g.groupby('tk')[['f','pnl']].sum()
    n = int(g.f.sum())
    if n == 0: return dict(n_mkts=len(gm), fills=0)
    F, P = gm.f.values, gm.pnl.values
    idx = rng.integers(0, len(gm), (1000, len(gm)))
    bs = P[idx].sum(1) / np.maximum(F[idx].sum(1), 1)
    nq = g.v.isin(['D']).any() and 2 or 1
    return dict(n_mkts=len(gm), fills=n, fill_rate=n / (len(gm) * nq), mean_pnl=P.sum() / n, total=P.sum(),
                ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                p_win=float(g.win.sum() / n), mean_quote=float(g.qp.sum() / n))
out = {}
for (var, v, k), g in R.groupby(['var','v','k']):
    out[f'{var}/{v}/k{k}'] = dict(all=stat(g), early=stat(g[g.open_ts < mid]), late=stat(g[g.open_ts >= mid]))
json.dump(out, open('results_maker.json', 'w'), indent=1, default=float)
print(len(M), 'markets')
for key, s in out.items():
    a = s['all']
    if a['fills']: print(key, f"fill {a['fill_rate']:.2f} pnl {a['mean_pnl']*100:+.2f}c ci[{a['ci95'][0]*100:+.2f},{a['ci95'][1]*100:+.2f}] tot {a['total']:.0f} win {a['p_win']:.3f} q {a['mean_quote']:.3f} | early {s['early'].get('mean_pnl',0)*100:+.2f} late {s['late'].get('mean_pnl',0)*100:+.2f}")
