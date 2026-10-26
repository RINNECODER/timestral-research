import glob, json, math, subprocess
import numpy as np, pandas as pd
from math import erf
D = 'live/'
fee = lambda p: 0.07 * p * (1 - p)
Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / math.sqrt(2))))
cat = lambda p: pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(D + p + '_*.parquet')))
k, s = cat('kalshi'), cat('spot')
out = {}; P = lambda *a: print(*a)

# 1. grid (t in epoch seconds despite "ms" in brief; checked)
s['sec'] = np.floor(s.t).astype(int)
t0, t1 = int(min(s.sec.min(), k.t.min())), int(max(s.sec.max(), k.t.max()))
G = pd.RangeIndex(t0, t1 + 1)
V = {}
for v, g in s.sort_values('t').groupby('venue'):
    last = g.groupby('sec').price.last().reindex(G)
    age = pd.Series(np.where(last.notna(), G, np.nan), index=G).ffill()
    V[v] = last.ffill().where((G - age) <= 10)  # stale >10s -> drop venue
S = pd.DataFrame(V).median(axis=1)
r = np.log(S).diff()
sig = r.rolling(900, min_periods=300).std()  # window ends at t: past+current return only
# causality check: perturbing the future must not change sigma at t
tt = len(G) // 2; r2 = r.copy(); r2.iloc[tt + 1:] = 9
assert np.isclose(r2.rolling(900, min_periods=300).std().iloc[tt], sig.iloc[tt], equal_nan=True)

k['sec'] = np.floor(k.t).astype(int)
mk = {}
for tk, g in k.sort_values('t').groupby('ticker'):
    g = g.groupby('sec').last()
    idx = pd.RangeIndex(g.index.min(), g.index.max() + 1)
    g = g.reindex(idx)
    age = pd.Series(np.where(g.yes_bid.notna(), idx, np.nan), index=idx).ffill()
    d = pd.DataFrame({'bid': g.yes_bid.ffill(), 'ask': g.yes_ask.ffill()}).where(((idx - age) <= 5).values[:, None] & np.ones((1, 2), bool))
    d['mid'] = (d.bid + d.ask) / 2; d['strike'] = g.strike.iloc[0]; d['close'] = g.close_ts.iloc[0]
    d['S'] = S.reindex(idx).values; d['sig'] = sig.reindex(idx).values
    tau = (d.close - idx).clip(lower=1)
    d['tau'] = tau
    d['pf'] = Phi(np.log(d.S / d.strike) / (d.sig * np.sqrt(tau)))
    d['ret'] = r.reindex(idx).values
    mk[tk] = d
P(f'data: {len(mk)} markets, {len(k)} kalshi rows, {len(s)} trades, span {(t1-t0)/60:.0f} min (SMALL SAMPLE)')
out['n_markets'] = len(mk)

# 3. lead-lag
lags = range(-30, 31)
def xc(col):
    res = []
    for L in lags:
        a, b = [], []
        for d in mk.values():
            x = d[col].diff(); y = d.mid.diff()
            m = (d.tau > 60)
            a.append(y[m]); b.append(x.shift(L)[m])  # L>0: x leads mid by L s
        a, b = pd.concat(a), pd.concat(b); ok = a.notna() & b.notna()
        res.append(np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 50 else np.nan)
    return np.array(res)
for col in ['pf', 'ret']:
    c = xc(col); i = np.nanargmax(np.abs(c))
    P(f'xcorr d(mid) vs d({col}): peak lag {list(lags)[i]:+d}s corr {c[i]:.3f} (lag0 {c[30]:.3f}, +1 {c[31]:.3f}, +5 {c[35]:.3f}); + = spot leads')
    out['xcorr_' + col] = {'peak_lag': list(lags)[i], 'peak': float(c[i]), 'all': [None if np.isnan(x) else float(x) for x in c]}
X = 2.0; ev = []
for d in mk.values():
    z = np.log(d.S).diff(3) / (d.sig * math.sqrt(3))
    last = -99
    for i in np.where((z.abs() > X) & (d.tau > 90))[0]:
        if i - last < 10 or i + 30 >= len(d) or i < 1: continue
        last = i; sg = np.sign(z.iloc[i])
        ev.append((sg * (d.pf.iloc[i] - d.pf.iloc[i - 3]), [sg * (d.mid.iloc[i + j] - d.mid.iloc[i]) for j in (0, 1, 2, 5, 10, 20, 30)]))
if ev:
    fp = np.nanmean([e[0] for e in ev]); mm = np.nanmean([e[1] for e in ev], axis=0)
    half = next((j for j, v in zip((0, 1, 2, 5, 10, 20, 30), mm) if v >= fp / 2), None)
    P(f'impulse |3s move|>{X}sig: n={len(ev)}, fair moved {fp:+.3f}; mean mid change after event at +0,1,2,5,10,20,30s: ' + ' '.join(f'{v:+.3f}' for v in mm) + f'; half-fair reached at {half}s')
    out['impulse'] = {'n': len(ev), 'fair_move': float(fp), 'mid_after': [float(v) for v in mm], 'half_s': half}
else: P('impulse: no events'); out['impulse'] = None

# 4. executable edge sim
res = {}
def result(tk):
    if tk not in res:
        try:
            m = json.loads(subprocess.run(['curl','-s','-m','10',f'https://api.elections.kalshi.com/trade-api/v2/markets/{tk}'],capture_output=True,text=True).stdout)['market']
            res[tk] = {'yes': 1, 'no': 0}.get(m.get('result'))
        except Exception: res[tk] = None
    return res[tk]
settled = {tk: result(tk) for tk in mk}
P(f'settled via REST: {sum(v is not None for v in settled.values())}/{len(mk)} markets (unsettled skipped)')
rng = np.random.default_rng(0)
out['sim'] = {}
for e in (0.03, 0.05, 0.10):
    for L in (1, 2, 5):
        tr = []  # (market, pnl, win)
        for tk, d in mk.items():
            y = settled[tk]
            if y is None: continue
            done = set(); idx = d.index.values
            for i in range(len(d) - L):
                row = d.iloc[i]
                if row.tau <= 60 or np.isnan(row.pf) or np.isnan(row.ask) or np.isnan(row.bid): continue
                for side in ('yes', 'no'):
                    if side in done: continue
                    pa = row.ask if side == 'yes' else 1 - row.bid
                    pf = row.pf if side == 'yes' else 1 - row.pf
                    if pf - pa - fee(pa) > e:
                        f = d.iloc[i + L]  # fill uses only t+L (>= signal time)
                        assert idx[i + L] >= idx[i] + L
                        px = f.ask if side == 'yes' else 1 - f.bid
                        if np.isnan(px): continue
                        done.add(side); w = int(y == (side == 'yes'))
                        tr.append((tk, w - px - fee(px), w))
        if not tr: out['sim'][f'{e}_{L}'] = None; P(f'e={e} L={L}: no trades'); continue
        df = pd.DataFrame(tr, columns=['m', 'pnl', 'w']); g = df.groupby('m').pnl.sum()
        bs = [rng.choice(g.values, len(g)).sum() for _ in range(2000)]
        o = dict(n=len(df), win=float(df.w.mean()), per=float(df.pnl.mean()), tot=float(df.pnl.sum()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], mkts=len(g))
        out['sim'][f'{e}_{L}'] = o
        P(f'e={e:.2f} L={L}: n={o["n"]} ({o["mkts"]} mkts) win={o["win"]:.2f} pnl/trade={o["per"]:+.3f} total={o["tot"]:+.2f} CI95[{o["ci"][0]:+.2f},{o["ci"][1]:+.2f}]')
P('notes: fair ignores 60s BRTI averaging (taus>60 only); spot=coinbase/kraken/bitstamp median, not BRTI; strike from Kalshi.')
json.dump(out, open('results_latency.json', 'w'), indent=1, default=float)
