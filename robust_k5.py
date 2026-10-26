#!/usr/bin/env python3
"""Pre-registered PLAN.md #9 robustness for k5 signal. Reuses eval_big builder/features/split/HGB config. python3 -I robust_k5.py"""
import sys, json
from pathlib import Path
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import eval_big as eb, eval as E
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
KS = [3, 4, 5, 6, 7]; EDGE = 0.02
E.OFFSETS = KS; eb.CTX = 121           # separate ctx file; don't clobber the 512 memmap
_tp = pd.DataFrame.to_parquet; pd.DataFrame.to_parquet = lambda *a, **k: None   # keep samples_big.parquet intact
df, mk, t0, B = eb.build(); pd.DataFrame.to_parquet = _tp
kc = eb.load_candles(); kc["mid"] = (kc.bid + kc.ask) / 2
kc = kc[kc.ticker.isin(set(df.ticker) | set(mk.ticker))]
kd = {t: (g.ts.to_numpy(), g.mid.to_numpy(), g.vol.to_numpy()) for t, g in kc.groupby("ticker")}
X = eb.featurize(df, mk, t0, B, kd)
ot = np.sort(df.open_ts.unique()); c1, c2 = ot[int(.60 * len(ot))], ot[int(.75 * len(ot))]
part = np.where(df.open_ts < c1, "train", np.where(df.open_ts < c2, "val", "holdout"))
nxt = kc.drop_duplicates(["ticker", "ts"]).set_index(["ticker", "ts"])[["bid", "ask"]]
rng = np.random.default_rng(0); logit = eb.logit
def summ(p, d):
    pn = eb.pnl_vec(p, d, EDGE); return eb.boot_ci(pn, d, rng) or dict(trades=0, mean=None, lo=None, hi=None)
res, sig5 = {}, {}
for k in KS:
    g = df[df.k == k]; Xk = X.loc[g.index]; P = part[g.index]
    dt, dv, dl = g[P == "train"], g[P == "val"], g[P == "holdout"]
    m = HGB(loss="log_loss", learning_rate=0.03, max_iter=300, max_leaf_nodes=6, min_samples_leaf=40, l2_regularization=5.0,
            early_stopping=False, random_state=0).fit(Xk.loc[dt.index], dt.y)
    ll = [E.scores(p[:, 1], dv.y.to_numpy())["logloss"] for p in m.staged_predict_proba(Xk.loc[dv.index])]
    it = int(np.argmin(ll)) + 1
    pg = list(m.staged_predict_proba(Xk.loc[dl.index]))[it - 1][:, 1]
    st = lambda d_: np.c_[logit(d_.mid), logit(d_.p_rw)]
    lr = LogisticRegression(C=1.0).fit(st(dt), dt.y); ps = lr.predict_proba(st(dl))[:, 1]
    res[f"k{k}"] = dict(n_holdout=len(dl), gbdt_iter=it, gbdt=summ(pg, dl), stacker=summ(ps, dl))
    if k == 5: sig5 = dict(dl=dl, pg=pg, ps=ps)
    print(k, res[f"k{k}"], flush=True)
dl, pg, ps = sig5["dl"], sig5["pg"], sig5["ps"]
mon = pd.to_datetime(dl.open_ts, unit="s").dt.strftime("%Y-%m").to_numpy()
res["k5_months"] = {mo: dict(gbdt=summ(pg[mon == mo], dl[mon == mo]), stacker=summ(ps[mon == mo], dl[mon == mo])) for mo in sorted(set(mon))}
# (c) next-minute fill: same signals, bid/ask of candle at ts+60
idx = pd.MultiIndex.from_arrays([dl.ticker.to_numpy(), (dl.open_ts + 60 * dl.k + 60).to_numpy()])
nb = nxt.reindex(idx); ok = (nb.bid.notna() & nb.ask.notna() & (nb.bid >= 0) & (nb.bid < nb.ask) & (nb.ask <= 1)).to_numpy()
d2 = dl[ok].assign(bid=nb.bid.to_numpy()[ok], ask=nb.ask.to_numpy()[ok])
res["k5_next_minute"] = dict(n_available=int(ok.sum()), n_holdout=len(dl), gbdt=summ(pg[ok], d2), stacker=summ(ps[ok], d2))
# YES/NO split and price buckets (gbdt, k5, t fill)
b, a, y = dl.bid.to_numpy(), dl.ask.to_numpy(), dl.y.to_numpy()
buy = (pg - a - E.fee(a) > EDGE) & (a < 1); sell = ~buy & ((1 - pg) - (1 - b) - E.fee(1 - b) > EDGE) & (b > 0)
pn = eb.pnl_vec(pg, dl, EDGE); px = np.where(buy, a, 1 - b)   # price paid per contract
res["k5_gbdt_side"] = dict(yes=int(buy.sum()), no=int(sell.sum()), frac_yes=float(buy.sum() / max(buy.sum() + sell.sum(), 1)))
res["k5_gbdt_price_bucket"] = {}
for nm, lo, hi in [("<.3", 0, .3), (".3-.7", .3, .7), (">.7", .7, 1.01)]:
    s = (buy | sell) & (px >= lo) & (px < hi)
    res["k5_gbdt_price_bucket"][nm] = dict(trades=int(s.sum()), per_trade=float(pn[s].mean()) if s.any() else None)
pos = lambda r: r["mean"] is not None and r["mean"] > 0
A = pos(res["k4"]["gbdt"]) and pos(res["k6"]["gbdt"])
Bc = sum(pos(v["gbdt"]) for v in res["k5_months"].values()) >= 2
C = pos(res["k5_next_minute"]["gbdt"])
res["criteria"] = dict(a_k4_k6_positive=bool(A), b_ge2_of_3_months_positive=bool(Bc), c_next_minute_positive=bool(C), PASS=bool(A and Bc and C), judged_on="gbdt")
print(json.dumps({k: res[k] for k in res if k not in KS and not k.startswith("k") or k.startswith("k5_") or k == "criteria"}, indent=1))
(HERE / "results_robust_k5.json").write_text(json.dumps(res, indent=1))
