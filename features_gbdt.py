#!/usr/bin/env python3
"""Handcrafted causal features + HistGradientBoosting vs Kalshi mid / p_rw. Research only, no orders.
Protocol: train on first 80% of early half (open_ts < median), validate (early stop + edge choice) on last 20%,
score ONCE on late_half. Configs tried: 1 GBDT hyperparam set x 3 offsets, 1 stacker x 3 offsets, 3 edges
(edge picked on validation) => see N_CONFIGS."""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import eval as E
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

HERE = Path(__file__).parent; DATA = HERE / "data"
N_CONFIGS = 3 + 3 + 9  # gbdt per k, stacker per k, edges per k (gbdt only counted for selection)
logit = lambda p: np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))


def load():
    mk = pd.read_parquet(DATA / "markets.parquet").dropna(subset=["strike"]).sort_values("open_ts").reset_index(drop=True)
    b = pd.read_parquet(DATA / "btc_1m.parquet").set_index("ts")
    t0 = int(b.index[0]); n = int((b.index[-1] - t0) // 60) + 1
    g = b.reindex(t0 + 60 * np.arange(n))
    kc = pd.read_parquet(DATA / "kalshi_candles.parquet")
    kc = kc.assign(**{c: pd.to_numeric(kc[c]) for c in ["bid", "ask", "vol"]}).sort_values(["ticker", "ts"])
    kd = {t: (d.ts.to_numpy(), ((d.bid + d.ask) / 2).to_numpy(), d.vol.to_numpy()) for t, d in kc.groupby("ticker")}
    return mk, t0, {c: g[c].to_numpy(float) for c in ["close", "high", "low", "volume"]}, kd


def rowfeat(r, mk_open, mk_close, mk_res, mk_tick, t0, B, kd):
    """Features for one (market, k). Uses only B[..] candles starting <= t-60 and Kalshi candles ts <= t."""
    t = r.open_ts + 60 * r.k
    end = (t - 60 - t0) // 60 + 1
    c = B["close"][:end]; hi = B["high"][:end]; lo = B["low"][:end]; v = B["volume"][:end]
    last = c[-1]; f = {}
    ts, mid, vol = kd[r.ticker]; m = ts <= t; ts, mid, vol = ts[m], mid[m], vol[m]
    f["mid"] = r.mid; f["spread"] = r.ask - r.bid; f["logit_mid"] = logit(r.mid)
    for d in sorted({1, 3, r.k}):
        j = np.where(ts == t - 60 * d)[0]
        f[f"dmid_{d}"] = r.mid - mid[j[0]] if len(j) else np.nan
    f["dmid_k"] = f.pop(f"dmid_{r.k}") if r.k not in (1, 3) else f[f"dmid_{r.k}"]
    f["kvol_sofar"] = vol.sum()
    p = np.searchsorted(mk_close, t, side="right") - 1  # markets sorted by open_ts; close_ts monotone too
    f["prev_res"] = f["prev_mid"] = np.nan
    if p >= 0 and mk_close[p] <= t and mk_tick[p] in kd:
        pts, pmid, _ = kd[mk_tick[p]]; pm = pts <= t
        f["prev_res"] = mk_res[p]; f["prev_mid"] = pmid[pm][-1] if pm.any() else np.nan
    lr = np.diff(np.log(c))
    f["dist_sig"] = math.log(last / r.strike) / (np.nanstd(lr[-120:]) * math.sqrt(r.h) + 1e-12)
    f["h"] = r.h
    for n in (1, 5, 15, 60): f[f"ret_{n}"] = math.log(last / c[-1 - n])
    for n in (15, 60, 240): f[f"rv_{n}"] = np.nanstd(lr[-n:])
    f["range_15"] = (np.nanmax(hi[-15:]) - np.nanmin(lo[-15:])) / last
    w = v[-240:]; f["cvol_z"] = (v[-1] - np.nanmean(w)) / (np.nanstd(w) + 1e-9)
    f["cvol_z5"] = (np.nansum(v[-5:]) / 5 - np.nanmean(w)) / (np.nanstd(w) + 1e-9)
    mod = (t % 86400) / 60; f["mod_sin"] = math.sin(2 * math.pi * mod / 1440); f["mod_cos"] = math.cos(2 * math.pi * mod / 1440)
    f["dow"] = (t // 86400 + 4) % 7
    return f


def features(df, mk, t0, B, kd):
    a = (mk.open_ts.to_numpy(), mk.close_ts.to_numpy(), mk.result.to_numpy(), mk.ticker.to_numpy())
    return pd.DataFrame([rowfeat(r, *a, t0, B, kd) for r in df.itertuples()], index=df.index)


def causal_check(df, mk, t0, B, kd, n=15):
    """Blank all data after t and recompute: features must be identical."""
    a = (mk.open_ts.to_numpy(), mk.close_ts.to_numpy(), mk.result.to_numpy(), mk.ticker.to_numpy())
    for r in df.sample(n, random_state=0).itertuples():
        t = r.open_ts + 60 * r.k; cut = (t - 60 - t0) // 60 + 1
        B2 = {k: np.where(np.arange(len(x)) < cut, x, np.nan) for k, x in B.items()}
        kd2 = {k: (x[0][x[0] <= t], x[1][x[0] <= t], x[2][x[0] <= t]) for k, x in kd.items()}
        f1, f2 = rowfeat(r, *a, t0, B, kd), rowfeat(r, *a, t0, B2, kd2)
        for k in f1: assert (np.isnan(f1[k]) and np.isnan(f2[k])) or f1[k] == f2[k], (k, f1[k], f2[k])
        pi = np.searchsorted(a[1], t, side="right") - 1
        assert pi < 0 or a[1][pi] <= t
    print("causal check ok")


def pnls(p, bid, ask, y, edge):
    out = []
    for pi, b, a_, yi in zip(p, bid, ask, y):
        if pi - a_ - E.fee(a_) > edge and a_ < 1: out.append(yi - a_ - E.fee(a_))
        elif (1 - pi) - (1 - b) - E.fee(1 - b) > edge and b > 0: out.append((1 - yi) - (1 - b) - E.fee(1 - b))
    return np.array(out)


def tr(p, d, e):
    x = E.trade(p, d.bid.to_numpy(), d.ask.to_numpy(), d.y.to_numpy(), e)
    assert abs(x["pnl"] - pnls(p, d.bid.to_numpy(), d.ask.to_numpy(), d.y.to_numpy(), e).sum()) < 1e-9
    return x


def main():
    df, _ = E.build()  # cols: ticker open_ts k h strike y bid ask mid last p_rw
    mk, t0, B, kd = load()
    df = df.sort_values(["open_ts", "k"]).reset_index(drop=True)
    X = features(df, mk, t0, B, kd); causal_check(df, mk, t0, B, kd)
    cut = df.open_ts.quantile(0.5); res = {"n_configs_tried": N_CONFIGS}
    rng = np.random.default_rng(0)
    for k in E.OFFSETS:
        d = df[df.k == k]; Xk = X.loc[d.index]
        early = d[d.open_ts < cut]; late = d[d.open_ts >= cut]
        nv = int(len(early) * 0.2); tr_i, va_i = early.index[:-nv], early.index[-nv:]
        dt, dv, dl = df.loc[tr_i], df.loc[va_i], late
        m = HGB(loss="log_loss", learning_rate=0.03, max_iter=300, max_leaf_nodes=6, min_samples_leaf=40,
                l2_regularization=5.0, early_stopping=False, random_state=0).fit(Xk.loc[tr_i], dt.y)
        ll = [E.scores(p[:, 1], dv.y.to_numpy())["logloss"] for p in m.staged_predict_proba(Xk.loc[va_i])]
        it = int(np.argmin(ll)) + 1
        stage = lambda idx: list(m.staged_predict_proba(Xk.loc[idx]))[it - 1][:, 1]
        pv, pl = stage(va_i), stage(late.index)
        st = lambda d_: np.c_[logit(d_.mid), logit(d_.p_rw)]
        lr = LogisticRegression(C=1.0).fit(st(df.loc[early.index]), early.y)
        sv, sl = lr.predict_proba(st(dv))[:, 1], lr.predict_proba(st(dl))[:, 1]
        # edge chosen on validation: best per_trade with >=20 trades, else none
        vt = {e: tr(pv, dv, e) for e in E.EDGES}
        ok = [e for e in E.EDGES if vt[e]["trades"] >= 20]
        ebest = max(ok, key=lambda e: vt[e]["per_trade"]) if ok else None
        r = dict(n_train=len(dt), n_val=len(dv), n_late=len(dl), best_iter=it, val_edge_trades={str(e): vt[e] for e in E.EDGES},
                 chosen_edge=ebest, val_logloss_gbdt=ll[it - 1], val_logloss_mid=E.scores(dv.mid.to_numpy(), dv.y.to_numpy())["logloss"])
        for name, p in [("mid", dl.mid.to_numpy()), ("p_rw", dl.p_rw.to_numpy()), ("gbdt", pl), ("stacker", sl)]:
            r[name] = dict(**E.scores(p, dl.y.to_numpy()), trade={str(e): tr(p, dl, e) for e in E.EDGES})
        if ebest is not None:
            x = pnls(pl, dl.bid.to_numpy(), dl.ask.to_numpy(), dl.y.to_numpy(), ebest)
            if len(x):
                bs = [rng.choice(x, len(x)).mean() for _ in range(2000)]
                r["gbdt_chosen_edge_ci95_per_trade"] = dict(edge=ebest, trades=len(x), mean=float(x.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
        pi = permutation_importance(m, Xk.loc[va_i], dv.y, scoring="neg_log_loss", n_repeats=5, random_state=0)
        r["top10_perm_importance_val"] = [(c, float(v)) for c, v in sorted(zip(Xk.columns, pi.importances_mean), key=lambda z: -z[1])[:10]]
        res[f"k{k}_late_half"] = r
        print(f"k={k} train/val/late={len(dt)}/{len(dv)}/{len(dl)} iter={it} edge={ebest}")
        for nme in ["mid", "p_rw", "gbdt", "stacker"]:
            print("  %-8s brier %.4f ll %.4f " % (nme, r[nme]["brier"], r[nme]["logloss"]),
                  " ".join(f"e{e}:{v['trades']}t/{v['per_trade']:+.3f}" for e, v in r[nme]["trade"].items()))
        print("  CI", r.get("gbdt_chosen_edge_ci95_per_trade"), "\n  top", r["top10_perm_importance_val"][:5])
    (HERE / "results_gbdt.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
