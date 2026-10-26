#!/usr/bin/env python3
"""Big-sample baselines (mid, p_rw, LR stacker, HistGBDT) on KXBTC15M, chronological 60/15/25 split.
usage: python3 -I eval_big.py [CTX=512]   (CTX = context length saved for Timestral; baselines don't need >=121)
Outputs: data/samples_big.parquet, data/samples_big_ctx{CTX}.npy (float32 memmap, N x CTX), results_big.json.
Causal rules as eval.py: ctx = Coinbase 1m candles starting <= t-60; Kalshi bid/ask = candle closing at t=open+60k;
strike basis-corrected by a rolling median over PRIOR markets only."""
import json, math, sys, time, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE)); DATA = HERE / "data"
import eval as E
import features_gbdt as F
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression

CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 512
assert CTX >= 121
W = 301  # candle window the feature builder needs (max lookback 240 returns / 60 + 1)
T0 = time.time()
logit = F.logit


def load_btc():
    b = pd.concat([pd.read_parquet(DATA / "btc_1m_long.parquet"), pd.read_parquet(DATA / "btc_1m.parquet")])
    b = b.drop_duplicates("ts", keep="last").set_index("ts").sort_index()
    t0 = int(b.index[0]); g = b.reindex(t0 + 60 * np.arange(int((b.index[-1] - t0) // 60) + 1))
    return t0, {c: g[c].to_numpy(float) for c in ["close", "high", "low", "volume"]}


def load_candles():
    fs = [DATA / "kalshi_candles.parquet"] + sorted((DATA / "hist_candles").glob("c_*.parquet"))
    kc = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    for c in ["bid", "ask", "last", "vol"]: kc[c] = pd.to_numeric(kc[c], errors="coerce")
    return kc.drop_duplicates(["ticker", "ts"], keep="first").sort_values(["ticker", "ts"])


def build():
    t0, B = load_btc(); arr = B["close"]
    mk = pd.read_parquet(DATA / "all_markets.parquet").dropna(subset=["strike", "result"]).sort_values("open_ts").reset_index(drop=True)
    kc = load_candles()
    i0 = ((mk.open_ts - 60 - t0) // 60).to_numpy()
    px = np.where((i0 >= 0) & (i0 < len(arr)), arr[np.clip(i0, 0, len(arr) - 1)], np.nan)
    raw = pd.Series(mk.strike.to_numpy() - px)
    basis = raw.rolling(200, min_periods=20).median().shift(1)  # prior markets only
    assert np.isnan(basis.iloc[0])
    mk = mk.assign(strike=mk.strike - basis.fillna(0).to_numpy())
    s = pd.concat([mk.assign(k=k, ts=mk.open_ts + 60 * k) for k in E.OFFSETS], ignore_index=True)
    s = s.merge(kc[["ticker", "ts", "bid", "ask"]], on=["ticker", "ts"])
    s = s[(s.bid >= 0) & (s.bid < s.ask) & (s.ask <= 1)].copy()
    s["end"] = (s.ts - 60 - t0) // 60 + 1
    s = s[(s.end - CTX >= 0) & (s.end <= len(arr))]
    last30 = arr[s.end.to_numpy()[:, None] - 30 + np.arange(30)]
    s = s[np.isnan(last30).mean(1) <= 0.2].sort_values(["open_ts", "k"]).reset_index(drop=True)
    N = len(s); ctx = np.lib.format.open_memmap(DATA / f"samples_big_ctx{CTX}.npy", "w+", np.float32, (N, CTX))
    last, sig = np.empty(N), np.empty(N); end = s.end.to_numpy()
    for i in range(0, N, 5000):
        c = arr[end[i:i + 5000, None] - CTX + np.arange(CTX)]; ctx[i:i + 5000] = c
        r = np.diff(np.log(c[:, -121:]), axis=1); sig[i:i + 5000] = np.nanstd(r, axis=1)
        last[i:i + 5000] = [x[~np.isnan(x)][-1] for x in c[:, -30:]]
    ctx.flush()
    h = ((s.close_ts - s.ts) // 60).to_numpy()
    out = pd.DataFrame(dict(ticker=s.ticker, open_ts=s.open_ts, k=s.k, h=h.astype(int), strike=s.strike, y=s.result.astype(int),
                            bid=s.bid, ask=s.ask, mid=(s.bid + s.ask) / 2, last=last,
                            p_rw=E.ncdf(np.log(last / s.strike.to_numpy()) / (sig * np.sqrt(h) + 1e-12))))
    out.to_parquet(DATA / "samples_big.parquet")
    # causal self-checks
    ctx = np.load(DATA / f"samples_big_ctx{CTX}.npy", mmap_mode="r")
    for i in np.random.default_rng(0).choice(N, 30, replace=False):
        t = int(out.open_ts[i] + 60 * out.k[i]); e = (t - 60 - t0) // 60 + 1
        assert t0 + 60 * (e - 1) <= t - 60 < t0 + 60 * e          # newest ctx candle starts <= t-60 (closed by t)
        assert np.array_equal(ctx[i], arr[e - CTX:e].astype(np.float32), equal_nan=True)
    assert (out.open_ts.diff().dropna() >= 0).all()
    return out, mk, t0, B


def featurize(df, mk, t0, B, kd, check=True):
    a = (mk.open_ts.to_numpy(), mk.close_ts.to_numpy(), mk.result.to_numpy(), mk.ticker.to_numpy())
    def one(r, B_, kd_):
        t = r.open_ts + 60 * r.k; end = (t - 60 - t0) // 60 + 1
        Bw = {c: x[end - W:end] for c, x in B_.items()}                       # window ends exactly at the causal edge
        return F.rowfeat(r, *a, t - 60 - (W - 1) * 60, Bw, kd_)               # rowfeat only needs the last W candles
    if check:
        for r in df.sample(25, random_state=1).itertuples():
            t = r.open_ts + 60 * r.k; cut = (t - 60 - t0) // 60 + 1
            B2 = {c: np.where(np.arange(len(x)) < cut, x, np.nan) for c, x in B.items()}
            kd2 = {k: tuple(v[x[0] <= t] for v in x) for k, x in kd.items()}
            f1, f2 = one(r, B, kd), one(r, B2, kd2)
            for k in f1: assert (np.isnan(f1[k]) and np.isnan(f2[k])) or f1[k] == f2[k], (k, f1[k], f2[k])
        print("causal feature check ok", flush=True)
    return pd.DataFrame([one(r, B, kd) for r in df.itertuples()], index=df.index)


def pnl_vec(p, d, edge):
    """per-row pnl (NaN = no trade); same rule as eval.trade."""
    b, a, y = d.bid.to_numpy(), d.ask.to_numpy(), d.y.to_numpy()
    buy = (p - a - E.fee(a) > edge) & (a < 1); sell = ~buy & ((1 - p) - (1 - b) - E.fee(1 - b) > edge) & (b > 0)
    out = np.full(len(p), np.nan)
    out[buy] = (y - a - E.fee(a))[buy]; out[sell] = ((1 - y) - (1 - b) - E.fee(1 - b))[sell]
    return out


def tr(p, d, e):
    x = E.trade(p, d.bid.to_numpy(), d.ask.to_numpy(), d.y.to_numpy(), e)
    assert abs(x["pnl"] - np.nansum(pnl_vec(p, d, e))) < 1e-8
    return x


def boot_ci(pnl, d, rng, n=2000):
    """bootstrap over markets of mean per-trade pnl."""
    m = pd.factorize(d.ticker.to_numpy())[0]; tm = ~np.isnan(pnl)
    S = np.bincount(m[tm], pnl[tm], minlength=m.max() + 1); C = np.bincount(m[tm], minlength=m.max() + 1)
    if C.sum() == 0: return None
    ix = rng.integers(0, len(S), (n, len(S))); mean = S[ix].sum(1) / np.maximum(C[ix].sum(1), 1)
    return dict(trades=int(C.sum()), mean=float(S.sum() / C.sum()), lo=float(np.percentile(mean, 2.5)), hi=float(np.percentile(mean, 97.5)))


def calib(d):
    q = np.minimum((d.mid * 10).astype(int), 9); rows = []
    for bkt in range(10):
        x = d[q == bkt]
        if not len(x): continue
        rows.append(dict(bucket=f"{bkt/10:.1f}-{(bkt+1)/10:.1f}", n=len(x), mean_mid=float(x.mid.mean()), freq_yes=float(x.y.mean()),
                         yes_pnl=float((x.y - x.ask - E.fee(x.ask)).mean()), no_pnl=float(((1 - x.y) - (1 - x.bid) - E.fee(1 - x.bid)).mean())))
    return rows


def main():
    df, mk, t0, B = build()
    print(f"samples {len(df)} markets {df.ticker.nunique()} ctx={CTX}  build {time.time()-T0:.0f}s", flush=True)
    kc = load_candles(); kc["mid"] = (kc.bid + kc.ask) / 2
    kc = kc[kc.ticker.isin(set(df.ticker) | set(mk.ticker))]
    kd = {t: (g.ts.to_numpy(), g.mid.to_numpy(), g.vol.to_numpy()) for t, g in kc.groupby("ticker")}
    X = featurize(df, mk, t0, B, kd); print(f"features {time.time()-T0:.0f}s", flush=True)
    ot = np.sort(df.open_ts.unique()); c1, c2 = ot[int(.60 * len(ot))], ot[int(.75 * len(ot))]
    part = np.where(df.open_ts < c1, "train", np.where(df.open_ts < c2, "val", "holdout"))
    assert df.open_ts[part == "train"].max() < df.open_ts[part == "val"].min() and df.open_ts[part == "val"].max() < df.open_ts[part == "holdout"].min()
    res = {"ctx": CTX, "n_samples": len(df), "n_markets": int(df.ticker.nunique()), "n_configs_tried": 3 + 3 + 9, "split": {}}
    for nm in ["train", "val", "holdout"]:
        s = df[part == nm]; res["split"][nm] = dict(n=len(s), markets=int(s.ticker.nunique()),
            start=str(pd.to_datetime(s.open_ts.min(), unit="s")), end=str(pd.to_datetime(s.open_ts.max(), unit="s")))
        print(nm, res["split"][nm])
    rng = np.random.default_rng(0)
    for k in E.OFFSETS:
        g = df[df.k == k]; Xk = X.loc[g.index]; P = part[g.index]
        dt, dv, dl = g[P == "train"], g[P == "val"], g[P == "holdout"]
        m = HGB(loss="log_loss", learning_rate=0.03, max_iter=300, max_leaf_nodes=6, min_samples_leaf=40, l2_regularization=5.0,
                early_stopping=False, random_state=0).fit(Xk.loc[dt.index], dt.y)
        ll = [E.scores(p[:, 1], dv.y.to_numpy())["logloss"] for p in m.staged_predict_proba(Xk.loc[dv.index])]
        it = int(np.argmin(ll)) + 1
        stage = lambda d_: list(m.staged_predict_proba(Xk.loc[d_.index]))[it - 1][:, 1]
        st = lambda d_: np.c_[logit(d_.mid), logit(d_.p_rw)]
        lr = LogisticRegression(C=1.0).fit(st(dt), dt.y)
        preds = {"mid": (dv.mid.to_numpy(), dl.mid.to_numpy()), "p_rw": (dv.p_rw.to_numpy(), dl.p_rw.to_numpy()),
                 "stacker": (lr.predict_proba(st(dv))[:, 1], lr.predict_proba(st(dl))[:, 1]), "gbdt": (stage(dv), stage(dl))}
        r = dict(n_train=len(dt), n_val=len(dv), n_holdout=len(dl), gbdt_best_iter=it, stacker_coef=lr.coef_.tolist())
        for nme, (pv, pl) in preds.items():
            vt = {e: tr(pv, dv, e) for e in E.EDGES}; ok = [e for e in E.EDGES if vt[e]["trades"] >= 20]
            eb = max(ok, key=lambda e: vt[e]["per_trade"]) if ok else None
            r[nme] = dict(**E.scores(pl, dl.y.to_numpy()), val_logloss=E.scores(pv, dv.y.to_numpy())["logloss"], chosen_edge=eb,
                          val_trade={str(e): v for e, v in vt.items()}, trade={str(e): tr(pl, dl, e) for e in E.EDGES})
            if eb is not None: r[nme]["chosen_edge_ci95_per_trade"] = boot_ci(pnl_vec(pl, dl, eb), dl, rng)
        res[f"k{k}"] = r
        print(f"k={k} train/val/holdout={len(dt)}/{len(dv)}/{len(dl)} iter={it}")
        for nme in preds:
            print("  %-8s brier %.4f ll %.4f edge %s ci %s" % (nme, r[nme]["brier"], r[nme]["logloss"], r[nme]["chosen_edge"],
                  {a: round(b, 3) for a, b in (r[nme].get("chosen_edge_ci95_per_trade") or {}).items()}),
                  " ".join(f"e{e}:{v['trades']}t/{v['per_trade']:+.3f}" for e, v in r[nme]["trade"].items()))
    hold = df[part == "holdout"]; res["mid_calibration_holdout_all_k"] = calib(hold)
    for row in res["mid_calibration_holdout_all_k"]: print(row)
    res["runtime_s"] = round(time.time() - T0)
    (HERE / "results_big.json").write_text(json.dumps(res, indent=1)); print("runtime", res["runtime_s"], "s")


if __name__ == "__main__":
    main()
