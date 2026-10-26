#!/usr/bin/env python3
"""Zero-shot Timestral vs Kalshi KXBTC15M prices vs a random-walk baseline.

For each settled market and entry offset k minutes after open, forecast BTC at close from 1-min
Coinbase closes, turn the quantiles into P(close >= strike), and score against the market mid.
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
MODELS = ["mistralai/Timestral-28M", "mistralai/Timestral-209M"]
OFFSETS = [1, 5, 10]
CTX = 2048
EDGES = [0.02, 0.05, 0.10]
LEVELS = np.array([0.01, 0.05] + [round(0.1 + 0.05 * i, 2) for i in range(17)] + [0.95, 0.99])


def prob_above(q, strike):
    """P(X >= strike) from sorted quantile values q (..., 21). Linear CDF between knots."""
    # shortcut: tails beyond the 1%/99% knots are clipped to 0.5%/99.5%; upgrade to a fitted tail if
    # many entries land there (the 10-minute offset often will)
    q = np.maximum.accumulate(q, axis=-1)
    cdf = np.array([np.interp(s, qi, LEVELS, left=0.005, right=0.995) for qi, s in zip(q, strike)])
    return 1.0 - cdf


def ncdf(x):
    return 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


def fee(p):
    return 0.07 * p * (1 - p)  # Kalshi taker fee per contract


def scores(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return dict(brier=float(np.mean((p - y) ** 2)), logloss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))))


def trade(p, bid, ask, y, edge):
    pnl = []
    for pi, b, a, yi in zip(p, bid, ask, y):
        if pi - a - fee(a) > edge and a < 1:
            pnl.append(yi - a - fee(a))
        elif (1 - pi) - (1 - b) - fee(1 - b) > edge and b > 0:
            pnl.append((1 - yi) - (1 - b) - fee(1 - b))
    return dict(trades=len(pnl), pnl=float(np.sum(pnl)), per_trade=float(np.mean(pnl)) if pnl else 0.0)


def build():
    mk = pd.read_parquet(DATA / "markets.parquet").dropna(subset=["strike"])
    btc = pd.read_parquet(DATA / "btc_1m.parquet").set_index("ts")["close"]
    kc = pd.read_parquet(DATA / "kalshi_candles.parquet")
    kc = kc.assign(bid=pd.to_numeric(kc.bid), ask=pd.to_numeric(kc.ask)).set_index(["ticker", "ts"])
    t0 = int(btc.index[0]); arr = btc.to_numpy(dtype=np.float64)
    # strike is BRTI; context is Coinbase. Shift strike by the BRTI-Coinbase basis seen in past markets only.
    basis = (mk.strike - btc.reindex(mk.open_ts - 60).to_numpy()).rolling(200, min_periods=20).median().shift(1)
    mk = mk.assign(strike=mk.strike - basis.fillna(0))
    rows, ctxs = [], []
    for m in mk.itertuples():
        for k in OFFSETS:
            t = m.open_ts + 60 * k
            key = (m.ticker, t)
            if key not in kc.index:
                continue
            bid, ask = kc.loc[key, ["bid", "ask"]]
            if not (0 <= bid < ask <= 1):
                continue
            end = (t - 60 - t0) // 60 + 1  # candles starting <= t-60 are closed by t
            if end - CTX < 0 or end > len(arr):
                continue
            ctx = arr[end - CTX:end]
            if np.isnan(ctx[-30:]).mean() > 0.2:
                continue
            last = ctx[~np.isnan(ctx)][-1]
            r = np.diff(np.log(ctx[-121:])); sig = np.nanstd(r)
            h = (m.close_ts - t) // 60
            rows.append(dict(ticker=m.ticker, open_ts=m.open_ts, k=k, h=int(h), strike=m.strike, y=m.result,
                             bid=bid, ask=ask, mid=(bid + ask) / 2, last=last,
                             p_rw=float(ncdf(np.log(last / m.strike) / (sig * math.sqrt(h) + 1e-12)))))
            ctxs.append(ctx)
    return pd.DataFrame(rows), np.stack(ctxs)


def run_model(name, ctxs, hs, strikes, bs=256):
    import torch
    from timestral import TimestralForecaster
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    f = TimestralForecaster.from_pretrained(name, device=dev)
    H = int(hs.max()); out = np.empty(len(ctxs)); t = time.time()
    for i in range(0, len(ctxs), bs):
        q = f.forecast(ctxs[i:i + bs, None, :], prediction_length=H).quantiles[:, 0].float().cpu().numpy()
        q = q[np.arange(len(q)), hs[i:i + bs] - 1]  # step ending at market close
        out[i:i + bs] = prob_above(q, strikes[i:i + bs])
    print(f"{name} on {dev}: {len(ctxs)} forecasts in {time.time() - t:.1f}s", flush=True)
    return out


def selftest():
    z = np.array([-2.326, -1.645, -1.2816, -1.0364, -0.8416, -0.6745, -0.5244, -0.3853, -0.2533, -0.1257, 0,
                  0.1257, 0.2533, 0.3853, 0.5244, 0.6745, 0.8416, 1.0364, 1.2816, 1.645, 2.326])
    p = prob_above(np.stack([z, z]), np.array([0.0, 1.0]))
    assert abs(p[0] - 0.5) < 1e-6 and abs(p[1] - 0.1587) < 0.01, p
    assert abs(ncdf(np.array([1.0]))[0] - 0.8413) < 1e-3
    assert trade([0.9], [0.4], [0.5], [1], 0.02)["pnl"] > 0 and trade([0.1], [0.5], [0.6], [1], 0.02)["pnl"] < 0
    print("selftest ok")


if __name__ == "__main__":
    selftest()
    df, ctxs = build()
    print(f"{len(df)} samples, {df.ticker.nunique()} markets", flush=True)
    for name in MODELS:
        df["p_" + name.split("-")[-1]] = run_model(name, ctxs, df.h.to_numpy(), df.strike.to_numpy())
    df.to_parquet(OUT / "preds.parquet")
    cols = ["mid", "p_rw"] + [c for c in df if c.startswith("p_") and c != "p_rw"]
    cut = df.open_ts.quantile(0.5)  # report later half separately: anything tuned later uses the first half
    res = {}
    for k in OFFSETS:
        for split, d in [("all", df[df.k == k]), ("late_half", df[(df.k == k) & (df.open_ts >= cut)])]:
            r = {c: scores(d[c].to_numpy(), d.y.to_numpy()) for c in cols}
            for c in cols[1:]:
                r[c]["trade"] = {str(e): trade(d[c], d.bid, d.ask, d.y, e) for e in EDGES}
            res[f"k{k}_{split}"] = dict(n=len(d), **r)
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    for key, r in res.items():
        print(f"\n{key} n={r['n']}")
        for c in cols:
            tr = " ".join(f"e{e}:{v['trades']}t/{v['per_trade']:+.3f}" for e, v in r[c].get("trade", {}).items())
            print(f"  {c:10s} brier {r[c]['brier']:.4f} logloss {r[c]['logloss']:.4f}  {tr}")
