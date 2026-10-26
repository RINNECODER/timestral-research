#!/usr/bin/env python3
"""Forward paper test of the FROZEN LR stacker [logit(mid), logit(p_rw)]. NO orders/auth: only reads live/*.parquet and
public GET endpoints (Coinbase candles, Kalshi /markets/{ticker}).
  python3 -I paper_trader.py --freeze   fit once on eval_big's train split (refuses if paper_model.json exists)
  python3 -I paper_trader.py            run loop (use nohup; see stop_paper.sh)   |  --selftest: unit checks"""
import json, math, ssl, sys, time, glob, urllib.request, urllib.error
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent; LIVE = HERE / "live"; MODEL = HERE / "paper_model.json"
CSV = HERE / "paper_trades.csv"; SUMM = HERE / "paper_summary.json"
KS = [3, 4, 5, 6, 7]; EDGE = 0.02; LAT = 60
COLS = ["time", "ticker", "k", "mid", "p_rw", "p", "side", "bid", "ask", "fill_now", "fill_60s", "result", "pnl_now", "pnl_60s"]
fee = lambda p: 0.07 * p * (1 - p)
logit = lambda p: np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))
ncdf = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))


def freeze():
    assert not MODEL.exists(), "model already frozen; never refit"
    sys.path.insert(0, str(HERE))
    sys.argv = sys.argv[:1]; import eval_big as eb, eval as E
    from sklearn.linear_model import LogisticRegression
    E.OFFSETS = KS; eb.CTX = 121
    pd.DataFrame.to_parquet = lambda *a, **k: None   # don't clobber samples_big.parquet
    df, mk, t0, B = eb.build()
    ot = np.sort(df.open_ts.unique()); c1 = ot[int(.60 * len(ot))]; tr = df[df.open_ts < c1]
    raw = pd.read_parquet(HERE / "data/all_markets.parquet").dropna(subset=["strike", "result"]).sort_values("open_ts")
    i0 = ((raw.open_ts - 60 - t0) // 60).to_numpy(); arr = B["close"]
    px = np.where((i0 >= 0) & (i0 < len(arr)), arr[np.clip(i0, 0, len(arr) - 1)], np.nan)
    basis = float(pd.Series(raw.strike.to_numpy() - px).rolling(200, min_periods=20).median().iloc[-1])
    out = {"frozen_at": pd.Timestamp.now("UTC").isoformat(), "train_end_open_ts": int(c1), "edge": EDGE, "basis": basis,
           "basis_note": "constant = last rolling-200 median of (kalshi strike - coinbase 1m close at open-60) over all archived markets; live strike_corr = strike - basis",
           "features": "logit(mid), logit(p_rw), clip 1e-3", "models": {}}
    for k in KS:
        d = tr[tr.k == k]; lr = LogisticRegression(C=1.0).fit(np.c_[logit(d.mid), logit(d.p_rw)], d.y)
        out["models"][str(k)] = dict(coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]), n_train=len(d))
    MODEL.write_text(json.dumps(out, indent=1)); (HERE / "data/samples_big_ctx121.npy").unlink(missing_ok=True)
    print(json.dumps(out, indent=1))


def decide(model, k, bid, ask, mid, p_rw):
    """-> (p, side) with eval.trade's rule; side in buy/sell/none"""
    m = model["models"][str(k)]; z = m["intercept"] + m["coef"][0] * logit(mid) + m["coef"][1] * logit(p_rw)
    p = 1 / (1 + math.exp(-z))
    if p - ask - fee(ask) > EDGE and ask < 1: return p, "buy"
    if (1 - p) - (1 - bid) - fee(1 - bid) > EDGE and bid > 0: return p, "sell"
    return p, "none"


def pnl(side, fill, result):  # result: 1 yes / 0 no; sell = buy NO at 1-bid
    if side == "buy": return result - fill - fee(fill)
    if side == "sell": return (1 - result) - (1 - fill) - fee(1 - fill)
    return np.nan


def p_rw_from(closes, strike_corr, h):
    """closes: 121+ minute closes, oldest first, NaN for gaps (as eval_big: last 121 -> 120 returns, nanstd; last = last non-nan of last 30)"""
    c = np.asarray(closes[-121:], float)
    if len(c) < 121 or np.isnan(c[-30:]).mean() > 0.2: return None
    sig = np.nanstd(np.diff(np.log(c)))
    if not np.isfinite(sig): return None
    last = c[-30:][~np.isnan(c[-30:])][-1]
    return ncdf(math.log(last / strike_corr) / (sig * math.sqrt(h) + 1e-12))


SSL = None
def get(url):
    global SSL
    import certifi
    SSL = SSL or ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
    with urllib.request.urlopen(req, timeout=20, context=SSL) as r: return json.load(r)


class Live:
    def __init__(self): self.closes = {}; self.rest_at = 0   # minute-start -> close
    def newest(self, kind, n):
        out = []
        for f in sorted(glob.glob(str(LIVE / f"{kind}_*.parquet")))[-n:]:
            try: out.append(pd.read_parquet(f))
            except Exception: pass   # file mid-write
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    def kalshi(self): return self.newest("kalshi", 4)
    def update_spot(self):
        s = self.newest("spot", 5)
        if len(s):
            s = s[s.venue == "coinbase"].sort_values("t")
            for m, v in s.groupby((s.t // 60).astype(int) * 60).price.last().items(): self.closes[int(m)] = float(v)
        for m in [m for m in self.closes if m < time.time() - 200 * 60]: del self.closes[m]
    def rest(self, now):
        if now - self.rest_at < 120: return
        try:
            d = get(f"https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=60&start={pd.Timestamp(now - 140 * 60, unit='s').isoformat()}&end={pd.Timestamp(now, unit='s').isoformat()}")
            live = dict(self.closes)
            self.closes = {int(r[0]): float(r[4]) for r in d}; self.closes.update(live)  # live trades win over REST
            self.rest_at = now
        except Exception as e: print("rest fail", e, flush=True)
    def series(self, t):  # 121 closes of minutes starting t-121*60 .. t-60 (closed by t)
        return [self.closes.get(m, np.nan) for m in range(t - 121 * 60, t, 60)]


def load_rows():
    return pd.read_csv(CSV) if CSV.exists() else pd.DataFrame(columns=COLS)


def summarize(rows):
    out = {"updated": pd.Timestamp.now("UTC").isoformat(), "n_rows": len(rows), "n_markets_settled": int(rows[rows.result.notna()].ticker.nunique())}
    rng = np.random.default_rng(0)
    for v in ("now", "60s"):
        r = rows[rows.side.isin(["buy", "sell"]) & rows[f"pnl_{v}"].notna()]
        if not len(r): out[v] = {"n": 0}; continue
        g = r.groupby("ticker")[f"pnl_{v}"].agg(["sum", "count"]); S, C = g["sum"].to_numpy(), g["count"].to_numpy()
        ix = rng.integers(0, len(S), (2000, len(S))); bs = S[ix].sum(1) / C[ix].sum(1)
        out[v] = dict(n=int(C.sum()), markets=len(S), per_trade=float(S.sum() / C.sum()), total=float(S.sum()),
                      ci95_by_market=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        for k, gk in r.groupby("k"): out[v][f"k{k}"] = dict(n=len(gk), per_trade=float(gk[f"pnl_{v}"].mean()))
    SUMM.write_text(json.dumps(out, indent=1)); return out


def run():
    model = json.loads(MODEL.read_text()); basis = model["basis"]; L = Live()
    t_frz = pd.Timestamp(model['frozen_at']).timestamp(); rows = load_rows(); done = set(zip(rows.ticker, rows.k)); logged = set(rows[rows.result.notna()].ticker)
    print(f"start frozen_at={model['frozen_at']} basis={basis:.2f} rows={len(rows)}", flush=True)
    while True:
        try:
            now = time.time(); ks = L.kalshi(); L.update_spot()
            if len(ks):
                ks = ks.dropna(subset=["yes_bid", "yes_ask", "strike"]).sort_values("t")
                for tk, g in ks.groupby("ticker"):
                    close = float(g.close_ts.iloc[-1]); op = close - 900; strike = float(g.strike.iloc[-1]) - basis
                    for k in KS:
                        t = op + 60 * k
                        if (tk, k) in done or g.t.iloc[-1] < t + 2 or t < t_frz: continue   # recorder files lag ~60s: act on data time, not wall time
                        _close[tk] = close; done.add((tk, k))   # once per k per market, even if skipped
                        q = g[g.t <= t]
                        if not len(q) or t - q.t.iloc[-1] > 10: continue
                        L.rest(now); b, a = float(q.yes_bid.iloc[-1]), float(q.yes_ask.iloc[-1])
                        if not (0 <= b < a <= 1): continue
                        prw = p_rw_from(L.series(int(t)), strike, (close - t) / 60)
                        if prw is None: print("skip: not enough spot history", tk, k, flush=True); continue
                        mid = (a + b) / 2; p, side = decide(model, k, b, a, mid, prw)
                        fill = a if side == "buy" else b if side == "sell" else np.nan
                        rows.loc[len(rows)] = [pd.Timestamp(t, unit="s", tz="UTC").isoformat(), tk, k, mid, prw, p, side, b, a, fill, np.nan, np.nan, np.nan, np.nan]
                        print(f"signal {tk} k={k} mid={mid:.3f} p_rw={prw:.3f} p={p:.3f} -> {side} @ {fill}", flush=True) if side != "none" else None
                        rows.to_csv(CSV, index=False)
                # 60s-later fill: latest quote at or after (entry time + 60s), within 15s
                for i in rows.index[rows.side.isin(["buy", "sell"]) & rows.fill_60s.isna()]:
                    te = pd.Timestamp(rows.time[i]).timestamp() + LAT
                    if ks.t.max() < te + 15: continue
                    q = ks[(ks.ticker == rows.ticker[i]) & (ks.t >= te) & (ks.t <= te + 15)]
                    if len(q): rows.loc[i, "fill_60s"] = q.yes_ask.iloc[0] if rows.side[i] == "buy" else q.yes_bid.iloc[0]
            # settle
            for tk in rows[rows.result.isna()].ticker.unique():
                close = op_close(tk)
                if close is None or now < close + 30: continue
                try: m = get(f"https://api.elections.kalshi.com/trade-api/v2/markets/{tk}")["market"]
                except Exception: continue
                if m.get("result") not in ("yes", "no"): continue
                res = int(m["result"] == "yes"); ix = rows.ticker == tk; rows.loc[ix, "result"] = res
                for i in rows.index[ix]:
                    s = rows.side[i]
                    rows.loc[i, "pnl_now"] = pnl(s, rows.fill_now[i], res)
                    rows.loc[i, "pnl_60s"] = pnl(s, rows.fill_60s[i], res) if pd.notna(rows.fill_60s[i]) else np.nan
                rows.to_csv(CSV, index=False); S = summarize(rows)
                sub = rows[ix & rows.side.isin(["buy", "sell"])]
                print(f"settled {tk} res={res} trades={len(sub)} pnl_now={sub.pnl_now.sum():+.3f} pnl_60s={sub.pnl_60s.sum():+.3f} | cum now n={S['now'].get('n')} {S['now'].get('per_trade', 0):+.4f}", flush=True)
            if len(rows): rows.to_csv(CSV, index=False)
        except Exception as e: print("loop error", repr(e), flush=True)
        time.sleep(5)


_close = {}
def op_close(tk):  # close time, cached; REST fallback after restart
    if tk not in _close:
        try: _close[tk] = pd.Timestamp(get(f"https://api.elections.kalshi.com/trade-api/v2/markets/{tk}")["market"]["close_time"]).timestamp()
        except Exception: return None
    return _close[tk]


def selftest():
    mdl = {"models": {"5": dict(coef=[1.0, 0.0], intercept=0.0)}}
    assert decide(mdl, 5, .40, .42, .41, .5)[1] == "none"
    mdl["models"]["5"]["coef"] = [0.0, 1.0]
    assert decide(mdl, 5, .40, .42, .41, .9)[1] == "buy" and decide(mdl, 5, .58, .60, .59, .1)[1] == "sell"
    assert abs(pnl("buy", .4, 1) - (.6 - fee(.4))) < 1e-12 and abs(pnl("sell", .6, 0) - (.6 - fee(.4))) < 1e-12
    c = list(100 + np.cumsum(np.random.default_rng(0).normal(0, .01, 121)))
    assert p_rw_from(c, c[-1] - 1, 10) > .99 and p_rw_from(c[:50], 100, 5) is None
    print("selftest ok")


if __name__ == "__main__":
    {"--freeze": freeze, "--selftest": selftest}.get(sys.argv[1] if len(sys.argv) > 1 else "", run)()
