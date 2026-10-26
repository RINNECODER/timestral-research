#!/usr/bin/env python3
"""1-min YES bid/ask candles for historical (pre-cutoff) KXBTC15M markets -> data/hist_candles/*.parquet (resumable)."""
import sys, time
import pandas as pd
from fetch import get, K, OUT

OUTD = OUT / "hist_candles"
OUTD.mkdir(exist_ok=True)


def num(c, k, sub):
    v = (c.get(k) or {})
    v = v.get(sub, v.get(sub + "_dollars"))  # historical uses unsuffixed keys, live uses *_dollars
    return float(v) if v not in (None, "") else None


if __name__ == "__main__":
    mk = pd.read_parquet(OUT / "hist_markets.parquet")
    recent = set(pd.read_parquet(OUT / "kalshi_candles.parquet").ticker)
    done = set()
    for f in OUTD.glob("*.parquet"):
        done |= set(pd.read_parquet(f, columns=["ticker"]).ticker)
    done |= set((OUTD / "empty.txt").read_text().split()) if (OUTD / "empty.txt").exists() else set()
    todo = mk[~mk.ticker.isin(done | recent)].sort_values("open_ts")
    print(f"todo {len(todo)}", file=sys.stderr, flush=True)
    rows, empty, t0 = [], [], time.time()
    for i, m in enumerate(todo.itertuples(), 1):
        d = get(f"{K}/historical/markets/{m.ticker}/candlesticks?period_interval=1"
                f"&start_ts={int(m.open_ts) - 60}&end_ts={int(m.close_ts)}")
        cs = d.get("candlesticks", [])
        if not cs:
            empty.append(m.ticker)
        for c in cs:
            rows.append(dict(ticker=m.ticker, ts=c["end_period_ts"], bid=num(c, "yes_bid", "close"),
                             ask=num(c, "yes_ask", "close"), last=num(c, "price", "close"),
                             vol=float(c.get("volume") or c.get("volume_fp") or 0)))
        if i % 1000 == 0 or i == len(todo):
            pd.DataFrame(rows).to_parquet(OUTD / f"c_{int(time.time())}.parquet"); rows = []
            with (OUTD / "empty.txt").open("a") as f: f.write("\n".join(empty) + "\n")
            empty = []
            print(f"{time.strftime('%H:%M:%S')} candles {i}/{len(todo)} {i / (time.time() - t0):.1f}/s", file=sys.stderr, flush=True)
        time.sleep(0.12)
