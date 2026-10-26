#!/usr/bin/env python3
"""Pull settled Kalshi KXBTC15M markets, their 1-min YES price candles, and Coinbase BTC-USD 1-min candles."""
import json, ssl, sys, time, urllib.request, urllib.error
import certifi
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

K = "https://api.elections.kalshi.com/trade-api/v2"
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)
CTX = ssl.create_default_context(cafile=certifi.where())


def get(url, tries=6):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
            with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (i + 1)); continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"failed {url}")


def iso(s):
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def markets():
    rows, cur = [], ""
    while True:
        d = get(f"{K}/markets?series_ticker=KXBTC15M&status=settled&limit=1000" + (f"&cursor={cur}" if cur else ""))
        for m in d["markets"]:
            if m.get("result") not in ("yes", "no"):
                continue
            rows.append(dict(ticker=m["ticker"], event=m["event_ticker"], open_ts=iso(m["open_time"]),
                             close_ts=iso(m["close_time"]), strike=m.get("floor_strike"),
                             exp_value=float(m["expiration_value"].replace(",", "")) if m.get("expiration_value") else None,
                             result=int(m["result"] == "yes"), volume=float(m.get("volume_fp") or 0)))
        cur = d.get("cursor")
        print(f"markets {len(rows)}", file=sys.stderr)
        if not cur or not d["markets"]:
            return pd.DataFrame(rows).sort_values("open_ts").reset_index(drop=True)


def market_candles(mk):
    path = OUT / "kalshi_candles.parquet"
    done = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    have = set(done["ticker"]) if len(done) else set()
    rows = []
    todo = mk[~mk.ticker.isin(have)]
    for i, m in enumerate(todo.itertuples()):
        d = get(f"{K}/series/KXBTC15M/markets/{m.ticker}/candlesticks?start_ts={m.open_ts - 60}"
                f"&end_ts={m.close_ts}&period_interval=1")
        for c in d.get("candlesticks", []):
            f = lambda k, sub: (c.get(k) or {}).get(sub)
            rows.append(dict(ticker=m.ticker, ts=c["end_period_ts"],
                             bid=f("yes_bid", "close_dollars"), ask=f("yes_ask", "close_dollars"),
                             last=f("price", "close_dollars"), vol=c.get("volume_fp") or c.get("volume")))
        if i % 200 == 199 or i == len(todo) - 1:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True); rows = []
            done.to_parquet(path); print(f"candles {i + 1}/{len(todo)}", file=sys.stderr)
        time.sleep(0.06)
    return done


def btc(start, end):
    # Coinbase: 300 candles/request, rows = [time, low, high, open, close, volume]
    rows, t = [], start
    while t < end:
        e = min(t + 300 * 60, end)
        s = lambda x: datetime.fromtimestamp(x, timezone.utc).isoformat()
        d = get(f"https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=60&start={s(t)}&end={s(e)}")
        rows += d; t = e; time.sleep(0.12)
    df = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"]).drop_duplicates("ts")
    df = df.sort_values("ts").set_index("ts")
    # reindex to a full minute grid; gaps stay NaN (Timestral takes NaN as missing)
    return df.reindex(range(int(df.index.min()), int(df.index.max()) + 60, 60)).rename_axis("ts").reset_index()


if __name__ == "__main__":
    mk = markets(); mk.to_parquet(OUT / "markets.parquet")
    print(f"{len(mk)} markets {datetime.fromtimestamp(mk.open_ts.min(), timezone.utc)} .. "
          f"{datetime.fromtimestamp(mk.close_ts.max(), timezone.utc)}", file=sys.stderr)
    b = btc(int(mk.open_ts.min()) - 4 * 86400, int(mk.close_ts.max()) + 60); b.to_parquet(OUT / "btc_1m.parquet")
    print(f"btc rows {len(b)} nan {b.close.isna().sum()}", file=sys.stderr)
    c = market_candles(mk)
    print(f"candle rows {len(c)} markets {c.ticker.nunique()}", file=sys.stderr)
