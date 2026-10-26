"""Read-only recorder: Kalshi KXBTC15M orderbook (~1/s) + BTC spot trades (Coinbase/Kraken/Bitstamp).
Writes small parquet parts to live/ every 60s: <src>_<YYYYmmdd_HH>_<epoch>.parquet (hour = local-receive UTC hour).
No auth, no trading endpoints."""
import asyncio, json, ssl, sys, time, urllib.request, urllib.error
from datetime import datetime, timezone
from pathlib import Path
import certifi, pyarrow as pa, pyarrow.parquet as pq, websockets

D = Path(__file__).parent / "live"; D.mkdir(exist_ok=True)
CTX = ssl.create_default_context(cafile=certifi.where())
API = "https://api.elections.kalshi.com/trade-api/v2"
buf = {"kalshi": [], "spot": []}  # bounded: flushed every 60s, capped below
CAP = 200_000
cnt = {"kalshi": 0, "spot": 0, "coinbase": 0, "kraken": 0, "bitstamp": 0, "err": 0}

def put(k, row):
    if len(buf[k]) < CAP: buf[k].append(row)

def get(path):
    with urllib.request.urlopen(urllib.request.Request(API + path, headers={"User-Agent": "btc15m-recorder"}), timeout=8, context=CTX) as r:
        return json.load(r)

def top(levels, n=3):  # levels ascending by price -> best-first
    return [(float(p), float(s)) for p, s in reversed(levels[-n:])]

def kalshi_once(tk):
    ob = get(f"/markets/{tk}/orderbook")["orderbook_fp"]
    y, n = top(ob.get("yes_dollars") or []), top(ob.get("no_dollars") or [])
    return y, n

async def kalshi():
    mk, backoff, strike = None, 1.0, None
    loop = asyncio.get_running_loop()
    while True:
        t0 = time.time()
        try:
            if mk is None or t0 >= mk[1]:
                ms = (await loop.run_in_executor(None, get, "/markets?series_ticker=KXBTC15M&status=open&limit=5"))["markets"]
                ms = [m for m in ms if datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")).timestamp() > t0]
                if ms:
                    m = min(ms, key=lambda m: m["close_time"])
                    mk = (m["ticker"], datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")).timestamp(), m.get("floor_strike"))
                else: mk = None; await asyncio.sleep(2); continue
            y, n = await loop.run_in_executor(None, kalshi_once, mk[0])
            tr = time.time()
            by, bn = (y[0] if y else (None, None)), (n[0] if n else (None, None))
            put("kalshi", dict(t=tr, ticker=mk[0], close_ts=mk[1], strike=mk[2],
                yes_bid=by[0], yes_bid_sz=by[1], yes_ask=(1 - bn[0]) if bn[0] is not None else None, yes_ask_sz=bn[1],
                yes_top3=json.dumps(y), no_top3=json.dumps(n)))
            cnt["kalshi"] += 1; backoff = 1.0
            await asyncio.sleep(max(0.0, 1.0 - (time.time() - t0)))
        except urllib.error.HTTPError as e:
            cnt["err"] += 1; backoff = min(backoff * 2, 10) if e.code == 429 else 2.0
            await asyncio.sleep(backoff if e.code == 429 else 2)
        except Exception:
            cnt["err"] += 1; await asyncio.sleep(2)

def spot(venue, price, size=None):
    put("spot", dict(t=time.time(), venue=venue, price=float(price), size=float(size) if size else None))
    cnt["spot"] += 1; cnt[venue] += 1

async def ws_loop(venue, url, sub, handle):
    while True:
        try:
            async with websockets.connect(url, ssl=CTX, ping_interval=20, open_timeout=10) as w:
                await w.send(json.dumps(sub))
                async for msg in w: handle(json.loads(msg))
        except Exception:
            cnt["err"] += 1
        await asyncio.sleep(3)

def h_cb(m):
    if m.get("type") == "ticker" and m.get("price"): spot("coinbase", m["price"], m.get("last_size"))
def h_kr(m):
    if m.get("channel") == "trade" and m.get("type") in ("update", "snapshot"):
        for t in m["data"]: spot("kraken", t["price"], t.get("qty"))
def h_bs(m):
    if m.get("event") == "trade": spot("bitstamp", m["data"]["price"], m["data"].get("amount"))

async def flusher():
    while True:
        await asyncio.sleep(60); flush()
        print(datetime.now().strftime("%F %T"), {k: v for k, v in cnt.items()}, flush=True)

def flush():
    ts = int(time.time()); hr = datetime.now(timezone.utc).strftime("%Y%m%d_%H")
    for k in buf:
        rows, buf[k] = buf[k], []
        if rows: pq.write_table(pa.Table.from_pylist(rows), D / f"{k}_{hr}_{ts}.parquet", compression="zstd")

async def main():
    await asyncio.gather(kalshi(), flusher(),
        ws_loop("coinbase", "wss://ws-feed.exchange.coinbase.com", {"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["ticker"]}, h_cb),
        ws_loop("kraken", "wss://ws.kraken.com/v2", {"method": "subscribe", "params": {"channel": "trade", "symbol": ["BTC/USD"]}}, h_kr),
        ws_loop("bitstamp", "wss://ws.bitstamp.net", {"event": "bts:subscribe", "data": {"channel": "live_trades_btcusd"}}, h_bs))

if __name__ == "__main__":
    try: asyncio.run(main())
    finally: flush()
