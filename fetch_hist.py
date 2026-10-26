#!/usr/bin/env python3
"""Resumable archive of Kalshi KXBTC15M history (markets + trades) and Binance BTCUSDT 1s klines.
Usage: python3 fetch_hist.py [test|markets|trades|binance|all]   (default all). Rerun = resume."""
import io, json, ssl, sys, time, urllib.request, urllib.error, zipfile, hashlib, os
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
import certifi
import pandas as pd

K = "https://api.elections.kalshi.com/trade-api/v2"
B = "https://data.binance.vision/data/spot"
D = Path(__file__).parent / "data"
TR = D / "trades"; BN = D / "binance_1s"
for p in (D, TR, BN): p.mkdir(exist_ok=True)
CTX = ssl.create_default_context(cafile=certifi.where())
CHUNK = 500
_last = [0.0]; _gap = [0.125]  # ~8 req/s, adaptive


def raw(url, tries=6):
    for i in range(tries):
        wait = _gap[0] - (time.time() - _last[0])
        if wait > 0: time.sleep(wait)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
            with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
                _gap[0] = max(0.125, _gap[0] * 0.98)
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404: raise
            if e.code in (429, 500, 502, 503, 504):
                if e.code == 429: _gap[0] = min(2.0, _gap[0] * 2)
                time.sleep(1.5 * (i + 1)); continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"failed {url}")


def get(url): return json.loads(raw(url))
def iso(s): return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
def log(*a): print(datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def mrow(m):
    ev = m.get("expiration_value")
    return dict(ticker=m["ticker"], event=m["event_ticker"], open_ts=int(iso(m["open_time"])), close_ts=int(iso(m["close_time"])),
                strike=m.get("floor_strike"), exp_value=float(ev.replace(",", "")) if ev else None,
                result=int(m["result"] == "yes") if m.get("result") in ("yes", "no") else None,
                volume=float(m.get("volume_fp") or 0))


def do_markets(limit_pages=None):
    cut = get(f"{K}/historical/cutoff"); (D / "hist_cutoff.json").write_text(json.dumps(cut)); log("cutoff", cut)
    path = D / "hist_markets.parquet"
    if not path.exists():
        rows, cur, n = [], "", 0
        while True:
            d = get(f"{K}/historical/markets?series_ticker=KXBTC15M&limit=1000" + (f"&cursor={cur}" if cur else ""))
            rows += [mrow(m) for m in d["markets"] if m.get("result") in ("yes", "no")]
            cur = d.get("cursor"); n += 1; log("hist markets", len(rows))
            if not cur or not d["markets"] or (limit_pages and n >= limit_pages): break
        pd.DataFrame(rows).drop_duplicates("ticker").sort_values("open_ts").to_parquet(path)
    h = pd.read_parquet(path)
    r = pd.read_parquet(D / "markets.parquet")
    a = pd.concat([h, r]).drop_duplicates("ticker").sort_values("open_ts").reset_index(drop=True)
    a.to_parquet(D / "all_markets.parquet"); log("all_markets", len(a), "hist", len(h), "recent", len(r))
    return a


def trade_rows(tk, hist):
    ep = "historical/trades" if hist else "markets/trades"
    rows, cur = [], ""
    while True:
        d = get(f"{K}/{ep}?limit=1000&ticker={tk}" + (f"&cursor={cur}" if cur else ""))
        for t in d["trades"]:
            rows.append(dict(trade_id=t["trade_id"], ticker=t["ticker"], created_time=iso(t["created_time"]),
                             yes_price=float(t["yes_price_dollars"]), no_price=float(t["no_price_dollars"]),
                             count=float(t["count_fp"]), taker_side=t["taker_side"]))
        cur = d.get("cursor")
        if not cur or not d["trades"]: return rows


def do_trades(mk, only=None):
    cut = iso(json.loads((D / "hist_cutoff.json").read_text())["trades_created_ts"])
    donef = TR / "done.txt"
    done = set(donef.read_text().split()) if donef.exists() else set()
    todo = mk[~mk.ticker.isin(done)].sort_values("open_ts")
    if only: todo = todo.head(only)
    n0 = len(done); t0 = time.time(); buf, bt = [], []
    nchunk = len(list(TR.glob("chunk_*.parquet")))

    def flush():
        nonlocal buf, bt, nchunk
        if bt:
            pd.DataFrame(buf, columns=["trade_id","ticker","created_time","yes_price","no_price","count","taker_side"]).to_parquet(TR / f"chunk_{nchunk:04d}.parquet")
            nchunk += 1
        with donef.open("a") as f: f.write("\n".join(bt) + "\n")
        buf, bt = [], []

    for i, m in enumerate(todo.itertuples(), 1):
        buf += trade_rows(m.ticker, m.close_ts < cut); bt.append(m.ticker)
        if len(bt) >= CHUNK: flush()
        if i % 200 == 0:
            rate = i / (time.time() - t0)
            log(f"trades {n0 + i}/{len(mk)} rate {rate:.2f} mkt/s eta {(len(todo) - i) / rate / 60:.0f} min")
    flush(); log("trades done", len(todo), "new markets")


def to_s(t):  # ms vs us epoch
    t = t.astype("int64")
    return (t // 1_000_000).where(t > 1e14, t // 1000)


def parse_zip(url, tmp):
    body = raw(url)
    try:
        exp = raw(url + ".CHECKSUM").split()[0].decode()
        if hashlib.sha256(body).hexdigest() != exp: raise RuntimeError("checksum mismatch " + url)
    except urllib.error.HTTPError: log("no checksum", url)
    tmp.write_bytes(body)
    with zipfile.ZipFile(tmp) as z:
        df = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=[0,1,2,3,4,5,8],
                         names=["open_time","open","high","low","close","volume","trades"])
    tmp.unlink()
    df["open_time"] = to_s(df["open_time"])
    return df


def do_binance(start=date(2025, 12, 9), end=None, test=False):
    tmp = BN / "tmp.zip"
    today = datetime.now(timezone.utc).date()
    first_of_month = lambda d: d.replace(day=1)
    months, d = [], first_of_month(start)
    while d <= today:
        months.append(d); d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    if test: months = months[:1]
    for mo in months:
        out = BN / f"{mo:%Y-%m}.parquet"
        nxt = (mo.replace(day=28) + timedelta(days=4)).replace(day=1)
        full = nxt <= today
        if out.exists() and full and not test: continue
        if test:  # one daily file only
            df = parse_zip(f"{B}/daily/klines/BTCUSDT/1s/BTCUSDT-1s-{start:%Y-%m-%d}.zip", tmp)
            log("test day rows", len(df), df.open_time.iloc[0], df.open_time.iloc[-1]); return df
        try:
            df = parse_zip(f"{B}/monthly/klines/BTCUSDT/1s/BTCUSDT-1s-{mo:%Y-%m}.zip", tmp)
        except urllib.error.HTTPError:  # no monthly yet -> dailies
            parts, d = [], max(mo, start) if mo == first_of_month(start) else mo
            old = pd.read_parquet(out) if out.exists() else None
            if old is not None: d = datetime.fromtimestamp(old.open_time.max(), timezone.utc).date() + timedelta(days=1)
            parts = [old] if old is not None else []
            while d < today and d < nxt:
                try: parts.append(parse_zip(f"{B}/daily/klines/BTCUSDT/1s/BTCUSDT-1s-{d:%Y-%m-%d}.zip", tmp))
                except urllib.error.HTTPError: break
                d += timedelta(days=1)
            if not parts: continue
            df = pd.concat(parts)
        if mo == first_of_month(start): df = df[df.open_time >= int(datetime.combine(start, datetime.min.time(), timezone.utc).timestamp())]
        df.drop_duplicates("open_time").to_parquet(out); log("binance", out.name, len(df))
    tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "test":
        # separate test dir state: only 2 markets/1 day, outputs to scratch dir, no done-list pollution
        import tempfile; D = Path(tempfile.mkdtemp()); TR = D / "trades"; BN = D / "binance_1s"; TR.mkdir(); BN.mkdir()
        import shutil; shutil.copy(Path(__file__).parent / "data/markets.parquet", D / "markets.parquet")
        a = do_markets(limit_pages=1)
        pick = pd.concat([a[a.close_ts < iso(json.loads((D / "hist_cutoff.json").read_text())["trades_created_ts"])].tail(1), a.tail(1)])
        do_trades(pick); t = pd.concat(pd.read_parquet(f) for f in TR.glob("chunk_*.parquet")); print(t.groupby("ticker").size(), t.head(3))
        do_binance(test=True); shutil.rmtree(D)
    else:
        if mode in ("all", "markets", "trades"): a = do_markets()
        if mode in ("all", "trades"): do_trades(a)
        if mode in ("all", "binance"): do_binance()
        log("ALL DONE")
