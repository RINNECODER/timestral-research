#!/usr/bin/env python3
"""Coinbase BTC-USD 1m candles for the 365 days before the late_half cut (median open_ts of markets.parquet)."""
from pathlib import Path
import pandas as pd
from fetch import btc  # same Coinbase pager + certifi SSL context

DATA = Path(__file__).parent / "data"

if __name__ == "__main__":
    cut = int(pd.read_parquet(DATA / "markets.parquet").open_ts.median())
    b = btc(cut - 365 * 86400, cut)
    b = b[b.ts < cut]
    assert b.ts.max() < cut
    b.to_parquet(DATA / "btc_1m_long.parquet")
    print(f"cut {cut} rows {len(b)} nan {b.close.isna().sum()} {b.ts.min()}..{b.ts.max()}")
