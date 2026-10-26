# Data sources for Kalshi KXBTC15M backtest (researched 2026-10-26)
V = verified by hitting the URL/API myself; U = unverified (search snippet / page not opened).

## 1. Kalshi's own /historical API  -- BEST, FREE, NO AUTH (V)
- Docs: https://docs.kalshi.com/getting_started/historical_data
- GET https://api.elections.kalshi.com/trade-api/v2/historical/cutoff -> today: market_settled_ts = trades_created_ts = 2026-08-26T00:00Z (live API only keeps ~2 months; cutoff moves forward, so ARCHIVE OLD DATA NOW).
- /historical/markets?series_ticker=KXBTC15M&limit=1000 (cursor paging, newest first): 24,051 markets, 25 pages, back to KXBTC15M-25DEC101700-00 (opened 2025-12-10). Fields incl. floor_strike, expiration_value, result.
- /historical/trades?ticker=<mkt>&limit=1000 (cursor): full public tape, microsecond created_time, count_fp, yes/no price, taker_side. Verified populated for 2025-12-15 (6 trades), 2026-01-15 (638), 2026-03-15 (1000+ page). Tape = far finer than 1-min candles; build own bars. Also per-series pull w/o ticker likely works (returned data for ticker filter; not tested unfiltered).
- /historical/markets/{t}/candlesticks: period_interval 1/60/1440, max 5000 candles/request, bid/ask OHLC + price OHLC + volume + OI. (Old-market candles for the oldest market returned empty; trades are the better source.)
- Gap: no historical orderbook snapshots (live /orderbook empty after settle). Need a recorder or a vendor for depth.

## 2. Vendors with Kalshi orderbook/trades
- Oddpool https://docs.oddpool.com/kalshi/overview.md (V page): historical orderbook snapshots, top-of-book, trades from 2026-03-19; free tier incl.; ~1h lag; cursor paging. Snapshot interval + KXBTC15M coverage NOT stated (U). Worth a test = only source of historical depth.
- Allium https://docs.allium.so/historical-data/predictions/kalshi (V page): common.predictions.kalshi_trades, kalshi_market_orderbook, hourly candles; start date/price/snapshot freq not stated (U); warehouse = likely paid.
- HF naratron33/fable5-kalshi-l2 https://huggingface.co/datasets/naratron33/fable5-kalshi-l2 : gated (401), claims recorded L2 snapshots+trades parquet; coverage unknown (U).
- Apify scrapers (automation-lab/kalshi-historical-trades-scraper, reapx/kalshi-scraper): just wrap the same public endpoint; paid; pointless. EntityML (entityml.com) U.
- No Kaggle/HF public full Kalshi dataset found. GitHub DeweyMarco/simple-kalshi-bot = live paper bot on KXBTC15M, no data (U).

## 3. Polymarket BTC 15m analogue
- gensx-x1/polymarket-btc-eth-updown-data (GitHub, V page): SQLite+zstd, 4.06M rows, top-of-book up/down bid/ask every few s, 5m+15m windows, ONLY 2026-08-06..08-18. Free, small.
- SII-WANGZJ/Polymarket_data (HF, U): 107GB+ all on-chain fills/trades/markets parquet; moose-code/polymarket-onchain-v1 (HF, 127GB, DuckDB over hf://) U; trentmkelly/polymarket_historical_data (HF): orderbook snapshots every ~5 min from 2026-07-23 (U).
- Polymarket CLOB prices-history (1m fidelity, per token) + Gamma metadata + data-api.polymarket.com/trades (V: public, responds). Used in the Substack below for 23.9k 15m markets Nov-2025..Jul-2026.
- PolyTest (paid, 90M snapshots, sub-second) U; hunted.space free 5m data U. Settlement there = Chainlink, not BRTI.

## 4. BTC reference data
- Binance https://data.binance.vision (V): daily zips, spot trades/aggTrades BTCUSDT (HEAD 200: 2026-10-20 trades 21MB; 2017-08-17 aggTrades exists). Free, tick-level, ms stamps. Best free proxy for BRTI (BRTI = 1/s from Coinbase, Kraken, Bitstamp, Gemini, itBit, etc.; Binance is NOT a constituent).
- CF Benchmarks BRTI history (V via search): no free/self-serve download; licensing request via https://www.cfbenchmarks.com/data/indices/BRTI (paid, negotiated). Bloomberg ticker BRTI.
- Kalshi markets themselves carry expiration_value (the 60s BRTI avg) and floor_strike (prior window avg) => free ground truth for settlement at 15m resolution; use to calibrate a proxy.
- Tardis.dev (Coinbase/Kraken/Binance tick+book, docs.tardis.dev): free first-of-month samples claimed but my guessed download URLs 404'd -> U; paid otherwise. QuantConnect Coinbase tick/second data from 2015 (U). CryptoDataDownload Coinbase = archived, OHLC only (U).
- Coinbase/Kraken/Bitstamp public REST trades are recent-only -> can't build history from them.

## 5. Prior analyses
- https://viktoriatsybko.substack.com/p/does-polymarket-know-which-way-btc (V): 23,909 PM BTC-15m markets, Nov-2025..Jul-2026; pre-open prices calibrated within ~1c; no edge vs pre-open skew, on mids, fees would kill it; maker question open.
- indiehackers latency-arb post (U, strategy outline, no results). Searches found NO Kalshi KXBTC15M backtest/latency study. Note 60s-average settlement dilutes late spot spikes.

## Recommendation
1. Today: crawl /historical/markets + /historical/trades for all 24k KXBTC15M markets into parquet (cutoff keeps advancing; live window only ~2 months). Gives tick tape since 2025-12 vs the 70-day candles.
2. Pair with data.binance.vision BTCUSDT trades (+ Coinbase/Kraken if obtainable) to rebuild a BRTI proxy; calibrate against expiration_value/floor_strike. Start a own L2 recorder (or test Oddpool free tier) because depth/fills-at-quote can't be recovered from trades.
3. Use Polymarket 15m (CLOB history, HF dumps) only as cross-check; settles on Chainlink and the one public study found no edge on mids.
