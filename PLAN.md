# BTC 15m Kalshi × Timestral — research loop

## Morning summary (2026-10-26 05:05 CT)
**No edge found.** Nothing beats the Kalshi price after fees on held-out data. ~25 model/strategy configs tried.
- Timestral zero-shot (28M, 209M) ≈ random walk; fine-tuned on 1y BTC 1m: slightly better than zero-shot,
  still clearly worse than the market (brier k5: ft .1881 vs mid .1863). Adds nothing when stacked with the market.
- Feature GBDT, LR stacker, calibration/longshot bias, maker quoting (even optimistic fills): all lose.
- 30k-market rerun (Dec 2025–Oct 2026, Kalshi /historical): one positive cell (k5 GBDT +2.2¢) **failed** the
  pre-registered robustness check (neighbours negative, −2.7¢ with 1-min fill delay) → noise.
- Latency (1s recorder, ~3.5h): Kalshi tracks fair value at lag 0; no exploitable lag yet (small sample).
- Matches prior art: Polymarket 15m markets calibrated to ~1¢ (Tsybko).

**Still running (stop scripts in repo):** recorder.py (1s Kalshi+spot, `stop_recorder.sh`), paper_trader.py
(frozen stacker forward test, no orders; `paper_summary.json`, `stop_paper.sh`), fetch_hist.py (trade tapes,
~17k/30k, `kill $(cat fetch_hist.pid)`). Data: 2.6 GB in data/.
**If continuing:** (1) read paper_summary.json after a few days — only a forward result counts; (2) rerun latency.py
on days of recorder data; (3) order-flow features from trade tapes once fetch_hist finishes.

Goal: a model whose P(up) beats the Kalshi KXBTC15M market price **after taker fees** on held-out, later-in-time data.
Research/backtest only. No live orders.

## Ground rules (honesty)
- Bar to beat = Kalshi mid AND the random-walk baseline (`p_rw`). Zero-shot dry run: rw ≈ market already.
- Final claims are judged only on the **late_half** (chronological holdout). Anything tuned (thresholds, epochs, features)
  uses the early half / an early validation slice only. Never pick the best of many configs on the holdout and call it edge:
  log every config tried here, and report the number of configs tried with any result.
- PnL uses taker fee 0.07·P·(1-P) and fills at the ask/bid of the candle closing at entry. No maker-fill fantasies.
- Coinbase 1m ≠ BRTI; strike is basis-corrected causally.

## Data (data/, Kaggle dataset YOUR_KAGGLE_USER/kalshi-btc15m)
- markets.parquet: 6,467 settled markets, 2026-08-18 .. 2026-10-26
- kalshi_candles.parquet: 1-min YES bid/ask/last/vol per market (97k rows)
- btc_1m.parquet: Coinbase BTC-USD 1m OHLCV from 4 days before first market

## Experiment log
| # | date | config | where | holdout result | note |
|---|------|--------|-------|----------------|------|
| 0 | 10-26 | dry run baselines | local | mid≈rw (brier k1 .2381/.2382, k10 .1281/.1296) | partial data |
| 1 | 10-26 | zero-shot 28M+209M, ctx 2048, basis-corrected strike | kaggle v1 (28M 58s, 209M 391s / 19.4k forecasts on T4) | brier late k1/5/10: mid .2322/.1863/.1273; 209M .2350/.1892/.1321; 28M .2357/.1894/.1323; rw .2346/.1897/.1330. Every model×edge loses after fees (−1.7..−15¢/trade) | Timestral ≈ random walk, slightly better logloss than rw, clearly worse than market. out/zeroshot/ |
| 4 | 10-26 | fine-tune 28M (4k steps bs64 lr1e-5) + 209M (3k bs32), pinball loss on 365d Coinbase 1m pre-cut; best-on-val ckpt (1 config each) | kaggle timestral-btc15m-finetune (~50 min each) | val pinball .2756→.2744 (tiny). late brier k1/5/10: ft28M .2341/.1881/.1313, ft209M .2342/.1881/.1316 vs zs .2350/.1892/.1321 vs mid .2322/.1863/.1273. Trades: 17/18 cells lose; one +1.2¢ (28M k5 e.10, 46 trades) = noise | fine-tuning helps a little over zero-shot, still clearly worse than market. out/ft/results_finetune.json |
| 5 | 10-26 | favourite-longshot calibration of mid by price bucket (10 buckets × yes/no) | local | early: favourites >0.8 YES +0.7..1.2¢; late: 0.8–0.9 flips to −1.8¢, >0.9 only +0.2..0.4¢ (noise) | market well calibrated; no bias edge |
| 6a | 10-26 01:00 | latency.py on 29 min of 1s data (3 mkts), 9 configs e×L | local | too small: Δmid tracks Δp_fair at lag 0 (corr .33), spot leads by ~1s; sim 1–2 trades | rerun after hours of recording |
| 7a | 10-26 02:15 | eval_big.py 60/15/25 split, recent 6,467 mkts only (hist candles pending); mid/rw/LR stacker/HistGBDT | local | holdout (10-09..10-26) brier k1/5/10: mid .2312/.1865/.1285 best; GBDT .2364/.1899/.1316; all trade cells negative (GBDT k10 −3.4¢ CI[−5.7,−1.0]) | rerun on ~30k mkts when fetch_hist_candles finishes |
| 6b | 10-26 02:35 | latency.py on 127 min (10 mkts) | local | Δmid~Δfair peak lag 0 (.28); after 114 >2σ 3s spot moves fair +.058 but mid only +.023 by 30s — yet fair-vs-ask sim loses at all 9 e×L (−2..−10¢) | mid "underreaction" looks like fair-model overreaction (σ too small / spot bursts revert), not Kalshi lag. Keep recording |
| 8 | 10-26 03:20 | maker quoting sim (join/improve/both sides; optimistic any-trade fill + pessimistic strict-cross), maker fee, hold to settle; 7,873 mkts Dec–spring | local maker_sim.py | optimistic join −0.7..−1.6¢/contract, both-sides −1.1..−1.5¢ (CI excl 0), improve worse, pessimistic −2..−4¢; P(win|fill) 1–3 pts below quote (adverse selection) | upper bound already negative → passive quoting dead. Late half less negative (trend only) |
| 7b | 10-26 04:41 | eval_big.py full: 29,358 mkts, train 12-15..06-24 / val ..08-09 / holdout 08-09..10-26 (7,340 mkts); 4 models × 3 k | local 13s | brier holdout k1/5/10: mid .2347/.1909/.1288; gbdt .2359/.1907/.1302; stacker .2350/.1907/.1294. k1,k10: all val-chosen edges lose. **k5: gbdt e.02 1,402t +2.2¢ CI[−0.1,+4.5]; stacker 229t +3.5¢ CI[−0.8,+7.4]; rw e.1 239t +3.4¢** | first positive cell. 12 model×k tested → likely chance; see #9 pre-registered check |
| 9 | 10-26 | PRE-REGISTERED robustness for k5 signal (written before running): same gbdt hyperparams, same train/val, edge 0.02 fixed. PASS only if (a) k=4 and k=6 both positive per-trade on holdout, (b) ≥2 of 3 holdout months (Aug/Sep/Oct) positive at k5, (c) positive with fill at the NEXT minute's ask/bid (1-min latency) | robust_k5.py | **FAIL** (a) k4 −1.9¢, k6 +0.7¢; (b) Aug +6.7 / Sep +0.1 / Oct −1.4 (technical pass); (c) next-minute fill −2.7¢ CI[−3.9,−1.6]. gbdt k3..7: −1.1/−1.9/+1.8/+0.7/−2.2¢. Stacker k3..7: +9.1(22t)/+5.3(100t)/+3.5(229t)/+1.3(867t)/+0.6(516t) but next-min fill −3.1¢ | k5 = noise. Stacker lead not pre-registered → forward paper test only (#10) |
| 10 | 10-26 04:43 | paper_trader.py: frozen stacker k3..7 (paper_model.json), edge .02, fill now + 60s later, forward on live recorder data | running | pending (days) | the only honest test of the stacker lead |
| 3 | 10-26 | LR stack logit(mid)+logit(p_209M)(+rw), fit early half (3 configs×3 k) | local | adding Timestral makes late brier WORSE than mid alone (k10 .1286 vs .1272); trades −5..−14¢ | zero-shot Timestral carries no info beyond the market |
| 2 | 10-26 | HistGBDT 20 causal feats per k + LR stacker(mid,rw); edge picked on val (15 configs) | local, features_gbdt.py | brier late: mid .2322/.1863/.1273 (k1/5/10) beats rw .2346/.1897/.1330 and gbdt .2385/.1907/.1323; all edges lose (k5 CI [-.069,-.008]) | market > random walk: it has info beyond Coinbase 1m. Top feat dist_sig, mid, ret_5 |

## Idea queue (ordered)
00. **More data (10-26):** Kalshi /historical API has 24k KXBTC15M markets back to 2025-12-10 + full trade tapes (µs);
   cutoff moves forward → archive now (fetch_hist.py). Binance 1s klines as BTC proxy. See research_data_sources.md.
   fetch_hist.py RUNNING (10-26 ~01:40): 29,833 markets total, trades ~3–8h, then Binance 1s. Binance last-60s avg vs
   expiration_value: median |diff| $41.7 (USDT basis?) → needs causal basis correction like eval.py.
   Prior art: Tsybko Substack, 23.9k Polymarket 15m markets — calibrated to ~1¢, no edge on mids.
0. **Latency (top priority after #2):** market beats rw at 1m resolution → its info is sub-minute. Live 1s recorder
   (recorder.py, Kalshi orderbook + Coinbase/Kraken/Bitstamp spot) RUNNING since 10-26 ~01:30 CT → live/*.parquet,
   stop with stop_recorder.sh. Kalshi ~60 rows/min; ~12s gap at each 15m roll. Analyze after several hours.
1. Zero-shot result (above).
2. Calibration: isotonic/Platt on early half for model + rw; blend model with market mid (logit stacking).
3. Covariates: Kalshi mid/spread/volume as past covariates; volume, high-low range of BTC; time-of-day.
4. Fine-tune Timestral-28M (then 209M) on BTC 1m with pinball loss, train only < holdout start; Kaggle T4.
5. Directly-trained classifier head on frozen Timestral embeddings → P(up) (log-loss), vs GBDT on handcrafted features.
6. Other foundation models zero-shot for comparison (Chronos-2, TimesFM) if cheap.
7. More data: more history of BTC 1m for fine-tuning (Coinbase back years); Kalshi markets older than ~70d are purged.
