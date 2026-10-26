# Timestral research: fine-tuning setup and findings

My notes, code and results from working with Mistral's [Timestral](https://huggingface.co/mistralai/Timestral-209M)
time-series foundation models (28M and 209M): a **fine-tuning script** (Mistral doesn't ship one), the eval harness,
and what zero-shot vs fine-tuned actually buys you. The test bed was a hard, well-priced real-world target:
Kalshi BTC 15-minute markets.

## Benchmark: Kalshi BTC 15-minute markets

Can Mistral's [Timestral](https://huggingface.co/mistralai/Timestral-209M) time-series foundation model, zero-shot or
fine-tuned, beat the market price on Kalshi's **KXBTC15M** ("Will BTC be up in the next 15 minutes?") after fees?

**Short answer: no.** Across ~25 model/strategy configurations on 29,358 settled markets (Dec 2025 – Oct 2026),
nothing beat the Kalshi mid after taker fees on a chronological holdout. The market is calibrated to about 1¢.
This repo publishes the code (including a **fine-tuning script for Timestral**, which Mistral does not ship),
the results, and the full experiment log so others don't have to repeat it.

> Research/backtest only. No orders were ever placed. Not financial advice.

## Setup

- **Market:** KXBTC15M settles YES if the CF Benchmarks BRTI 60-second average at close ≥ the 60-second average at open (`floor_strike`).
- **Fees:** taker `0.07·P·(1−P)`, maker `0.0175·P·(1−P)` per contract.
- **Entry points:** k = 1, 5, 10 minutes into the market (later: k = 3..7). Fills at the ask/bid of the candle closing at entry.
- **BTC data:** Coinbase BTC-USD 1m (Binance 1s as a proxy for latency work). BRTI ≠ Coinbase: the strike runs ~$2.6 above
  the Coinbase close, so the strike is basis-corrected with a causal rolling-200 median.
- **Kalshi data:** the live `/markets` endpoint keeps only ~70 days, but the unauthenticated `/historical/markets`,
  `/historical/trades` and `/historical/markets/{t}/candlesticks` endpoints go back to 2025-12-10 (field names are unsuffixed
  there, e.g. `close`, vs `*_dollars` live).
- **Split:** chronological. First run: early/late half of 6,467 recent markets. Full run: 60/15/25 train/val/holdout —
  train 2025-12-15..2026-06-24 (17,614 markets), val ..2026-08-09 (4,404), holdout 2026-08-09..2026-10-26 (7,340).
  Thresholds/epochs/edges chosen on val only. Every config tried is logged in [PLAN.md](PLAN.md).
- **Probability from quantiles:** `prob_above(q, strike)` in `eval.py` linearly interpolates Timestral's 21-quantile CDF at the strike.

## Results (holdout)

### Probability quality — Brier score (lower is better), k = 1 / 5 / 10 min

| Model | k1 | k5 | k10 |
|---|---|---|---|
| **Kalshi mid** | **.2322** | **.1863** | **.1273** |
| Random walk (Gaussian, realised vol) | .2346 | .1897 | .1330 |
| Timestral-28M zero-shot | .2357 | .1894 | .1323 |
| Timestral-209M zero-shot | .2350 | .1892 | .1321 |
| Timestral-28M fine-tuned | .2341 | .1881 | .1313 |
| Timestral-209M fine-tuned | .2342 | .1881 | .1316 |
| HistGBDT, 20 causal features | .2385 | .1907 | .1323 |

(First run, 6,467 markets, late-half holdout.) On the 29k-market run: mid .2347/.1909/.1288, GBDT .2359/.1907/.1302,
LR stacker (mid + random walk) .2350/.1907/.1294.

### Findings

1. **Timestral zero-shot ≈ random walk** on 1-minute BTC. Slightly better log-loss than the random walk, clearly worse than the market.
   Every model × edge cell lost after fees (−1.7 to −15¢ per trade).
2. **Fine-tuning helps a little, not enough.** Pinball loss on 365 days of Coinbase 1m before the holdout (28M: 4k steps bs64;
   209M: 3k steps bs32; lr 1e-5; ~50 min each on a Kaggle T4). Validation pinball .2756 → .2744. 17 of 18 trading cells lose; the
   one positive (+1.2¢, 46 trades) is noise.
3. **Timestral adds no information to the market.** Logistic stacking of logit(mid) + logit(p_Timestral) makes holdout Brier *worse* than mid alone (k10 .1286 vs .1272).
4. **The market beats the random walk** → it carries information beyond Coinbase 1m prices (top GBDT features: distance-to-strike in σ, mid, 5-min return).
5. **No favourite-longshot bias to exploit.** Early half: favourites > 0.8 earned +0.7..1.2¢; late half the 0.8–0.9 bucket flipped to −1.8¢.
6. **Passive (maker) quoting loses even with optimistic fills:** −0.7 to −1.6¢ per contract (CI excludes 0), pessimistic −2 to −4¢.
   Adverse selection: P(win | filled) is 1–3 points below the quoted price.
7. **One positive cell failed a pre-registered robustness check.** Full run, k5 GBDT edge 0.02: +2.2¢ over 1,402 trades, CI [−0.1, +4.5].
   Criteria written before running: (a) k4 and k6 positive → k4 −1.9¢, k6 +0.7¢ **fail**; (b) ≥2 of 3 months positive → Aug +6.7, Sep +0.1,
   Oct −1.4 (technical pass); (c) positive with a 1-minute-later fill → **−2.7¢** CI [−3.9, −1.6] **fail**. Across k3..7: −1.1/−1.9/+1.8/+0.7/−2.2¢. Noise.
   The stacker looked positive at k3–7 on small counts (+9.1¢/22 trades … +0.6¢/516) but −3.1¢ with the 1-minute delay.
8. **No exploitable latency at 1-second resolution** (recorder: Kalshi orderbook + Coinbase/Kraken/Bitstamp trades, ~3.5 h, 10 markets).
   Δmid tracks Δfair at lag 0 (corr .28); spot leads by ~1 s. After large 3-s spot moves fair value jumps more than the mid, but trading that
   gap loses at all 9 edge × lag settings (−2 to −10¢) — the "underreaction" is the fair model overreacting to bursts that revert.
9. **Forward paper test** (frozen stacker, k3–7, edge 0.02) ran ~5 h: 4 trades in 1 of 19 settled markets (+12¢/trade, i.e. one coin flip). Too sparse to mean anything; stopped.
10. **Prior art agrees:** a study of 23.9k Polymarket 15-minute BTC markets found them calibrated to ~1¢ (see [research_data_sources.md](research_data_sources.md)).

**Takeaway:** for a near-martingale target where the market already prices public spot data in real time, a larger forecasting model
doesn't help. An edge would need information or speed the market doesn't have, not a better model of the same price series.

## Files

| File | What |
|---|---|
| `fetch.py` | Settled markets, 1-min Kalshi candles, Coinbase 1m (certifi SSL) |
| `fetch_hist.py`, `fetch_hist_candles.py` | Kalshi `/historical` markets, trade tapes, candles; Binance 1s |
| `fetch_btc_long.py` | 365 days of Coinbase 1m for fine-tuning |
| `eval.py` | Zero-shot evaluator: dataset build, `prob_above`, Brier/log-loss, fee-aware `trade()` |
| `finetune.py` | **Timestral fine-tuning** (pinball loss through the library's own `forecast` path) |
| `features_gbdt.py`, `eval_big.py` | GBDT + stacker baselines; 30k-market 60/15/25 evaluation |
| `robust_k5.py` | Pre-registered robustness check |
| `maker_sim.py` | Maker quoting simulation |
| `recorder.py`, `latency.py` | 1-s live recorder and lead-lag analysis |
| `paper_trader.py`, `paper_model.json` | Frozen forward paper trader |
| `kaggle/` | Kaggle kernels (zero-shot, fine-tune) and dataset metadata |
| `results/` | All result JSONs, paper trades |
| `PLAN.md` | Full experiment log, including every config tried |

## Fine-tuning Timestral

Mistral's repo ships inference only. `finetune.py` calls `TimestralForecaster.forecast.__wrapped__` (the same function without
`@inference_mode`) so masking, causal patch normalisation, asinh, the quantile head and denormalisation are the library's own code,
and trains with a horizon-weighted pinball loss:

```bash
pip install -r requirements.txt
python fetch.py && python fetch_btc_long.py         # builds data/
python finetune.py --data data --out . --model mistralai/Timestral-28M --steps 3000
```

On Kaggle: upload `data/*.parquet` + `eval.py` (as `eval_mod.py`) + `finetune.py` as a dataset, then push `kaggle/finetune`.

Market data is not redistributed here; the fetch scripts rebuild it from public endpoints.

## License

Code: MIT. Timestral weights are under Mistral's own license and are not included.
