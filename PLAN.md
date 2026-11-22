# Project 4: the cheapest order-flow model that still pays (PLAN)

Written 2026-11-21, before any project code. The "Changes during the build" log at the end records every deviation.

## Question

What is the cheapest order-flow model whose short-horizon forecasts still pay the spread and fees on a real
crypto perpetual? Build a ladder of models of increasing cost, score each on net P&L after realistic costs and
latency, and report the frontier of compute cost against net profit.

## Literature (verified by me on 2026-11-21; only these claims are cited)

| Paper | What I verified | Where |
|---|---|---|
| Cont, Kukanov & Stoikov, "The Price Impact of Order Book Events", J. Financial Econometrics 12(1):47-88, 2014 (arXiv 1011.6402) | Event contribution `e_n = I{Pb_n>=Pb_n-1} qb_n - I{Pb_n<=Pb_n-1} qb_n-1 - I{Pa_n<=Pa_n-1} qa_n + I{Pa_n>=Pa_n-1} qa_n-1`; `OFI_k` = sum of `e_n` over the interval; model `dP_k = beta OFI_k + eps_k` (eq. 2) with `beta_i = c / AD_i^lambda` (eq. 3), AD = average depth at the best quotes. They use a 10-second grid, TAQ April 2010, 50 S&P 500 stocks. **The relation is contemporaneous**, not a forecast. | arXiv PDF p.4 (e_n, OFI), p.6 (eq. 1-3), p.8 (data, grid) |
| Kolm, Turiel & Westray, "Deep order flow imbalance: Extracting alpha at multiple horizons from the limit order book", Mathematical Finance 33(4):1044-1081, 2023 | 115 Nasdaq stocks; "models trained on order flow significantly outperform most models trained directly on order books"; "the effective horizon of stock specific forecasts is approximately two average price changes". (Multi-level OFI detail not verified from the abstract, so not cited.) | EconPapers abstract page |
| Zhang, Zohren & Roberts, "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books", IEEE Trans. Signal Processing 67(11), 2019 (arXiv 1808.03668) | CNN filters over the order book plus LSTM for time dependence; one year of LSE quotes. | arXiv abstract |
| Briola, Bartolucci & Aste, "Deep limit order book forecasting: a microstructural guide", Quantitative Finance 25(7):1101-1131, 2025 (arXiv 2403.09267) | "high forecasting power does not necessarily correspond to actionable trading signals"; "traditional machine learning metrics fail to adequately assess the quality of forecasts"; horizons in LOB updates (10, 50, 100); propose the probability of forecasting complete transactions. | arXiv v4 abstract |

## Data (measured 2026-11-21 from the data.binance.vision S3 listing)

- USDT-M futures **bookTicker** daily files exist from **2023-05-16 to 2024-03-30** (320 days, no gaps) for BTCUSDT,
  ETHUSDT and SOLUSDT. (The brief said "before 2024-06"; the listing shows the last file is 2024-03-30.)
  - BTCUSDT: 52.6 GB zipped in total, 164 MB/day on average (54-150+ MB); one day unzips to ~1.1 GB CSV, ~11.7 M updates.
  - ETHUSDT: 43.3 GB, 135 MB/day. SOLUSDT: 23.5 GB, 73 MB/day.
- **aggTrades** (USDT-M, daily): BTC ~15 MB/day, ETH ~12 MB/day zipped.
- Download speed measured at ~15 MB/s; parsing one BTC day with polars on 2 threads takes ~5 s.

**Sampling (fixed before looking at any results).** A full year of all three symbols is ~120 GB, above the 60 GB raw budget.
I keep every **Tuesday, Thursday and Saturday** (one weekend day per week) in the window, minus the two gap days below:
~136 days per symbol. Raw bytes: BTC ~24.5 GB, ETH ~20 GB, SOL ~11 GB, so ~56 GB **downloaded in total** (and never more than
~2 GB on disk at once, since zips are deleted after their parquet is verified). Order: BTC, then ETH, then SOL; SOL is dropped
first if time or disk runs short. Each sampled day is a self-contained session (features warm up inside the day; targets that
cross midnight are dropped), so non-contiguous days cause no leakage.

## Pipeline and layout (same style as project 1)

`python -m ofp.cli <stage>` with stages `download | build | train | validate | freeze | test | report`.

- `download`: for each (symbol, day): download bookTicker + aggTrades zips and their `.CHECKSUM`, verify SHA-256, convert to a
  1-second **bar parquet** (86,400 rows), verify it (row count, quote-update count = raw rows, traded quantity sums equal the
  raw totals, trade counts equal), write a manifest row, then delete the zips. Resumable: a verified day is skipped. Download
  and bar-building are fused per day so that raw zips never pile up on disk.
- `build`: features and targets per (symbol, day) parquet, plus split assignment.
- `train`: fit every rung on the train split; record parameters, training time, CPU microseconds per prediction.
- `validate`: forecasts and trading simulations on validation for the whole grid.
- `freeze`: choose horizon, threshold multiple and hyperparameters on validation only; write `results/frozen_config.json` with a SHA-256 hash.
- `test`: run once on test; append to `results/test_log.jsonl` (timestamp, config hash, code hash). A rerun needs `--force` and is logged.
- `report`: tables, figures, `results/tables.md`; README numbers come from these files only.

Caches: `~/.cache/quant-research/p4/` (bars, features, manifest). `~/.cache/qal/binance/` is not touched.
Every process sets `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=POLARS_MAX_THREADS=2`, `torch.set_num_threads(2)`, LightGBM `num_threads=2`.
One heavy process at a time; long jobs run in the background with logs.

## Bars (1 second, per symbol-day)

Bucket k holds events with `transaction_time` in `[k*1000, (k+1)*1000)` ms of the day.
- `ofi`: sum of CKS `e_n` over the quote updates in the bucket (updates ordered by `update_id`, first update of the day has e=0).
- quote state at the bucket close (last update before `(k+1)*1000`, carried forward): bid, ask, bid size, ask size;
  quote state at close + 100 ms (for the 100 ms latency); number of updates; mean depth `(qb+qa)/2` over the updates.
- trades: buy/sell quantity and counts (taker side from `is_buyer_maker`), min and max trade price (maker fill check).

## Features (at the end of bucket t, from bars <= t only)

OFI normalised by a trailing 300-s mean depth (`ofi_n`, the CKS depth scaling), its lags 1-2 and trailing sums over 5, 30, 60,
300 s; trade-count imbalance over 1, 5, 30 s; signed volume / depth over 1, 5, 30 s; queue imbalance `(qb-qa)/(qb+qa)`; spread
in bps; past mid returns over 1, 5, 30, 60, 300 s; realised vol over 60 and 300 s; log update activity over 60 s.
The first 300 s of every day (warm-up) are dropped.

**Targets:** `y_h = (mid_{t+h} - mid_t) / mid_t * 1e4` bps, mid at bucket close, h in {1, 5, 30, 60, 300} s.

## Model ladder (same features and targets everywhere; pooled over symbols, one model per rung)

1. **Sign of the last OFI bucket**: forecast = sign(OFI_t); 0 parameters. It has no magnitude, so it trades on every non-zero
   OFI bucket when flat; only its horizon is chosen on validation.
2. **Linear CKS**: `y_h = a_h + b_h * ofi_n_t` (OLS, one feature, 2 parameters per horizon).
3. **Ridge**: all features, standardised; alpha from a 3-value grid by validation MSE.
4. **LightGBM**: one model per horizon, early stopping on validation, num_leaves in {15, 63}.
5. **MLP**: features -> 64 -> 32 -> 5 horizons (standardised targets), Adam, early stopping on validation loss.
6. **DeepLOB-style CNN+LSTM** on the last K=30 buckets of 8 per-bucket order-flow channels -> 5 horizons (optional; skipped
   and recorded if time is short).

Training rows: a seeded random subsample of train buckets (to fit the 2-core budget); validation and test use every bucket.
Regression targets are winsorised at the train 0.05/99.95 % quantiles for fitting only (never for scoring).
Reported per rung: parameters (trees: 2*leaves-1 per tree), training seconds, CPU µs per prediction (single-row call and
amortised in a 10,000-row batch, 2 threads).

## Trading simulation (the point of the project)

- Signal at the end of bucket t. Execution after latency L in {0, 100 ms, 1 s} at the then-prevailing best quote: taker buys
  at the ask and sells at the bid. Exit after holding h, as a taker, at the then-prevailing quote.
- One position per symbol at a time (signals are ignored while a position is open); unit notional per trade.
- Trade only when `|forecast| > m * expected_cost`, m from the pre-set grid {0.25, 0.5, 1, 1.5, 2, 3} (0.25 added after the thin slice, see changes); expected cost =
  round-trip fees + mean train spread (bps).
- **Fees** (Binance USDT-M, levels as given in the task brief on 2026-11-21; the Binance fee page is JS-rendered and could not be
  fetched to re-check): VIP0 taker 0.05 % / maker 0.02 %; cheaper tier taker 0.03 % / maker 0.01 %; zero fees.
- **Maker variant (optimistic, labelled):** post at the touch at t+L (bid to buy, ask to sell); filled at that price if any
  trade prints strictly through it during the next h seconds (bucket-level min/max, so up to 100 ms of trades before posting
  can count at L=100 ms); exit as taker at t+L+h. Maker fee in, taker fee out.
- **Accounting:** gross = direction * mid move between execution times (bps of entry); spread cost = executed prices vs mids;
  net = direction*(exit-entry)/entry - fee_in - fee_out*exit/entry. Daily P&L = sum of net bps of trades entered that day,
  summed over symbols (equal notional). Days without trades count as 0.
- **Metrics:** gross and net bps/trade, hit rate, trades, daily net P&L, Sharpe (daily, annualised sqrt(365), sampled days),
  deflated Sharpe (Bailey & López de Prado; functions **copied** from Alpha Audit with attribution; trials = every
  (model, hyperparameter, horizon, multiple) configuration evaluated on validation for that cost scenario; trial Sharpe variance
  from those validation runs), turnover (notional traded per day per unit capital), net P&L at every latency x fee cell,
  forecast R² (vs a zero forecast) and directional accuracy (non-zero moves).
- Main scenario for the main table: **taker, VIP0, L = 100 ms**. Every latency x fee x {taker, maker} cell gets its own
  validation-chosen (horizon, multiple), frozen before test.
- Selection rule on validation: maximise total net P&L subject to >= 50 validation trades (so a rung that never trades is not
  "best" by default). If even the best config loses on validation, that is reported as "does not pay".

## Protocol (no tuning on test)

Calendar split of 2023-05-16 → 2024-03-30 (320 days), then sampled:
- train 2023-05-16 → 2023-11-21 (190 days, 59 %); gap day 2023-11-22;
- validation 2023-11-23 → 2024-01-08 (47 days, 15 %); gap day 2024-01-09;
- test 2024-01-10 → 2024-03-30 (81 days, 25 %).
Thresholds, horizons and hyperparameters are chosen on validation only, frozen with a hash, then test runs once.

## Tests (pytest)

1. CKS OFI on a hand-made update sequence equals the formula. 2. Targets strictly in the future. 3. Taker fills use ask to buy /
bid to sell after the latency (toy book). 4. Fee and P&L accounting on a toy trade list. 5. Leak test: changing raw data after
bucket t leaves features and forecasts at t unchanged. 6. Sharpe and DSR equal the Alpha Audit functions on fixed inputs (the
test reads the Alpha Audit source text and executes those two functions in isolation, so nothing is imported from that package).
7. Bucketing loses no trades (volumes and counts sum to the raw totals).

## Risks and fallbacks

- Download too slow / disk tight: drop SOL, then ETH, then go to 2 days/week (recorded).
- Training too slow on 2 cores: smaller train subsample; skip the CNN rung (recorded).
- Fees dominate: BTC spreads are ~0.03 bps, so taker round-trip VIP0 costs ~10 bps. It is likely that nothing pays at VIP0;
  the frontier at zero/low fees and the maker variant then carry the comparison. This is an answer, not a failure.
- Stalls: every job > 2 min runs in the background with a log; downloads resumable per day.
- Extension (2024-06 → 2026-10, trades + bookDepth only): only if time allows; otherwise recorded as not done.

## Changes during the build

(appended as they happen)
- 2026-11-21 (data): bookTicker ends 2024-03-30, not "before 2024-06"; the window is 2023-05-16 → 2024-03-30. The first
  BTC file (2023-05-16) starts mid-day (~50k empty seconds); its early hours simply have no valid buckets.
- 2026-11-21 (pipeline): `download` fuses download + bar building per day (with a 1-day prefetch thread) so that raw zips
  never accumulate; `build` makes features from the verified bars.
- 2026-11-21 (thin slice, first 6 BTC days): linear CKS OFI has 70 % directional accuracy at 1 s, R² 1.9 %, yet at VIP0 no
  forecast ever exceeds 1x the ~10 bps round-trip cost. To make the VIP0 column informative, the threshold grid gains a
  0.25x multiple: {0.25, 0.5, 1, 1.5, 2, 3} (fixed before any validation run).
- 2026-11-21 (selection): the per-horizon test table is selected on validation for two scenarios (taker VIP0 100 ms and
  taker zero-fee 100 ms) instead of reusing VIP0 multiples at other fee levels.
- 2026-11-21 (dev dry run on 14 BTC days, scratch outputs, before any real validation): DSR trial-Sharpe variance is taken
  over the configs with >= 50 validation trades in the scenario (all simulated configs still count in the number of
  trials); LightGBM budget set to 1.5 M train rows, learning rate 0.1, <= 300 trees (2-core budget).
- 2026-11-21 (scope): the 2024-06 → 2026-10 trades + bookDepth extension is **not done**. It was optional; without L1
  quotes it cannot measure the spread (the question is whether forecasts pay the spread and fees), and the main study already
  transfers ~56 of the 60 GB raw budget. Instead the report adds a day-level bootstrap CI of daily net P&L and an
  "edge before fees" table (pre-fee bps per trade of the validation-chosen strategy at each fee level).
- 2026-11-21 (data quality): 10 BTC days (e.g. 2023-11-21, 2024-02-10..02-20) have bookTicker rows out of file order. A check
  of 2024-02-10 (re-downloaded to a scratch folder, then deleted): 28.0 M rows, all update_ids unique, no duplicate rows, and
  after sorting by update_id the timestamps are monotonic. The file interleaves two ordered streams, so the bar builder (which
  sorts by update_id for OFI and by time for quote states) is correct on these days. Every day passed the trade-volume and
  trade-count checks exactly (max relative volume difference 4e-16).
- 2026-11-22 (train, before any validation simulation): pooled over symbols, the one-slope CKS regression had validation R²
  of only 0.45 % at 1 s (1.9 % on BTC alone): in bps, the impact of OFI/depth scales with the tick size in bps, which differs
  a lot between BTC, ETH and SOL, and CKS fit one slope per stock. The two linear rungs (linear OFI, ridge) are therefore fit
  **per symbol** (parameters x3); LightGBM, MLP and CNN stay pooled (they see spread and volatility, which carry the scale).
  The CNN epoch cap goes from 3 to 8 (its validation loss was still falling at epoch 3; early stopping still applies).
- 2026-11-22 (results): test run once at 2026-11-22T00:24Z (config d0f8794d05315277, code 8d8398de5f95ea24), 291 s.
  After the test only `report.py` changed: a PSR column (from the saved test daily P&L) and the figure layout (symlog
  frontier panel, split cumulative-P&L figure). No computation of any test number changed. The "done" line of the test log
  repeats the start timestamp (cosmetic; left as is so the tested code stays unchanged).
- 2026-11-22 (totals): 411 symbol-days, 4.67 B quote updates, 377 M aggTrades, 55.3 GB transferred, 5.6 GB cache
  (bars 1.1, features 3.0, predictions 1.6). Wall clock about 2 h 25 min from start (22:12 UTC) to README (00:37 UTC).
