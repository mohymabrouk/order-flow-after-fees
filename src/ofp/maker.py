"""Maker-or-taker per order (Change 16 in PLAN.md): a NEW analysis on the frozen models, after the overlay.

Same parent orders and latency as the execution-timing overlay (`overlay.py`). For each order, decided once at the close
of its start bucket k0: if the frozen model's forecast says the price will move away by more than theta bp, cross now as a
taker; otherwise post a limit at the near touch 100 ms later, rest for T seconds, and if it is not filled cancel and cross
as a taker (effective at the end of the wait, at the quote prevailing then). Two fill rules for the resting order:
(a) trade-through: an aggTrade prints strictly through the limit price; (b) queue: opposite-aggressor volume at or through
the limit price after posting exceeds the displayed queue at posting plus the order size. Costs are in bp of the arrival
mid and include fees (maker on a passive fill, taker otherwise).

Data: the fill rules need every aggTrade and the queue at the touch, which the 1-s bars do not keep, so `maker_data`
re-downloads the validation and test days (same Binance files, SHA-256 checked), streams bookTicker into a 100 ms grid of
quote states, keeps aggTrades, and verifies both against the cached bars before deleting the zips.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone

from ofp import config  # noqa: F401  (thread pins first)

import numpy as np
import pandas as pd
import polars as pl

from ofp import bars as B
from ofp import overlay as O

log = logging.getLogger("maker")

# ---- pre-declared design (PLAN.md, Change 16) ------------------------------------------------------------------------
WAITS = (1, 5, 10, 30)  # T, seconds the passive order rests
THETAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)  # bp: cross now if side * forecast > theta
FORECAST_H = (1, 5, 30)  # forecast horizons (s) tried as the signal
RULES = ("through", "queue")
HEADLINE_RULE = "through"
ORDER_NOTIONAL_USDT = 10_000.0  # order size for the queue rule
MARKOUTS = (1, 10)  # seconds after a passive fill
FEE_TAKER, FEE_MAKER = config.FEES["vip0"]  # 5 bp / 2 bp (published during the test period)
FEE_TAKER_ALT = 0.0004  # 4 bp taker: the published rate until at least 2023-09-27 (sensitivity)
TEST_MODELS = ("sign_ofi", "ridge", "mlp")  # the three models scored in the overlay test
LATENCY_MS = config.LATENCY_OFFSET_MS  # 100 ms for every action, as in the overlay
GRID_MS = 100
SPD = config.SECONDS_PER_DAY
N_GRID = SPD * 1000 // GRID_MS + 2  # grid points j = 0 .. 864,001 (state at day start + j * 100 ms)
CHUNK_BYTES = 128 << 20
DEV_DAYS = (("BTCUSDT", date(2023, 11, 21)), ("SOLUSDT", date(2023, 11, 18)))  # train days for the code dry run

MAKER_DIR = config.CACHE / "maker"
H = list(config.HORIZONS)
QUOTE_DTYPES = {"update_id": pl.Int64, "best_bid_price": pl.Float64, "best_bid_qty": pl.Float64,
                "best_ask_price": pl.Float64, "best_ask_qty": pl.Float64, "transaction_time": pl.Int64,
                "event_time": pl.Int64}


def design() -> dict:
    return {"waits_s": WAITS, "thetas_bp": THETAS, "forecast_h": FORECAST_H, "rules": RULES, "headline_rule": HEADLINE_RULE,
            "order_notional_usdt": ORDER_NOTIONAL_USDT, "markouts_s": MARKOUTS, "fee_taker": FEE_TAKER,
            "fee_maker": FEE_MAKER, "fee_taker_alt": FEE_TAKER_ALT, "latency_ms": LATENCY_MS, "grid_ms": GRID_MS,
            "test_models": TEST_MODELS, "start_step_s": O.START_STEP, "overlay_windows": O.WINDOWS}


# ================================================================================================= data stage
def day_start_ms(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def data_dir(dev: bool = False):
    return MAKER_DIR / "dev" if dev else MAKER_DIR


def grid_path(symbol: str, d: date, dev: bool = False):
    return data_dir(dev) / symbol / f"{symbol}_{d.isoformat()}_q100.parquet"


def trades_path(symbol: str, d: date, dev: bool = False):
    return data_dir(dev) / symbol / f"{symbol}_{d.isoformat()}_trades.parquet"


def iter_csv_chunks(zip_path, chunk_bytes: int = CHUNK_BYTES):
    """Yield the CSV inside a zip as byte chunks that end on a line break (streamed: never the whole file in memory)."""
    with zipfile.ZipFile(zip_path) as zf:
        name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        with zf.open(name) as fh:
            buf = b""
            while True:
                data = fh.read(chunk_bytes)
                if not data:
                    if buf.strip():
                        yield buf
                    return
                buf += data
                cut = buf.rfind(b"\n") + 1
                if cut > 0:
                    yield buf[:cut]
                    buf = buf[cut:]


def grid_states_from_chunks(chunks, day_start: int, n_grid: int = N_GRID, grid_ms: int = GRID_MS) -> tuple[dict, int]:
    """Quote state at G_j = day_start + j * grid_ms for j = 0..n_grid-1: the last bookTicker update with ts < G_j, ties
    broken by the highest update_id (the bars' rule). Works on any row order: each chunk is reduced to the last row per grid
    cell, then the cells are merged and carried forward. Returns ({bid, ask, bidq, askq}, number of rows read)."""
    names, parts, n_rows = None, [], 0
    for chunk in chunks:
        if names is None:
            if not chunk[:1].isdigit():  # header line
                nl = chunk.index(b"\n")
                names = [c.strip() for c in chunk[:nl].decode().strip().split(",")]
                chunk = chunk[nl + 1:]
            else:
                names = B.QUOTE_COLS + ["event_time"]
            if not chunk.strip():
                continue
        df = pl.read_csv(chunk, has_header=False, schema={c: QUOTE_DTYPES.get(c, pl.String) for c in names}).select(B.QUOTE_COLS)
        n_rows += df.height
        g = ((pl.col("transaction_time") - day_start) // grid_ms + 1).clip(lower_bound=0)
        part = (df.with_columns(g.alias("g")).filter(pl.col("g") < n_grid)
                .sort(["g", "transaction_time", "update_id"]).group_by("g", maintain_order=True).last())
        parts.append(part)
        del df
    empty = {c: np.full(n_grid, np.nan) for c in ("bid", "ask", "bidq", "askq")}
    if not parts:
        return empty, n_rows
    allp = (pl.concat(parts).sort(["g", "transaction_time", "update_id"]).group_by("g", maintain_order=True).last())
    gs = allp["g"].to_numpy()
    idx = np.searchsorted(gs, np.arange(n_grid), side="right") - 1  # last non-empty cell g <= j
    out = {}
    for src, dst in (("best_bid_price", "bid"), ("best_ask_price", "ask"), ("best_bid_qty", "bidq"), ("best_ask_qty", "askq")):
        v = allp[src].to_numpy().astype(np.float64)
        out[dst] = np.where(idx >= 0, v[np.maximum(idx, 0)], np.nan)
    return out, n_rows


def verify_day(grid: dict, trades: pl.DataFrame, bars: pl.DataFrame, day_start: int, n_buckets: int = SPD) -> dict:
    """Checks against the cached 1-s bars: grid states at close and close + 100 ms equal the bars' quotes exactly; per-bucket
    trade counts and buy/sell quantities equal the bars."""
    k = np.arange(n_buckets)
    jc, j1 = 10 * (k + 1), 10 * (k + 1) + 1
    res = {}
    for gcol, bcol, j in (("bid", "bid", jc), ("ask", "ask", jc), ("bid", "bid100", j1), ("ask", "ask100", j1)):
        res[f"eq_{bcol}"] = bool(np.array_equal(grid[gcol][j], bars[bcol].to_numpy().astype(np.float64), equal_nan=True))
    kt = ((trades["ts"].to_numpy() - day_start) // 1000).astype(np.int64)
    buy = ~trades["is_buyer_maker"].to_numpy()
    qty = trades["qty"].to_numpy()
    for name, m in (("buy", buy), ("sell", ~buy)):
        n_ = np.bincount(kt[m], minlength=n_buckets)
        q_ = np.bincount(kt[m], weights=qty[m], minlength=n_buckets)
        res[f"eq_{name}_n"] = bool(np.array_equal(n_, bars[f"{name}_n"].to_numpy()))
        res[f"eq_{name}_qty"] = bool(np.allclose(q_, bars[f"{name}_qty"].to_numpy(), rtol=1e-9, atol=1e-9))
    res["ok"] = all(res.values())
    return res


MANIFEST_FIELDS = ["symbol", "day", "split", "bt_bytes", "at_bytes", "n_quotes_raw", "n_trades_raw", "n_trades_in_day",
                   "eq_bid", "eq_ask", "eq_bid100", "eq_ask100", "eq_buy_n", "eq_buy_qty", "eq_sell_n", "eq_sell_qty",
                   "quotes_match_main_manifest", "trades_match_main_manifest", "secs_download", "secs_build", "written_utc"]


def _manifest(dev: bool):
    return data_dir(dev) / "manifest.csv"


def manifest_rows(dev: bool = False) -> list[dict]:
    p = _manifest(dev)
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def process_day(symbol: str, d: date, fetched, dev: bool = False) -> dict:
    from ofp import download as D

    (bt_zip, bt_bytes), (at_zip, at_bytes), secs_dl = fetched
    t1 = time.time()
    ds = day_start_ms(d)
    grid, n_q = grid_states_from_chunks(iter_csv_chunks(bt_zip), ds)
    trades = B.read_trades_csv(D._read_zip(at_zip))
    n_t_raw = trades.height
    trades = (trades.filter((pl.col("ts") >= ds) & (pl.col("ts") < ds + SPD * 1000))
              .sort("ts", maintain_order=True).select("ts", "price", "qty", "is_buyer_maker"))
    bars = pl.read_parquet(D.bar_path(symbol, d), columns=["k", "bid", "ask", "bid100", "ask100", "buy_qty", "sell_qty",
                                                           "buy_n", "sell_n"]).sort("k")
    chk = verify_day(grid, trades, bars, ds)
    main = {(r["symbol"], r["day"]): r for r in D.manifest_rows()}.get((symbol, d.isoformat()))
    chk_q = main is not None and int(main["n_quotes_raw"]) == n_q
    chk_t = main is not None and int(main["n_trades_raw"]) == n_t_raw
    if not (chk["ok"] and chk_q and chk_t):
        raise RuntimeError(f"maker data verification failed for {symbol} {d}: {chk}, quotes {chk_q}, trades {chk_t}")
    gp, tp = grid_path(symbol, d, dev), trades_path(symbol, d, dev)
    gp.parent.mkdir(parents=True, exist_ok=True)
    for frame, path in ((pl.DataFrame(grid), gp), (trades, tp)):
        tmp = path.with_suffix(".tmp.parquet")
        frame.write_parquet(tmp, compression="zstd")
        tmp.replace(path)
    bt_zip.unlink()
    at_zip.unlink()
    row = {"symbol": symbol, "day": d.isoformat(), "split": config.day_split(d), "bt_bytes": bt_bytes,
           "at_bytes": at_bytes, "n_quotes_raw": n_q, "n_trades_raw": n_t_raw, "n_trades_in_day": trades.height,
           **{k: v for k, v in chk.items() if k != "ok"}, "quotes_match_main_manifest": chk_q,
           "trades_match_main_manifest": chk_t, "secs_download": round(secs_dl, 1), "secs_build": round(time.time() - t1, 1),
           "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    new = not _manifest(dev).exists()
    with open(_manifest(dev), "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in MANIFEST_FIELDS})
    return row


def data_days(split: str) -> list[tuple[str, date]]:
    from ofp import features as F

    if split == "dev":
        return list(DEV_DAYS)
    splits = ("val", "test") if split == "all" else (split,)
    return [(s, d) for sp in splits for (s, d) in F.available(sp)]


def prepare(split: str = "all") -> None:
    """Download, reduce and verify the raw files of every (symbol, day) of a split; resumable per day."""
    from ofp import download as D

    dev = split == "dev"
    config.RAW_TMP.mkdir(parents=True, exist_ok=True)
    done = {(r["symbol"], r["day"]) for r in manifest_rows(dev)}
    todo = [(s, d) for s, d in data_days(split)
            if (s, d.isoformat()) not in done or not grid_path(s, d, dev).exists() or not trades_path(s, d, dev).exists()]
    log.info("maker_data %s: %d symbol-days to process", split, len(todo))
    t0, gb = time.time(), 0.0
    pool = ThreadPoolExecutor(max_workers=1)  # prefetch the next day's zips while this one is parsed
    nxt = pool.submit(D.fetch_day, *todo[0]) if todo else None
    for i, (s, d) in enumerate(todo):
        fut = nxt
        nxt = pool.submit(D.fetch_day, *todo[i + 1]) if i + 1 < len(todo) else None
        row = process_day(s, d, fut.result(), dev=dev)
        gb += (row["bt_bytes"] + row["at_bytes"]) / 1e9
        log.info("%s %s [%d/%d] quotes %d trades %d verified; dl %.0fs build %.0fs; %.2f GB so far, %.0f min", s, d, i + 1,
                 len(todo), row["n_quotes_raw"], row["n_trades_in_day"], row["secs_download"], row["secs_build"], gb,
                 (time.time() - t0) / 60)
    pool.shutdown(wait=True)
    log.info("maker_data %s done in %.0f min", split, (time.time() - t0) / 60)


# ================================================================================================= fills and costs
def fill_delays(t_post, p, side, q0, size, tr_ts, tr_px, tr_qty, tr_ibm, horizon_ms: int) -> dict:
    """Per order: delay (ms after t_post) of the passive fill under each rule within (t_post, t_post + horizon_ms]
    (np.inf if none). Trades must be sorted by ts. Only trades strictly after t_post are seen."""
    n = len(t_post)
    out = {r: np.full(n, np.inf) for r in RULES}
    lo = np.searchsorted(tr_ts, t_post, side="right")
    hi = np.searchsorted(tr_ts, t_post + horizon_ms, side="right")
    for i in range(n):
        a, b = lo[i], hi[i]
        if a >= b or not np.isfinite(p[i]):
            continue
        px = tr_px[a:b]
        if side[i] > 0:
            through = px < p[i]
            opp = tr_ibm[a:b] & (px <= p[i])  # seller-initiated, at or through our bid
        else:
            through = px > p[i]
            opp = ~tr_ibm[a:b] & (px >= p[i])  # buyer-initiated, at or through our ask
        j = int(np.argmax(through))
        if through[j]:
            out["through"][i] = tr_ts[a + j] - t_post[i]
        need = q0[i] + size[i]
        if np.isfinite(need):
            cum = np.cumsum(np.where(opp, tr_qty[a:b], 0.0))
            j = int(np.searchsorted(cum, need, side="right"))  # first trade with cumulative volume > need
            if j < len(cum):
                out["queue"][i] = tr_ts[a + j] - t_post[i]
    return out


def mid_after(grid: dict, day_start: int, t_abs: np.ndarray) -> np.ndarray:
    """Mid of the quote state at the first 100 ms grid point strictly after t_abs."""
    j = (t_abs - day_start) // GRID_MS + 1
    j = np.clip(j, 0, len(grid["bid"]) - 1).astype(np.int64)
    return (grid["bid"][j] + grid["ask"][j]) / 2


def segment_orders(k0, side, bid, ask, bid100, ask100, grid: dict, trades: dict, day_start: int) -> dict:
    """Per-order quantities of one (symbol, day) segment. k0: start buckets (bars index), side: +1/-1 per order.
    bid..ask100: the segment's 1-s bar quotes. grid: 100 ms quote states. trades: ts, price, qty, is_buyer_maker arrays."""
    k0 = np.asarray(k0, dtype=np.int64)
    side = np.asarray(side, dtype=np.float64)
    buy = side > 0
    m0 = (bid[k0] + ask[k0]) / 2
    od = {"k0": k0, "side": side, "m0": m0,
          "p_post": np.where(buy, bid100[k0], ask100[k0])}  # near touch when the order arrives (close + 100 ms)
    for T in (0,) + WAITS:  # taker price at close(k0 + T) + 100 ms; T = 0 is taker at once
        od[f"p_cross_{T}"] = np.where(buy, ask100[k0 + T], bid100[k0 + T])
    jpost = 10 * (k0 + 1) + 1
    od["q0"] = np.where(buy, grid["bidq"][jpost], grid["askq"][jpost])
    od["grid_matches_bars"] = bool(np.array_equal(np.where(buy, grid["bid"][jpost], grid["ask"][jpost]), od["p_post"],
                                                  equal_nan=True))
    od["size"] = ORDER_NOTIONAL_USDT / m0
    t_post = day_start + (k0 + 1) * 1000 + LATENCY_MS
    d = fill_delays(t_post, od["p_post"], side, od["q0"], od["size"], trades["ts"], trades["price"], trades["qty"],
                    trades["is_buyer_maker"], max(WAITS) * 1000)
    for r in RULES:
        od[f"delay_{r}"] = d[r]
        tf = t_post + np.where(np.isfinite(d[r]), d[r], 0).astype(np.int64)
        f = np.isfinite(d[r])
        mf = mid_after(grid, day_start, tf)
        for tau in MARKOUTS:
            mt = mid_after(grid, day_start, tf + tau * 1000)
            od[f"adv{tau}_{r}"] = np.where(f, -side * (mt - mf) / m0 * 1e4, np.nan)
            od[f"mk{tau}_{r}"] = np.where(f, side * (mt - od["p_post"]) / m0 * 1e4, np.nan)
    return od


def cost_bp(price, fee, m0, side):
    """Execution cost in bp of the arrival mid, fee included (positive = cost)."""
    return side * (price - m0) / m0 * 1e4 + fee * price / m0 * 1e4


def decide_cross(forecast_at_start: np.ndarray, side: np.ndarray, theta: float) -> np.ndarray:
    """Cross now when the forecast says the price moves away by more than theta bp. Missing forecast: post."""
    f = np.where(np.isfinite(forecast_at_start), forecast_at_start, 0.0)
    return side * f > theta


def policy(od: dict, rule: str, T: int, cross: np.ndarray, fee_taker: float = FEE_TAKER, fee_maker: float = FEE_MAKER):
    """Cost per order of: cross now where `cross`, else post at the near touch for T s, then cross if unfilled."""
    posted = ~cross
    filled = posted & (od[f"delay_{rule}"] <= T * 1000) if T > 0 else np.zeros_like(posted)
    price = np.where(cross, od["p_cross_0"], np.where(filled, od["p_post"], od[f"p_cross_{T}"]))
    fee = np.where(filled, fee_maker, fee_taker)
    return cost_bp(price, fee, od["m0"], od["side"]), posted, filled


def taker_now_cost(od: dict, fee_taker: float = FEE_TAKER) -> np.ndarray:
    return cost_bp(od["p_cross_0"], fee_taker, od["m0"], od["side"])


# ================================================================================================= panel-level
def build_orders(book: "O.Book", dev: bool = False) -> dict:
    """Per-order table over all segments of an overlay Book, in the overlay's layout (all buys, then all sells)."""
    t0 = time.time()
    starts = book.starts
    n = len(starts)
    st = np.concatenate([starts, starts])
    side = np.concatenate([np.ones(n), -np.ones(n)])
    seg = st // SPD
    cols = {}
    for s_i in np.unique(seg):
        symbol, d = book.keys[s_i]
        m = seg == s_i
        lo = s_i * SPD
        sl = slice(lo, lo + SPD)
        grid = {c: v.to_numpy() for c, v in pl.read_parquet(grid_path(symbol, d, dev)).to_dict().items()}
        tr = pl.read_parquet(trades_path(symbol, d, dev))
        trades = {"ts": tr["ts"].to_numpy(), "price": tr["price"].to_numpy(), "qty": tr["qty"].to_numpy(),
                  "is_buyer_maker": tr["is_buyer_maker"].to_numpy()}
        od = segment_orders(st[m] - lo, side[m], book.bid[sl], book.ask[sl], book.bid100[sl], book.ask100[sl], grid, trades,
                            day_start_ms(d))
        if not od.pop("grid_matches_bars"):
            raise RuntimeError(f"100 ms grid does not match the bars' bid100/ask100 for {symbol} {d}")
        for c, v in od.items():
            if c not in cols:
                cols[c] = np.full(2 * n, np.nan)
            cols[c][m] = v
    cols["start"] = st
    cols["side"] = side
    cols["seg"] = seg
    cols["day"] = book.seg_day[seg]
    cols["symbol"] = np.array([s for s, _ in book.keys])[seg]
    locked = int(np.sum(side * (cols["p_cross_0"] - cols["p_post"]) <= 0))  # near touch not better than far touch
    log.info("built %d orders over %d segments (%.0fs); book locked/crossed at posting for %d orders", 2 * n,
             len(np.unique(seg)), time.time() - t0, locked)
    return cols


def model_forecast(book: "O.Book", split: str, name: str, h: int, od: dict) -> np.ndarray:
    p = book.preds(split, name)
    return np.asarray(p[od["start"], H.index(h)], dtype=np.float64)


def overlay_cost(book: "O.Book", split: str, name: str, h: int, W: int, od: dict, fee_taker: float = FEE_TAKER) -> np.ndarray:
    """The existing overlay timing for this model (taker fill at its chosen second) plus the taker fee, same order layout."""
    res = O.simulate_overlay(O._signal(book.preds(split, name), h), book.valid, book.bid, book.ask, book.bid100, book.ask100,
                             book.starts, W)
    assert np.array_equal(res["start"], od["start"]) and np.array_equal(res["side"], od["side"])
    px = np.where(res["side"] > 0, book.ask100[res["exec_row"]], book.bid100[res["exec_row"]])
    return res["cost_rule"] + fee_taker * px / od["m0"] * 1e4


class Boot:
    """The overlay's daily block bootstrap (same seed, same number of draws)."""

    def __init__(self, day: np.ndarray, n_days: int, seed: int = config.SEED):
        rng = np.random.default_rng(seed)
        self.day, self.n_days = day, n_days
        self.cnt = np.bincount(day, minlength=n_days).astype(float)
        self.draws = rng.integers(0, n_days, size=(O.N_BOOT, n_days))

    def __call__(self, x: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float, float, float]:
        day = self.day if mask is None else self.day[mask]
        x = x if mask is None else x[mask]
        cnt = np.bincount(day, minlength=self.n_days).astype(float)
        s = np.bincount(day, weights=x, minlength=self.n_days)
        boot = s[self.draws].sum(1) / np.maximum(cnt[self.draws].sum(1), 1e-300)
        dm = np.divide(s, cnt, out=np.zeros_like(s), where=cnt > 0)
        return (float(x.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)),
                100 * float(np.mean(dm[cnt > 0] > 0)))


def score(od: dict, cost, posted, filled, rule: str, T: int, boot: Boot, benchmarks: dict, mask=None) -> dict:
    """Summary of one policy: savings vs each benchmark cost (CI, days > 0), fill rate, adverse selection."""
    m = np.ones(len(cost), bool) if mask is None else mask
    out = {"orders": int(m.sum()), "cost_bp": float(cost[m].mean()), "posted_pct": 100 * float(posted[m].mean()),
           "fill_rate_pct": 100 * float(filled[m].sum() / max(posted[m].sum(), 1)),
           "filled_pct_of_orders": 100 * float(filled[m].mean())}
    for name, bc in benchmarks.items():
        mean, lo, hi, pos = boot(bc - cost, None if mask is None else m)
        out.update({f"save_vs_{name}_bp": mean, f"save_vs_{name}_ci_lo": lo, f"save_vs_{name}_ci_hi": hi,
                    f"pos_days_vs_{name}_pct": pos, f"cost_{name}_bp": float(bc[m].mean())})
    fm, cm = filled & m, posted & ~filled & m
    out["cost_when_filled_bp"] = float(cost[fm].mean()) if fm.any() else np.nan
    out["cost_when_chased_bp"] = float(cost[cm].mean()) if cm.any() else np.nan
    out["cost_when_crossed_now_bp"] = float(cost[~posted & m].mean()) if (~posted & m).any() else np.nan
    wait = np.where(filled, od[f"delay_{rule}"], T * 1000.0) / 1000
    out["mean_wait_posted_s"] = float(wait[posted & m].mean()) if (posted & m).any() else 0.0
    for tau in MARKOUTS:
        out[f"adv_move_{tau}s_bp"] = float(np.mean(od[f"adv{tau}_{rule}"][fm])) if fm.any() else np.nan
        out[f"markout_{tau}s_bp"] = float(np.mean(od[f"mk{tau}_{rule}"][fm])) if fm.any() else np.nan
    return out


def _grid(name: str):
    if name == "sign_ofi":
        return [(1, 0.0)]
    return [(h, th) for h in FORECAST_H for th in THETAS]


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _pick(g: pd.DataFrame) -> pd.Series:
    return g.sort_values(["save_vs_taker_now_bp", "T", "theta", "h"], ascending=[False, True, False, True]).iloc[0]


# ================================================================================================= stages
def validate() -> None:
    """Score the 736-configuration grid on validation; choose per (model, rule); freeze."""
    t0 = time.time()
    book = O.Book("val")
    od = build_orders(book)
    boot = Boot(od["day"], len(book.days))
    base = {"taker_now": taker_now_cost(od)}
    rows = []
    for rule in RULES:
        for T in WAITS:
            c, p, f = policy(od, rule, T, np.zeros(len(od["side"]), bool))
            rows.append({"rule": rule, "model": "always_post", "h": 0, "theta": np.inf, "T": T,
                         **score(od, c, p, f, rule, T, boot, base)})
    for name in config.MODELS_LADDER:
        for h, th in _grid(name):
            cross = decide_cross(model_forecast(book, "val", name, h, od), od["side"], th)
            for rule in RULES:
                for T in WAITS:
                    c, p, f = policy(od, rule, T, cross)
                    rows.append({"rule": rule, "model": name, "h": h, "theta": th, "T": T,
                                 **score(od, c, p, f, rule, T, boot, base)})
        log.info("val %s done", name)
    g = pd.DataFrame(rows)
    g.to_csv(config.RESULTS / "maker_validation.csv", index=False)
    choices = []
    for (rule, name), gm in g.groupby(["rule", "model"], sort=False):
        b = _pick(gm)
        choices.append({"rule": rule, "model": name, "h": int(b["h"]), "theta": float(b["theta"]), "T": int(b["T"]),
                        "val_save_vs_taker_now_bp": float(b["save_vs_taker_now_bp"]),
                        "val_fill_rate_pct": float(b["fill_rate_pct"]), "val_posted_pct": float(b["posted_pct"])})
    frozen = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "rule": "per (model, fill rule): (h, theta, T) with the highest mean validation saving vs taker at once "
                      "(ties: smaller T, larger theta, smaller h); always_post: T the same way",
              "design": design(), "n_configs": int(len(g)), "n_val_orders": int(len(od["side"])),
              "overlay_frozen_hash": json.loads((config.RESULTS / "overlay_frozen.json").read_text())["hash"],
              "choices": choices}
    frozen["hash"] = _hash(frozen)
    (config.RESULTS / "maker_frozen.json").write_text(json.dumps(frozen, indent=1))
    log.info("maker validation done in %.0fs; frozen %s", time.time() - t0, frozen["hash"])
    for c in choices:
        log.info("choice %s", c)


def reproduce_overlay(book: "O.Book") -> dict:
    """Recompute the overlay's primary test rows with the unchanged overlay code; compare with overlay_test.csv."""
    ref = pd.read_csv(config.RESULTS / "overlay_test.csv").set_index("model")
    frozen = json.loads((config.RESULTS / "overlay_frozen.json").read_text())
    diffs = {}
    for c in frozen["choices"]:
        s = book.run(O._signal(book.preds("test", c["model"]), c["h"]), c["W"])
        r = ref.loc[c["model"]]
        diffs[c["model"]] = max(abs(float(s[k]) - float(r[k])) for k in s)
    return {"overlay_reproduced": bool(all(v <= 1e-9 for v in diffs.values())), "overlay_max_abs_diff": diffs}


def _choice(frozen: dict, rule: str, model: str) -> dict:
    return next(c for c in frozen["choices"] if c["rule"] == rule and c["model"] == model)


def test(force: bool = False) -> None:
    from ofp.pipeline import code_hash

    log_path = config.RESULTS / "maker_test_log.jsonl"
    if log_path.exists() and any(json.loads(x).get("event") == "done" for x in log_path.read_text().splitlines() if x.strip()):
        if not force:
            raise SystemExit("maker test already run once (results/maker_test_log.jsonl); --force reruns it (logged)")
    frozen = json.loads((config.RESULTS / "maker_frozen.json").read_text())
    saved = frozen.pop("hash")
    if _hash(frozen) != saved:
        raise SystemExit("maker frozen config hash mismatch")
    ov = json.loads((config.RESULTS / "overlay_frozen.json").read_text())
    t0 = time.time()
    base_ev = {"frozen_hash": saved, "code_hash": code_hash(), "forced_rerun": bool(force)}

    def _log(ev):
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({**ev, "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")

    _log({"event": "start", **base_ev})
    book = O.Book("test")
    repro = reproduce_overlay(book)
    log.info("overlay reproduction: %s", repro)
    od = build_orders(book)
    rows, parts, by_t, fee_alt = score_frozen(book, "test", od, frozen, ov)
    pd.DataFrame(rows).to_csv(config.RESULTS / "maker_test.csv", index=False)
    pd.DataFrame(parts).to_csv(config.RESULTS / "maker_test_breakdown.csv", index=False)
    pd.DataFrame(by_t).to_csv(config.RESULTS / "maker_test_by_T.csv", index=False)
    pd.DataFrame(fee_alt).to_csv(config.RESULTS / "maker_test_fee4bp.csv", index=False)
    _log({"event": "done", **base_ev, **repro, "n_days": len(book.days), "orders": int(len(od["side"])),
          "seconds": round(time.time() - t0)})
    log.info("maker test done in %.0fs", time.time() - t0)


def score_frozen(book: "O.Book", split: str, od: dict, frozen: dict, ov: dict):
    """Score the frozen choices on one split: primary rows, by-symbol/side parts, every T, and the 4 bp taker variant."""
    n = len(od["side"])
    boot = Boot(od["day"], len(book.days))
    rows, parts, by_t, fee_alt = [], [], [], []
    for fee_t, sink in ((FEE_TAKER, rows), (FEE_TAKER_ALT, fee_alt)):
        for rule in RULES:
            ap = _choice(frozen, rule, "always_post")
            ap_cost = policy(od, rule, ap["T"], np.zeros(n, bool), fee_taker=fee_t)[0]
            for model in ("always_post",) + TEST_MODELS:
                ch = _choice(frozen, rule, model)
                cross = (np.zeros(n, bool) if model == "always_post"
                         else decide_cross(model_forecast(book, split, model, ch["h"], od), od["side"], ch["theta"]))
                bench = {"taker_now": taker_now_cost(od, fee_t)}
                if model != "always_post":
                    oc = next(c for c in ov["choices"] if c["model"] == model)
                    bench["always_post"] = ap_cost
                    bench["overlay"] = overlay_cost(book, split, model, oc["h"], oc["W"], od, fee_t)
                c, p, f = policy(od, rule, ch["T"], cross, fee_taker=fee_t)
                head = {"rule": rule, "model": model, "h": ch["h"], "theta": ch["theta"], "T": ch["T"],
                        "fee_taker_bp": fee_t * 1e4, "val_save_vs_taker_now_bp": ch["val_save_vs_taker_now_bp"]}
                sink.append({**head, **score(od, c, p, f, rule, ch["T"], boot, bench)})
                if fee_t != FEE_TAKER:
                    continue
                for part, m in ([(f"symbol {s}", od["symbol"] == s) for s in config.SYMBOLS]
                                + [("buys", od["side"] > 0), ("sells", od["side"] < 0)]):
                    parts.append({**head, "part": part, **score(od, c, p, f, rule, ch["T"], boot, bench, mask=m)})
                for T in WAITS:
                    cT, pT, fT = policy(od, rule, T, cross)
                    by_t.append({**head, "T": T, "primary": T == ch["T"],
                                 **score(od, cT, pT, fT, rule, T, boot, {"taker_now": bench["taker_now"]})})
        log.info("%s scoring at taker fee %.1f bp done", split, fee_t * 1e4)
    return rows, parts, by_t, fee_alt


# ================================================================================================= report
RULE_LABEL = {"through": "(a) trade-through", "queue": "(b) queue"}


def report() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ofp.report import LABEL, md_table

    lab = {**LABEL, "always_post": "always post (no signal)"}
    R = config.RESULTS
    v = pd.read_csv(R / "maker_validation.csv")
    t = pd.read_csv(R / "maker_test.csv")
    bd = pd.read_csv(R / "maker_test_breakdown.csv")
    bt = pd.read_csv(R / "maker_test_by_T.csv")
    fa = pd.read_csv(R / "maker_test_fee4bp.csv")
    frozen = json.loads((R / "maker_frozen.json").read_text())
    tl = [json.loads(x) for x in (R / "maker_test_log.jsonl").read_text().splitlines() if x.strip()]

    def ci(r, b):
        if f"save_vs_{b}_bp" not in r or not np.isfinite(r[f"save_vs_{b}_bp"]):
            return "–"
        return f"{r[f'save_vs_{b}_bp']:+.3f} [{r[f'save_vs_{b}_ci_lo']:+.3f}, {r[f'save_vs_{b}_ci_hi']:+.3f}]"

    def theta(x):
        return "∞" if not np.isfinite(x) else f"{x:g}"

    def main_rows(df, extra=True):
        recs = []
        for _, r in df.iterrows():
            rec = {"rule": RULE_LABEL[r["rule"]], "model": lab[r["model"]],
                   "h / θ / T": f"{'–' if r['model'] in ('always_post', 'sign_ofi') else int(r['h'])} / "
                                f"{theta(r['theta'])} / {int(r['T'])} s",
                   "saving vs taker now bp [95% CI]": ci(r, "taker_now"), "days > 0 %": r["pos_days_vs_taker_now_pct"]}
            if extra:
                rec["vs always-post"] = ci(r, "always_post")
                rec["vs overlay timing"] = ci(r, "overlay")
            rec.update({"cost bp (taker now)": f"{r['cost_bp']:.3f} ({r['cost_taker_now_bp']:.3f})",
                        "posted %": r["posted_pct"], "fill rate %": r["fill_rate_pct"],
                        "adverse move 1 s / 10 s bp": f"{r['adv_move_1s_bp']:+.2f} / {r['adv_move_10s_bp']:+.2f}",
                        "markout 1 s / 10 s bp": f"{r['markout_1s_bp']:+.2f} / {r['markout_10s_bp']:+.2f}",
                        "cost filled / chased bp": f"{r['cost_when_filled_bp']:.2f} / {r['cost_when_chased_bp']:.2f}"})
            recs.append(rec)
        return pd.DataFrame(recs)

    fm = {"days > 0 %": 0, "posted %": 1, "fill rate %": 1}
    out = ["# Maker or taker per order (Change 16; frozen models, no retraining)\n",
           "Cost = execution price vs the arrival mid plus fees, bp of the arrival mid. Saving = benchmark cost − policy cost "
           "(positive = the policy is cheaper). Fees: taker 5 bp, maker 2 bp (VIP0). Every action takes effect 100 ms after "
           "its decision. Orders: one buy and one sell every 60 s per symbol-day (the overlay's orders), 10,000 USDT each. "
           "CI: 95 % daily block bootstrap (5,000 resamples of days). Adverse move = mid move against the order after a "
           "passive fill; markout = mid after τ vs the fill price, signed so positive = good for the order.\n",
           f"\n## Test (frozen {frozen['hash']}): validation-chosen (h, θ, T) per model and fill rule\n",
           md_table(main_rows(t), fm),
           "\n## Test by symbol and side (descriptive)\n"]
    recs = [{"rule": RULE_LABEL[r["rule"]], "model": lab[r["model"]], "part": r["part"], "orders": int(r["orders"]),
             "saving vs taker now bp [95% CI]": ci(r, "taker_now"), "days > 0 %": r["pos_days_vs_taker_now_pct"],
             "vs overlay timing": ci(r, "overlay"), "fill rate %": r["fill_rate_pct"],
             "adverse move 10 s bp": r["adv_move_10s_bp"]} for _, r in bd.iterrows()]
    out.append(md_table(pd.DataFrame(recs), {"days > 0 %": 0, "fill rate %": 1, "adverse move 10 s bp": 2}))
    out.append("\n## Test, every wait T at the chosen (h, θ) (descriptive; same single run)\n")
    out.append(md_table(main_rows(bt, extra=False), fm))
    out.append("\n## Fee sensitivity: taker 4 bp / maker 2 bp (the rate published until at least 2023-09-27), same frozen "
               "choices (descriptive)\n")
    out.append(md_table(main_rows(fa), fm))
    out.append("\n## Validation choices\n")
    out.append(md_table(pd.DataFrame([{"rule": RULE_LABEL[c["rule"]], "model": lab[c["model"]], "h": c["h"],
                                       "θ": theta(c["theta"]), "T": c["T"],
                                       "val saving vs taker now bp": c["val_save_vs_taker_now_bp"],
                                       "val fill rate %": c["val_fill_rate_pct"], "val posted %": c["val_posted_pct"]}
                                      for c in frozen["choices"]]),
                        {"h": 0, "T": 0, "val saving vs taker now bp": 3, "val fill rate %": 1, "val posted %": 1}))
    out.append(f"\n## Validation grid: best per (rule, model, T) ({len(v)} configurations in `maker_validation.csv`)\n")
    vb = v.loc[v.groupby(["rule", "model", "T"], sort=False)["save_vs_taker_now_bp"].idxmax()]
    out.append(md_table(main_rows(vb, extra=False), fm))
    out.append("\n## Maker test log\n")
    for e in tl:
        out.append(f"- {e['event']} {e['timestamp_utc']} frozen {e['frozen_hash']} code {e['code_hash']} "
                   f"forced={e['forced_rerun']}" + (f" overlay_reproduced={e['overlay_reproduced']}" if "overlay_reproduced" in e else ""))
    (R / "maker_tables.md").write_text("\n".join(out) + "\n", encoding="utf-8")

    # figure: saving vs taker at once by wait T (test), per rule; marker = validation choice; horizontal = overlay timing
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    colors = {"always_post": "#7f7f7f", "sign_ofi": "#1f77b4", "ridge": "#2ca02c", "mlp": "#9467bd"}
    for ax, rule in zip(axes, RULES):
        g = bt[bt["rule"] == rule]
        for i, (m, gm) in enumerate(g.groupby("model", sort=False)):
            gm = gm.sort_values("T")
            x = np.arange(len(gm)) + (i - 1.5) * 0.1
            y = gm["save_vs_taker_now_bp"].to_numpy()
            err = np.vstack([y - gm["save_vs_taker_now_ci_lo"].to_numpy(), gm["save_vs_taker_now_ci_hi"].to_numpy() - y])
            ax.errorbar(x, y, yerr=err, fmt="o-", color=colors[m], capsize=3, label=lab[m])
            prim = gm["primary"].to_numpy()
            ax.plot(x[prim], y[prim], "s", ms=11, mfc="none", color=colors[m])
        tr = t[(t["rule"] == rule) & t["model"].isin(TEST_MODELS)]
        for _, r in tr.iterrows():
            ax.axhline(r["cost_taker_now_bp"] - r["cost_overlay_bp"], color=colors[r["model"]], ls=":", lw=1)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(WAITS)), [f"T={x}s" for x in WAITS])
        ax.set_title(f"fill rule {RULE_LABEL[rule]}", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("bp saved per order vs taker at once (test, fees incl.)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Post at the touch, then cross: test 2024-01-11 to 2024-03-30, BTC/ETH/SOL perps, VIP0 fees (square = "
                 "validation-chosen T; dotted = overlay timing; bars = 95% daily block bootstrap)", fontsize=9)
    fig.tight_layout()
    fig.savefig(R / "fig_maker.png", dpi=130)
    plt.close(fig)
    log.info("maker report written")
