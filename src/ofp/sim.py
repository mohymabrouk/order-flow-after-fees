"""Trading simulator.

Panel layout: (symbol, day) segments of `seg_len` 1-second rows concatenated; row k of a segment is bucket k of that day.
A signal is formed at the end of bucket k. With latency L = s seconds + r ms (r in {0, 100}), execution happens at the end of
bucket k+s (+100 ms if r) at the then-prevailing best quote; the position is closed h seconds later, as a taker, at the then
prevailing quote. One position per symbol at a time.

Taker: buy at the ask, sell at the bid. Maker (optimistic): post at the touch on our side at the entry time, filled at that
price if a trade prints strictly through it within the next h buckets; exit as taker at the same exit time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

from ofp import config
from ofp.metrics import deflated_sharpe, per_period_sharpe, sharpe


@dataclass
class Book:
    bid: np.ndarray
    ask: np.ndarray
    bid100: np.ndarray
    ask100: np.ndarray
    tmin: np.ndarray  # NaN when no trade in the bucket
    tmax: np.ndarray
    seg_len: int = config.SECONDS_PER_DAY

    def __post_init__(self):
        self._rmin, self._rmax = {}, {}

    def window_min(self, h: int) -> np.ndarray:
        """window_min(h)[i] = min trade price over buckets i-h+1 .. i (NaN if none)."""
        if h not in self._rmin:
            s = pl.Series(self.tmin).fill_nan(None)
            self._rmin[h] = s.rolling_min(h, min_samples=1).fill_null(np.nan).to_numpy()
        return self._rmin[h]

    def window_max(self, h: int) -> np.ndarray:
        if h not in self._rmax:
            s = pl.Series(self.tmax).fill_nan(None)
            self._rmax[h] = s.rolling_max(h, min_samples=1).fill_null(np.nan).to_numpy()
        return self._rmax[h]


def select_entries(cand: np.ndarray, hold: int) -> np.ndarray:
    """Greedy non-overlapping entries: take a candidate only if the previous position is closed."""
    out = []
    nxt = -1
    for c in cand.tolist():
        if c >= nxt:
            out.append(c)
            nxt = c + hold
    return np.asarray(out, dtype=np.int64)


def simulate(signal: np.ndarray, tradable: np.ndarray, book: Book, h: int, latency_ms: int, mode: str,
             fee_taker: float, fee_maker: float, threshold) -> dict:
    """Return a dict of per-trade arrays. `threshold` is a scalar or a per-row array (bps of forecast)."""
    s, r = divmod(int(latency_ms), 1000)
    assert r in (0, 100), "latency must be whole seconds, optionally + 100 ms"
    n = len(signal)
    k = np.arange(n) % book.seg_len
    hold = h + s + (1 if r else 0)
    ok = tradable & (signal != 0) & (np.abs(signal) > threshold) & (k + s + h < book.seg_len)
    sel = select_entries(np.flatnonzero(ok), hold)
    d = np.sign(signal[sel]).astype(np.float64)
    e, x = sel + s, sel + s + h
    qb, qa = (book.bid100, book.ask100) if r else (book.bid, book.ask)
    eb, ea, xb, xa = qb[e], qa[e], qb[x], qa[x]
    mid_e, mid_x = (eb + ea) / 2, (xb + xa) / 2
    exit_px = np.where(d > 0, xb, xa)  # taker exit: sell at the bid, buy back at the ask
    if mode == "taker":
        entry_px = np.where(d > 0, ea, eb)
        filled = np.ones(len(sel), dtype=bool)
        fee_in = fee_taker
    elif mode == "maker":
        entry_px = np.where(d > 0, eb, ea)
        last = e + h  # fill window: buckets e+1 .. e+h
        wmin, wmax = book.window_min(h)[last], book.window_max(h)[last]
        filled = np.where(d > 0, wmin < entry_px, wmax > entry_px)  # NaN compares False -> not filled
        fee_in = fee_maker
    else:
        raise ValueError(mode)
    sel, d, entry_px, exit_px, mid_e, mid_x = sel[filled], d[filled], entry_px[filled], exit_px[filled], mid_e[filled], mid_x[filled]
    ret_px = d * (exit_px - entry_px) / entry_px
    fees = fee_in + fee_taker * exit_px / entry_px
    gross = d * (mid_x - mid_e) / mid_e * 1e4
    return {
        "idx": sel,
        "dir": d,
        "entry_px": entry_px,
        "exit_px": exit_px,
        "gross_bps": gross,
        "spread_bps": gross - ret_px * 1e4,
        "fee_bps": fees * 1e4,
        "net_bps": (ret_px - fees) * 1e4,
        "n_orders": int(len(filled)),
    }


def daily_pnl(trades: dict, seg_day: np.ndarray, days: list, seg_len: int = config.SECONDS_PER_DAY) -> pd.Series:
    """Daily net P&L (bps of unit notional, summed over symbols), on every day of the split (0 if no trade)."""
    seg = trades["idx"] // seg_len
    day = seg_day[seg]
    s = pd.Series(trades["net_bps"]).groupby(day).sum() if len(seg) else pd.Series(dtype=float)
    return s.reindex(range(len(days)), fill_value=0.0).set_axis(pd.Index(days, name="day"))


def summarize(trades: dict, seg_day: np.ndarray, days: list, n_segments: int, trials: int | None = None,
              trial_var: float | None = None) -> dict:
    n = len(trades["idx"])
    pnl = daily_pnl(trades, seg_day, days)
    out = {
        "trades": n,
        "orders": trades["n_orders"],
        "fill_rate": n / trades["n_orders"] if trades["n_orders"] else np.nan,
        "gross_bps": float(np.mean(trades["gross_bps"])) if n else np.nan,
        "spread_bps": float(np.mean(trades["spread_bps"])) if n else np.nan,
        "fee_bps": float(np.mean(trades["fee_bps"])) if n else np.nan,
        "net_bps": float(np.mean(trades["net_bps"])) if n else np.nan,
        "hit_rate": float(np.mean(trades["net_bps"] > 0)) if n else np.nan,
        "total_net_bps": float(np.sum(trades["net_bps"])),
        "daily_mean_bps": float(pnl.mean()),
        "daily_sd_bps": float(pnl.std(ddof=1)),
        "sharpe_ann": sharpe(pnl, config.DAYS_PER_YEAR),
        "sharpe_pp": per_period_sharpe(pnl),
        "turnover": 2 * n / max(n_segments, 1),  # notional traded per symbol-day, per unit of notional capital
        "days": len(days),
    }
    if trials is not None:
        out["dsr"] = deflated_sharpe(pnl, trials, trial_var or 0.0)
    return out
