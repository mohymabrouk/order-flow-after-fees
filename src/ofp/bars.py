"""Raw Binance bookTicker + aggTrades for one day -> 1-second bars.

Bucket k holds events with day-relative time in [k*1000, (k+1)*1000) ms. Quote states are "the last update strictly before"
the bucket close (and before close + 100 ms, for the latency-100ms fills).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ofp import config

QUOTE_COLS = ["update_id", "best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty", "transaction_time"]
TRADE_COLS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker"]


def read_quotes_csv(data: bytes) -> pl.DataFrame:
    has_header = not data[:1].isdigit()
    df = pl.read_csv(
        data,
        has_header=has_header,
        new_columns=None if has_header else QUOTE_COLS + ["event_time"],
        columns=QUOTE_COLS,
        schema_overrides={"update_id": pl.Int64, "best_bid_price": pl.Float64, "best_bid_qty": pl.Float64,
                          "best_ask_price": pl.Float64, "best_ask_qty": pl.Float64, "transaction_time": pl.Int64},
    )
    return df.rename({"best_bid_price": "bid", "best_bid_qty": "bidq", "best_ask_price": "ask", "best_ask_qty": "askq",
                      "transaction_time": "ts"})


def read_trades_csv(data: bytes) -> pl.DataFrame:
    has_header = not data[:1].isdigit()
    df = pl.read_csv(
        data,
        has_header=has_header,
        new_columns=None if has_header else TRADE_COLS,
        columns=["price", "quantity", "transact_time", "is_buyer_maker"],
        schema_overrides={"price": pl.Float64, "quantity": pl.Float64, "transact_time": pl.Int64},
    )
    if df["is_buyer_maker"].dtype != pl.Boolean:
        df = df.with_columns(pl.col("is_buyer_maker").cast(pl.String).str.to_lowercase().eq("true"))
    return df.rename({"quantity": "qty", "transact_time": "ts"})


def cks_e_expr() -> pl.Expr:
    """Cont-Kukanov-Stoikov (2014) event contribution e_n; the first event of the sequence gets 0."""
    pb, qb, pa, qa = pl.col("bid"), pl.col("bidq"), pl.col("ask"), pl.col("askq")
    pb1, qb1, pa1, qa1 = pb.shift(1), qb.shift(1), pa.shift(1), qa.shift(1)
    e = (
        pl.when(pb >= pb1).then(qb).otherwise(0.0)
        - pl.when(pb <= pb1).then(qb1).otherwise(0.0)
        - pl.when(pa <= pa1).then(qa).otherwise(0.0)
        + pl.when(pa >= pa1).then(qa1).otherwise(0.0)
    )
    return pl.when(pb1.is_null()).then(0.0).otherwise(e)


def cks_e_numpy(bid, bidq, ask, askq) -> np.ndarray:
    """Reference loop implementation of e_n (used by the tests)."""
    n = len(bid)
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = ((bid[i] >= bid[i - 1]) * bidq[i] - (bid[i] <= bid[i - 1]) * bidq[i - 1]
                - (ask[i] <= ask[i - 1]) * askq[i] + (ask[i] >= ask[i - 1]) * askq[i - 1])
    return e


def _asof_state(q: pl.DataFrame, grid_ms: np.ndarray, suffix: str) -> pl.DataFrame:
    """Quote state = last update with ts < grid time (strictly before)."""
    g = pl.DataFrame({"k": np.arange(len(grid_ms), dtype=np.int64), "t": grid_ms.astype(np.int64) - 1})
    s = g.join_asof(q.select("ts", "bid", "ask", "bidq", "askq"), left_on="t", right_on="ts", strategy="backward")
    return s.select(
        "k",
        pl.col("bid").alias(f"bid{suffix}"), pl.col("ask").alias(f"ask{suffix}"),
        pl.col("bidq").alias(f"bidq{suffix}"), pl.col("askq").alias(f"askq{suffix}"),
    )


def make_bars(quotes: pl.DataFrame, trades: pl.DataFrame, day_start_ms: int, n_buckets: int = config.SECONDS_PER_DAY):
    """Return (bars, stats). `quotes`: update_id, bid, bidq, ask, askq, ts. `trades`: price, qty, ts, is_buyer_maker."""
    bms = config.BUCKET_MS
    end_ms = day_start_ms + n_buckets * bms
    stats = {
        "uid_sorted": bool(quotes["update_id"].is_sorted()),
        "ts_sorted": bool(quotes["ts"].is_sorted()),
        "n_quotes_raw": quotes.height,
        "n_trades_raw": trades.height,
        "qty_raw": float(trades["qty"].sum()) if trades.height else 0.0,
    }
    q = quotes.sort("update_id", maintain_order=True).with_columns(cks_e_expr().alias("e"))
    # The CKS sequence runs over the whole file (book sequence); bucket by event time.
    q = q.with_columns(((pl.col("ts") - day_start_ms) // bms).alias("k"))
    in_day = (pl.col("ts") >= day_start_ms) & (pl.col("ts") < end_ms)
    stats["n_quotes_out_of_day"] = int(q.filter(~in_day).height)
    qd = q.filter(in_day)
    agg = qd.group_by("k").agg(
        pl.col("e").sum().alias("ofi"),
        pl.len().cast(pl.Int64).alias("n_upd"),
        ((pl.col("bidq") + pl.col("askq")) / 2).mean().alias("depth_mean"),
    )
    # Quote states by event time; equal timestamps keep the last update in book order.
    qs = q.sort(["ts", "update_id"]).unique(subset="ts", keep="last", maintain_order=True)
    closes = (np.arange(n_buckets, dtype=np.int64) + 1) * bms + day_start_ms
    st0 = _asof_state(qs, closes, "")
    st1 = _asof_state(qs, closes + config.LATENCY_OFFSET_MS, "100").select("k", "bid100", "ask100")

    t = trades.with_columns(((pl.col("ts") - day_start_ms) // bms).alias("k"))
    t_in = (pl.col("ts") >= day_start_ms) & (pl.col("ts") < end_ms)
    stats["n_trades_out_of_day"] = int(t.filter(~t_in).height)
    td = t.filter(t_in)
    buy = ~pl.col("is_buyer_maker")  # buyer is the taker
    tagg = td.group_by("k").agg(
        pl.col("qty").filter(buy).sum().alias("buy_qty"),
        pl.col("qty").filter(~buy).sum().alias("sell_qty"),
        buy.sum().cast(pl.Int64).alias("buy_n"),
        (~buy).sum().cast(pl.Int64).alias("sell_n"),
        pl.col("price").min().alias("tmin"),
        pl.col("price").max().alias("tmax"),
    )
    grid = pl.DataFrame({"k": np.arange(n_buckets, dtype=np.int64)})
    bars = (
        grid.join(agg, on="k", how="left")
        .join(st0, on="k", how="left")
        .join(st1, on="k", how="left")
        .join(tagg, on="k", how="left")
        .sort("k")
        .with_columns(
            pl.col("ofi").fill_null(0.0),
            pl.col("n_upd").fill_null(0),
            pl.col("buy_qty").fill_null(0.0),
            pl.col("sell_qty").fill_null(0.0),
            pl.col("buy_n").fill_null(0),
            pl.col("sell_n").fill_null(0),
        )
        .with_columns(pl.col("depth_mean").fill_null((pl.col("bidq") + pl.col("askq")) / 2))
    )
    stats["n_quotes_in_bars"] = int(bars["n_upd"].sum())
    stats["n_trades_in_bars"] = int(bars["buy_n"].sum() + bars["sell_n"].sum())
    stats["qty_in_bars"] = float(bars["buy_qty"].sum() + bars["sell_qty"].sum())
    stats["qty_in_day"] = float(td["qty"].sum()) if td.height else 0.0
    stats["n_empty_buckets"] = int((bars["n_upd"] == 0).sum())
    return bars, stats
