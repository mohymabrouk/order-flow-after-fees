"""Test 7: bucketing loses no trades: volumes and counts in the bars sum to the raw totals."""

import numpy as np
import polars as pl

from ofp import bars as B


def test_volumes_sum_to_raw(raw):
    quotes, trades, day = raw
    bars, stats = B.make_bars(quotes, trades, day, n_buckets=900)
    raw_buy = trades.filter(~pl.col("is_buyer_maker"))["qty"].sum()
    raw_sell = trades.filter(pl.col("is_buyer_maker"))["qty"].sum()
    np.testing.assert_allclose(bars["buy_qty"].sum(), raw_buy, rtol=1e-12)
    np.testing.assert_allclose(bars["sell_qty"].sum(), raw_sell, rtol=1e-12)
    assert bars["buy_n"].sum() + bars["sell_n"].sum() == trades.height
    assert bars["n_upd"].sum() == quotes.height
    assert stats["n_trades_in_bars"] == stats["n_trades_raw"]


def test_boundary_trades_land_in_the_right_bucket():
    day = 1_700_000_000_000
    quotes = pl.DataFrame({"update_id": [1, 2], "bid": [10.0, 10.0], "bidq": [1.0, 1.0], "ask": [10.1, 10.1],
                           "askq": [1.0, 1.0], "ts": [day, day + 1500]})
    trades = pl.DataFrame({"price": [10.0, 10.1, 10.1, 10.0], "qty": [1.0, 2.0, 4.0, 8.0],
                           "ts": [day + 999, day + 1000, day + 1999, day + 2000], "is_buyer_maker": [True, False, False, True]})
    bars, _ = B.make_bars(quotes, trades, day, n_buckets=3)
    assert bars["sell_qty"].to_list() == [1.0, 0.0, 8.0]
    assert bars["buy_qty"].to_list() == [0.0, 6.0, 0.0]
    assert bars["tmin"].to_list()[1] == 10.1 and bars["tmax"].to_list()[1] == 10.1
    assert bars["buy_qty"].sum() + bars["sell_qty"].sum() == trades["qty"].sum()
