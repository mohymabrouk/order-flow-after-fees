"""Test 4: fee and P&L accounting on a toy trade list."""

import numpy as np
import pandas as pd
import pytest

from ofp import sim


def test_fee_and_pnl_on_toy_trades():
    n = 10
    bid = np.array([100.0, 101, 102, 103, 104, 105, 104, 103, 102, 101])
    ask = bid + 1.0
    book = sim.Book(bid, ask, bid, ask, np.full(n, np.nan), np.full(n, np.nan), seg_len=n)
    signal = np.zeros(n)
    signal[0] = 1.0   # long at ask[0]=101, exit at bid[2]=102
    signal[5] = -1.0  # short at bid[5]=105, exit at ask[7]=104
    taker, maker = 0.0005, 0.0002
    out = sim.simulate(signal, np.ones(n, bool), book, 2, 0, "taker", taker, maker, 0.0)
    assert out["idx"].tolist() == [0, 5]
    r1 = (102 - 101) / 101 - taker - taker * 102 / 101
    r2 = -(104 - 105) / 105 - taker - taker * 104 / 105
    np.testing.assert_allclose(out["net_bps"], [r1 * 1e4, r2 * 1e4])
    np.testing.assert_allclose(out["fee_bps"], [(taker + taker * 102 / 101) * 1e4, (taker + taker * 104 / 105) * 1e4])
    # gross (mid to mid) - spread cost = executed return before fees
    np.testing.assert_allclose(out["gross_bps"] - out["spread_bps"], [(102 - 101) / 101 * 1e4, (105 - 104) / 105 * 1e4])
    np.testing.assert_allclose(out["gross_bps"], [(102.5 - 100.5) / 100.5 * 1e4, (105.5 - 103.5) / 105.5 * 1e4])
    # daily P&L and summary
    days = ["d0"]
    s = sim.summarize(out, np.array([0]), days, n_segments=1)
    assert s["trades"] == 2
    assert s["total_net_bps"] == pytest.approx((r1 + r2) * 1e4)
    assert s["hit_rate"] == pytest.approx(np.mean(np.array([r1, r2]) > 0))
    assert s["turnover"] == 4
    pnl = sim.daily_pnl(out, np.array([0]), days)
    assert isinstance(pnl, pd.Series) and pnl.iloc[0] == pytest.approx((r1 + r2) * 1e4)


def test_maker_fee_on_entry_only():
    n = 6
    bid = np.full(n, 100.0)
    ask = np.full(n, 100.5)
    tmin = np.full(n, 99.0)
    book = sim.Book(bid, ask, bid, ask, tmin, np.full(n, 101.0), seg_len=n)
    signal = np.zeros(n)
    signal[0] = 1.0
    out = sim.simulate(signal, np.ones(n, bool), book, 2, 0, "maker", 0.0005, 0.0002, 0.0)
    # bought at the bid 100 (maker 2 bp), sold at the bid 100 (taker 5 bp): net = -7 bp, spread earned = 0.25/100.25
    np.testing.assert_allclose(out["net_bps"], [-(0.0002 + 0.0005) * 1e4])
    np.testing.assert_allclose(out["fee_bps"], [7.0])
