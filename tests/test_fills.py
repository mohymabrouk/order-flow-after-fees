"""Test 3: taker fills buy at the ask and sell at the bid of the quote prevailing after the latency (toy book)."""

import numpy as np

from ofp import sim

N = 20


def toy_book():
    k = np.arange(N, dtype=float)
    bid = 100 + k  # close-of-bucket quotes
    ask = bid + 0.5
    bid100 = 1000 + k  # +100 ms quotes (distinct values so we can see which one was used)
    ask100 = bid100 + 0.5
    tmin = bid - 1.0
    tmax = ask + 1.0
    return sim.Book(bid, ask, bid100, ask100, tmin, tmax, seg_len=N)


def one_signal(at: int, sign: float):
    s = np.zeros(N)
    s[at] = sign
    return s


def test_taker_fill_prices_by_latency():
    b = toy_book()
    valid = np.ones(N, bool)
    h, t = 3, 5
    cases = {0: (b.bid, b.ask, t), 100: (b.bid100, b.ask100, t), 1000: (b.bid, b.ask, t + 1)}
    for lat, (qb, qa, e) in cases.items():
        long = sim.simulate(one_signal(t, +1.0), valid, b, h, lat, "taker", 0.0, 0.0, 0.0)
        assert long["entry_px"][0] == qa[e] and long["exit_px"][0] == qb[e + h], lat
        short = sim.simulate(one_signal(t, -1.0), valid, b, h, lat, "taker", 0.0, 0.0, 0.0)
        assert short["entry_px"][0] == qb[e] and short["exit_px"][0] == qa[e + h], lat
        # zero fees: net return = signed executed price change; gross = signed mid change
        np.testing.assert_allclose(long["net_bps"][0], (qb[e + h] - qa[e]) / qa[e] * 1e4)
        mid_e, mid_x = (qb[e] + qa[e]) / 2, (qb[e + h] + qa[e + h]) / 2
        np.testing.assert_allclose(long["gross_bps"][0], (mid_x - mid_e) / mid_e * 1e4)


def test_no_trade_beyond_day_end_and_no_overlap():
    b = toy_book()
    valid = np.ones(N, bool)
    s = np.ones(N)
    out = sim.simulate(s, valid, b, 4, 100, "taker", 0.0, 0.0, 0.0)
    idx = out["idx"]
    assert np.all(idx + 4 < N)
    assert np.all(np.diff(idx) >= 4 + 1)  # 100 ms latency: next signal only after the exit


def test_maker_fill_needs_trade_through():
    b = toy_book()
    valid = np.ones(N, bool)
    b.tmin[:] = np.nan  # no trades at all -> never filled
    out = sim.simulate(one_signal(5, 1.0), valid, b, 3, 0, "maker", 0.0005, 0.0002, 0.0)
    assert len(out["idx"]) == 0 and out["n_orders"] == 1
    b2 = toy_book()
    b2.tmin[:] = np.nan
    b2.tmin[7] = b2.bid[5] - 0.1  # a sell prints through our bid inside the window (buckets 6..8)
    out = sim.simulate(one_signal(5, 1.0), valid, b2, 3, 0, "maker", 0.0005, 0.0002, 0.0)
    assert len(out["idx"]) == 1 and out["entry_px"][0] == b2.bid[5]
    b3 = toy_book()
    b3.tmin[:] = np.nan
    b3.tmin[7] = b3.bid[5]  # touching our price is not enough
    out = sim.simulate(one_signal(5, 1.0), valid, b3, 3, 0, "maker", 0.0005, 0.0002, 0.0)
    assert len(out["idx"]) == 0
