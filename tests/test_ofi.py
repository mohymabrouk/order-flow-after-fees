"""Test 1: OFI on a hand-made sequence of best bid/ask updates equals the Cont-Kukanov-Stoikov formula."""

import numpy as np
import polars as pl

from ofp import bars as B

DAY = 1_700_000_000_000


def test_cks_hand_computed():
    # update: (bid, bidq, ask, askq, ms)
    seq = [
        (100.0, 5.0, 100.1, 4.0, 100),   # n=0: first event, e=0
        (100.0, 7.0, 100.1, 4.0, 200),   # bid size up 2 at same price: e = 7 - 5 = +2
        (100.0, 7.0, 100.1, 1.0, 300),   # ask size down 3 at same price: e = -1 + 4 = +3
        (100.1, 2.0, 100.2, 6.0, 1200),  # bid improves (e += 2), ask moves up (e += +1 old ask size): e = 2 + 1 = +3
        (100.0, 9.0, 100.2, 6.0, 1500),  # bid drops: e = -2 (old bid size removed); ask unchanged: -6 + 6 = 0 -> -2
        (100.0, 9.0, 100.1, 3.0, 2100),  # ask improves: e = -(3) ; bid same: 9 - 9 = 0 -> -3
    ]
    expected_e = [0.0, 2.0, 3.0, 3.0, -2.0, -3.0]
    q = pl.DataFrame({"update_id": np.arange(len(seq)), "bid": [s[0] for s in seq], "bidq": [s[1] for s in seq],
                      "ask": [s[2] for s in seq], "askq": [s[3] for s in seq],
                      "ts": [DAY + s[4] for s in seq]})
    e_polars = q.with_columns(B.cks_e_expr().alias("e"))["e"].to_numpy()
    e_loop = B.cks_e_numpy(q["bid"].to_numpy(), q["bidq"].to_numpy(), q["ask"].to_numpy(), q["askq"].to_numpy())
    np.testing.assert_allclose(e_polars, expected_e)
    np.testing.assert_allclose(e_loop, expected_e)
    trades = pl.DataFrame({"price": [100.1], "qty": [1.0], "ts": [DAY + 250], "is_buyer_maker": [False]})
    bars, stats = B.make_bars(q, trades, DAY, n_buckets=3)
    # OFI per 1 s bucket = sum of e_n with event time in [k, k+1) s
    np.testing.assert_allclose(bars["ofi"].to_numpy(), [0 + 2 + 3, 3 - 2, -3])
    assert bars["n_upd"].to_list() == [3, 2, 1]
    # quote state at the bucket close = last update strictly before the close
    assert bars["bid"].to_list() == [100.0, 100.0, 100.0] and bars["ask"].to_list() == [100.1, 100.2, 100.1]
    # +100 ms state: bucket 0 close + 100 ms = 1100 ms -> still the 300 ms update; bucket 1: 2100 ms update is not < 2100
    assert bars["ask100"].to_list()[:2] == [100.1, 100.2]


def test_polars_equals_loop_on_random_book(raw):
    quotes, _, _ = raw
    e1 = quotes.with_columns(B.cks_e_expr().alias("e"))["e"].to_numpy()
    e2 = B.cks_e_numpy(*(quotes[c].to_numpy() for c in ["bid", "bidq", "ask", "askq"]))
    np.testing.assert_allclose(e1, e2)
