"""Execution-timing overlay: toy-book fill test and leak tests (signal-level and end to end from raw data)."""

import numpy as np
import polars as pl

from ofp import bars as B
from ofp import config
from ofp import features as F
from ofp import models as M
from ofp import overlay as O


def toy():
    n = 40
    k = np.arange(n, dtype=float)
    bid = 100 + k
    ask = bid + 0.5
    bid100 = 1000 + k  # distinct values: we can see which quote was used
    ask100 = bid100 + 0.5
    return n, bid, ask, bid100, ask100


def test_overlay_fills_on_toy_book():
    n, bid, ask, bid100, ask100 = toy()
    valid = np.ones(n, bool)
    starts = np.array([5, 20])
    sig = np.zeros(n)
    sig[8] = 1.0    # first favourable second for a buy starting at 5
    sig[22] = -1.0  # first favourable second for a sell starting at 20
    res = O.simulate_overlay(sig, valid, bid, ask, bid100, ask100, starts, W=5)
    # layout: buys for each start, then sells for each start
    assert res["side"].tolist() == [1, 1, -1, -1]
    assert res["exec_row"].tolist() == [8, 24, 9, 22]  # buy@20 and sell@5: no favourable second -> window end
    assert res["forced"].tolist() == [False, True, True, False]
    m0 = (bid[starts] + ask[starts]) / 2
    np.testing.assert_allclose(res["cost_rule"][:2], (ask100[[8, 24]] - m0) / m0 * 1e4)  # buys at ask100 of the decision row
    np.testing.assert_allclose(res["cost_rule"][2:], (m0 - bid100[[9, 22]]) / m0 * 1e4)  # sells at bid100
    np.testing.assert_allclose(res["cost_now"][:2], (ask100[starts] - m0) / m0 * 1e4)
    rand_buy = np.array([ask100[s:s + 5].mean() for s in starts])
    np.testing.assert_allclose(res["cost_random"][:2], (rand_buy - m0) / m0 * 1e4)
    np.testing.assert_allclose(res["save_now"], res["cost_now"] - res["cost_rule"])
    # a buy executed later on a rising book costs more -> negative saving vs now; positive = better price
    assert res["save_now"][0] < 0
    # an invalid row is never favourable
    v2 = valid.copy()
    v2[8] = False
    res2 = O.simulate_overlay(sig, v2, bid, ask, bid100, ask100, starts, W=5)
    assert res2["exec_row"][0] == 9


def test_bootstrap_summary_counts():
    n, bid, ask, bid100, ask100 = toy()
    starts = np.array([5, 20])
    res = O.simulate_overlay(np.ones(n), np.ones(n, bool), bid, ask, bid100, ask100, starts, W=5)
    s = O.summarize(res, np.array([0, 1, 0, 1]), 2)
    assert s["orders"] == 4 and s["days"] == 2
    assert s["save_vs_now_ci_lo"] <= s["save_vs_now_bp"] <= s["save_vs_now_ci_hi"]


def test_overlay_decision_uses_only_the_past():
    rng = np.random.default_rng(3)
    n = 3000
    mid = 100 + np.cumsum(rng.normal(0, 0.01, n))
    bid, ask = mid - 0.005, mid + 0.005
    bid100, ask100 = bid + rng.normal(0, 0.001, n), ask + rng.normal(0, 0.001, n)
    valid = rng.random(n) > 0.05
    sig = rng.normal(size=n)
    starts = np.arange(100, 2800, 37)
    for W in O.WINDOWS:
        base = O.simulate_overlay(sig, valid, bid, ask, bid100, ask100, starts, W)
        for t_star in (500, 1500, 2200):
            sig2, v2, a2, b2 = sig.copy(), valid.copy(), ask100.copy(), bid100.copy()
            sig2[t_star + 1:] = rng.normal(size=n - t_star - 1) * 5
            v2[t_star + 1:] = rng.random(n - t_star - 1) > 0.5
            a2[t_star + 1:] += 7.0
            b2[t_star + 1:] -= 7.0
            pert = O.simulate_overlay(sig2, v2, bid, ask, b2, a2, starts, W)
            # every decision up to t_star is unchanged: "executed at t <= t_star" and where
            np.testing.assert_array_equal(np.minimum(base["exec_row"], t_star + 1), np.minimum(pert["exec_row"], t_star + 1))
            done = base["exec_row"] <= t_star
            np.testing.assert_array_equal(base["cost_rule"][done], pert["cost_rule"][done])


def test_overlay_leak_end_to_end_from_raw(raw):
    """Perturb raw quotes/trades after the close of bucket T: decisions up to T and fills priced before close(T) unchanged."""
    from test_leak import _perturb, T  # tests/ is on sys.path (pytest prepend import mode)

    quotes, trades, day = raw
    n_sec = 900
    f1 = F.compute(B.make_bars(quotes, trades, day, n_buckets=n_sec)[0])
    q2, t2 = _perturb(quotes, trades, day)
    f2 = F.compute(B.make_bars(q2, t2, day, n_buckets=n_sec)[0])
    ok = f1["valid"].to_numpy()
    x1 = f1.select(F.FEATURES).to_numpy().astype(np.float32)
    x2 = f2.select(F.FEATURES).to_numpy().astype(np.float32)
    y = np.nan_to_num(f1.select(F.TARGETS).to_numpy().astype(np.float32))
    ridge = M.Ridge().fit(x1[ok], y[ok], M.Scaler(x1[ok]), x1[ok], y[ok])
    starts = np.arange(config.WARMUP, n_sec - 60, 13)
    starts = starts[ok[starts]]
    for W in O.WINDOWS:
        out = []
        for f, x in ((f1, x1), (f2, x2)):
            sig = np.where(f["valid"].to_numpy(), ridge.predict(x)[:, 0], np.nan)
            cols = [f[c].to_numpy().astype(float) for c in ("bid", "ask", "bid100", "ask100")]
            out.append(O.simulate_overlay(sig, f["valid"].to_numpy(), *cols, starts, W))
        a, b = out
        keep = a["start"] <= T
        np.testing.assert_array_equal(np.minimum(a["exec_row"][keep], T + 1), np.minimum(b["exec_row"][keep], T + 1))
        early = keep & (a["exec_row"] <= T - 1)  # fill quote at close(t) + 100 ms < close(T)
        assert early.any()
        np.testing.assert_array_equal(a["cost_rule"][early], b["cost_rule"][early])


def test_next_true():
    m = np.array([False, True, False, False, True, False])
    assert O.next_true(m).tolist() == [1, 1, 4, 4, 4, 6]
    assert pl is not None
