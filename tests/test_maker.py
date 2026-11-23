"""Maker-or-taker per order (Change 16): fill rules on tiny synthetic tapes, the 100 ms quote grid against the bars,
no look-ahead in the decision or the fills, T = 0 equals taker at once, and (with the local cache) the overlay's published
test numbers are reproduced unchanged."""

import numpy as np
import polars as pl
import pytest

from ofp import bars as B
from ofp import config
from ofp import features as F
from ofp import maker as M
from ofp import models as Mo

N_SEC = 900


def _chunks(quotes: pl.DataFrame, rows: int = 400, header: bool = False, seed: int | None = None) -> list[bytes]:
    df = quotes.select(pl.col("update_id"), pl.col("bid").alias("best_bid_price"), pl.col("bidq").alias("best_bid_qty"),
                       pl.col("ask").alias("best_ask_price"), pl.col("askq").alias("best_ask_qty"),
                       pl.col("ts").alias("transaction_time"), pl.col("ts").alias("event_time"))
    if seed is not None:  # out-of-order rows, as on some Binance days
        df = df.sample(fraction=1.0, shuffle=True, seed=seed)
    lines = df.write_csv(include_header=header).encode().splitlines(keepends=True)
    return [b"".join(lines[i:i + rows]) for i in range(0, len(lines), rows)]


def _segment(quotes, trades, day):
    bars = B.make_bars(quotes, trades, day, n_buckets=N_SEC)[0]
    grid, _ = M.grid_states_from_chunks(_chunks(quotes), day, n_grid=N_SEC * 10 + 2)
    tr = trades.sort("ts", maintain_order=True)
    tape = {c: tr[c].to_numpy() for c in ("ts", "price", "qty", "is_buyer_maker")}
    q = {c: bars[c].to_numpy().astype(float) for c in ("bid", "ask", "bid100", "ask100")}
    return bars, grid, tape, q


def test_grid_equals_bars_quote_states(raw):
    quotes, trades, day = raw
    bars = B.make_bars(quotes, trades, day, n_buckets=N_SEC)[0]
    for header, seed in ((False, None), (True, None), (False, 7)):
        grid, n = M.grid_states_from_chunks(_chunks(quotes, header=header, seed=seed), day, n_grid=N_SEC * 10 + 2)
        assert n == quotes.height
        chk = M.verify_day(grid, trades.filter(pl.col("ts") < day + N_SEC * 1000), bars, day, n_buckets=N_SEC)
        assert chk["ok"], chk


def _tape(rows):
    ts, px, qty, ibm = zip(*rows)
    return np.array(ts, np.int64), np.array(px, float), np.array(qty, float), np.array(ibm, bool)


def test_trade_through_fills_only_strictly_through_and_after_posting():
    tape = _tape([(900, 99.9, 1.0, True),     # before posting: ignored
                  (1000, 99.9, 1.0, True),    # at the posting instant: ignored (strictly after only)
                  (1500, 100.0, 9.0, True),   # at our price: touching is not a trade-through
                  (2500, 99.9, 0.1, True)])   # strictly below our bid: filled here
    one = lambda v: np.array([v], float)  # noqa: E731
    d = M.fill_delays(np.array([1000]), one(100.0), one(1), one(5.0), one(1.0), *tape, horizon_ms=30_000)
    assert d["through"][0] == 1500
    d = M.fill_delays(np.array([1000]), one(100.0), one(1), one(5.0), one(1.0), *tape, horizon_ms=1_000)
    assert np.isinf(d["through"][0])  # the print through comes after the window
    # a sell posted at the ask: only prints above it count
    tape = _tape([(1100, 100.5, 3.0, False), (1200, 100.4, 3.0, True), (1300, 100.6, 0.1, False)])
    d = M.fill_delays(np.array([1000]), one(100.5), one(-1), one(5.0), one(1.0), *tape, horizon_ms=30_000)
    assert d["through"][0] == 300


def test_queue_rule_respects_queue_ahead_and_aggressor_side():
    one = lambda v: np.array([v], float)  # noqa: E731
    tape = _tape([(1100, 100.0, 3.0, True),    # seller-initiated at our bid: counts (3)
                  (1200, 100.0, 10.0, False),  # buyer-initiated: not against our bid
                  (1300, 100.1, 10.0, True),   # seller-initiated above our bid: not at or through
                  (1400, 100.0, 3.0, True),    # 6.0 = queue 5 + size 1: not yet MORE than queue + size
                  (1500, 100.0, 0.5, True)])   # 6.5 > 6: filled
    d = M.fill_delays(np.array([1000]), one(100.0), one(1), one(5.0), one(1.0), *tape, horizon_ms=30_000)
    assert d["queue"][0] == 500 and np.isinf(d["through"][0])
    d = M.fill_delays(np.array([1000]), one(100.0), one(1), one(6.0), one(1.0), *tape, horizon_ms=30_000)
    assert np.isinf(d["queue"][0])  # a bigger queue ahead: never reached
    # volume through the price counts toward the queue; a small print through fills (a) but not (b)
    tape = _tape([(1100, 99.9, 0.2, True)])
    d = M.fill_delays(np.array([1000]), one(100.0), one(1), one(50.0), one(1.0), *tape, horizon_ms=30_000)
    assert d["through"][0] == 100 and np.isinf(d["queue"][0])
    d = M.fill_delays(np.array([1000]), one(100.0), one(1), one(0.0), one(0.1), *tape, horizon_ms=30_000)
    assert d["queue"][0] == 100


def _orders(raw, seed=0):
    quotes, trades, day = raw
    bars, grid, tape, q = _segment(quotes, trades, day)
    valid = F.compute(bars)["valid"].to_numpy()
    k0 = np.arange(config.WARMUP, N_SEC - 60, 11)
    k0 = k0[valid[k0]]
    side = np.concatenate([np.ones(len(k0)), -np.ones(len(k0))])
    k0 = np.concatenate([k0, k0])
    od = M.segment_orders(k0, side, q["bid"], q["ask"], q["bid100"], q["ask100"], grid, tape, day)
    assert od.pop("grid_matches_bars")
    return od


def test_zero_wait_equals_taker_at_once(raw):
    od = _orders(raw)
    rng = np.random.default_rng(0)
    now = M.taker_now_cost(od)
    for rule in M.RULES:
        for cross in (np.zeros(len(now), bool), rng.random(len(now)) < 0.5):
            c, posted, filled = M.policy(od, rule, 0, cross)
            np.testing.assert_array_equal(c, now)
            assert not filled.any()
        # crossing every order at once is also taker at once, whatever T
        for T in M.WAITS:
            np.testing.assert_array_equal(M.policy(od, rule, T, np.ones(len(now), bool))[0], now)


def test_policy_costs_and_fees(raw):
    od = _orders(raw)
    T = 5
    for rule in M.RULES:
        c, posted, filled = M.policy(od, rule, T, np.zeros(len(od["side"]), bool))
        assert filled.any() and (~filled).any()
        f, u = filled, ~filled
        np.testing.assert_allclose(c[f], M.cost_bp(od["p_post"][f], M.FEE_MAKER, od["m0"][f], od["side"][f]))
        np.testing.assert_allclose(c[u], M.cost_bp(od[f"p_cross_{T}"][u], M.FEE_TAKER, od["m0"][u], od["side"][u]))
        assert np.all(od[f"delay_{rule}"][f] <= T * 1000) and np.all(od[f"delay_{rule}"][u] > T * 1000)
        # unfilled orders cross at the far touch 100 ms after close(k0 + T): ask100 for a buy, bid100 for a sell
        assert np.all(np.isfinite(od[f"p_cross_{T}"]))


def test_no_look_ahead_in_decision_or_fills(raw):
    """Perturb raw quotes and trades from the close of bucket Tc on, and every forecast after each decision row: every
    order whose whole wait ends before the cut keeps its decision, fill and cost."""
    from test_leak import _perturb, T as Tc  # tests/ is on sys.path

    quotes, trades, day = raw
    q2, t2 = _perturb(quotes, trades, day)
    cut = day + (Tc + 1) * 1000
    outs = []
    for qq, tt in ((quotes, trades), (q2, t2)):
        bars, grid, tape, q = _segment(qq, tt, day)
        f = F.compute(bars)
        x = f.select(F.FEATURES).to_numpy().astype(np.float32)
        outs.append((bars, grid, tape, q, f, x))
    f1, x1 = outs[0][4], outs[0][5]
    ok = f1["valid"].to_numpy()
    y = np.nan_to_num(f1.select(F.TARGETS).to_numpy().astype(np.float32))
    ridge = Mo.Ridge().fit(x1[ok], y[ok], Mo.Scaler(x1[ok]), x1[ok], y[ok])
    k0 = np.arange(config.WARMUP, Tc + 1, 7)
    k0 = k0[ok[k0]]
    side = np.concatenate([np.ones(len(k0)), -np.ones(len(k0))])
    k0 = np.concatenate([k0, k0])
    res = []
    for bars, grid, tape, q, f, x in outs:
        fc = ridge.predict(x)[:, 1]
        fc = np.where(f["valid"].to_numpy(), fc, np.nan)
        od = M.segment_orders(k0, side, q["bid"], q["ask"], q["bid100"], q["ask100"], grid, tape, day)
        res.append((od, M.decide_cross(fc[k0], side, 0.1)))
    (oa, ca), (ob, cb) = res
    np.testing.assert_array_equal(ca, cb)  # decisions at rows <= Tc only see data before the cut
    t_post = day + (k0 + 1) * 1000 + M.LATENCY_MS
    n_checked = 0
    for rule in M.RULES:
        for T in M.WAITS:
            done = t_post + T * 1000 < cut
            a = M.policy(oa, rule, T, ca)[0]
            b = M.policy(ob, rule, T, cb)[0]
            np.testing.assert_array_equal(a[done], b[done])
            n_checked += int(done.sum())
    assert n_checked > 0
    # sanity: the perturbation does change orders whose wait crosses the cut
    late = t_post + 30_000 >= cut
    assert not np.array_equal(M.policy(oa, "through", 30, ca)[0][late], M.policy(ob, "through", 30, cb)[0][late])


def _have_cache() -> bool:
    return (config.PREDS / "test" / "ridge.npy").exists() and bool(F.available("test"))


@pytest.mark.skipif(not _have_cache(), reason="needs the local data cache (features + cached test forecasts)")
def test_overlay_published_numbers_reproduced():
    from ofp import overlay as O

    rep = M.reproduce_overlay(O.Book("test"))
    assert rep["overlay_reproduced"], rep
