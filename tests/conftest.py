import numpy as np
import polars as pl
import pytest

from ofp import config  # noqa: F401


def toy_raw(n_seconds: int = 900, seed: int = 0, day_start: int = 1_700_000_000_000):
    """Random-walk L1 book with ~5 updates per second and ~2 trades per second (ms timestamps)."""
    rng = np.random.default_rng(seed)
    n_q = n_seconds * 5
    ts = np.sort(rng.integers(day_start, day_start + n_seconds * 1000, n_q))
    tick = 0.1
    mid_ticks = 300_000 + np.cumsum(rng.choice([-1, 0, 0, 0, 1], n_q))
    spread = rng.choice([1, 1, 1, 2], n_q)
    bid = mid_ticks * tick
    ask = (mid_ticks + spread) * tick
    quotes = pl.DataFrame({
        "update_id": np.arange(n_q, dtype=np.int64) + 10,
        "bid": np.round(bid, 1), "bidq": rng.uniform(0.1, 5, n_q).round(3),
        "ask": np.round(ask, 1), "askq": rng.uniform(0.1, 5, n_q).round(3),
        "ts": ts.astype(np.int64),
    })
    n_t = n_seconds * 2
    tts = np.sort(rng.integers(day_start, day_start + n_seconds * 1000, n_t))
    pos = np.searchsorted(ts, tts, side="right") - 1
    pos = np.clip(pos, 0, n_q - 1)
    is_bm = rng.random(n_t) < 0.5
    price = np.where(is_bm, bid[pos] - tick * rng.integers(0, 2, n_t), ask[pos] + tick * rng.integers(0, 2, n_t))
    trades = pl.DataFrame({"price": np.round(price, 1), "qty": rng.uniform(0.001, 2, n_t).round(3),
                           "ts": tts.astype(np.int64), "is_buyer_maker": is_bm})
    return quotes, trades, day_start


@pytest.fixture
def raw():
    return toy_raw()
