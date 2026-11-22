"""Test 2: mid-price targets are aligned strictly in the future."""

import numpy as np

from ofp import bars as B
from ofp import config
from ofp import features as F


def test_targets_strictly_future(raw):
    quotes, trades, day = raw
    bars, _ = B.make_bars(quotes, trades, day, n_buckets=900)
    f = F.compute(bars)
    mid = f["mid"].to_numpy()
    for h in config.HORIZONS:
        y = f[f"y_{h}"].to_numpy().astype(float)
        exp = (mid[h:] - mid[:-h]) / mid[:-h] * 1e4
        np.testing.assert_allclose(y[:-h], exp, rtol=1e-5, atol=1e-4)
        assert np.all(np.isnan(y[-h:]) | (y[-h:] == None))  # noqa: E711  no target past the end of the day
        # the target at t depends on mid at t+h (> t): it is not a function of the past only
        t = 400
        assert y[t] == np.float32((mid[t + h] - mid[t]) / mid[t] * 1e4)


def test_mid_is_close_state(raw):
    """mid_t uses the quote state at the END of bucket t, so y_h[t] starts after every feature input of row t."""
    quotes, trades, day = raw
    bars, _ = B.make_bars(quotes, trades, day, n_buckets=900)
    f = F.compute(bars)
    k = 500
    close_ms = day + (k + 1) * 1000
    last = quotes.filter(quotes["ts"] < close_ms).tail(1)
    assert f["mid"][k] == (last["bid"][0] + last["ask"][0]) / 2
