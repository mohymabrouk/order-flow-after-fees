"""Test 5 (leak test): changing raw data after the end of bucket t leaves the features and forecasts for t unchanged."""

import numpy as np
import polars as pl
import torch

from ofp import bars as B
from ofp import config
from ofp import features as F
from ofp import models as M

N_SEC = 900
T = 600  # perturb everything from the close of bucket T on


def _feats(quotes, trades, day):
    bars, _ = B.make_bars(quotes, trades, day, n_buckets=N_SEC)
    return F.compute(bars)


def _perturb(quotes, trades, day, seed=1):
    rng = np.random.default_rng(seed)
    cut = day + (T + 1) * 1000  # close of bucket T
    q_after = quotes["ts"] >= cut
    t_after = trades["ts"] >= cut
    q2 = quotes.with_columns(
        pl.when(q_after).then(pl.col("bid") + 50.0).otherwise(pl.col("bid")).alias("bid"),
        pl.when(q_after).then(pl.col("ask") + 50.0).otherwise(pl.col("ask")).alias("ask"),
        pl.when(q_after).then(pl.Series(rng.uniform(0.1, 9, quotes.height))).otherwise(pl.col("bidq")).alias("bidq"),
    )
    t2 = trades.with_columns(
        pl.when(t_after).then(pl.col("qty") * 7.0).otherwise(pl.col("qty")).alias("qty"),
        pl.when(t_after).then(~pl.col("is_buyer_maker")).otherwise(pl.col("is_buyer_maker")).alias("is_buyer_maker"),
    )
    return q2, t2


def test_features_and_forecasts_unchanged_by_future(raw):
    quotes, trades, day = raw
    f1 = _feats(quotes, trades, day)
    q2, t2 = _perturb(quotes, trades, day)
    f2 = _feats(q2, t2, day)
    cols = list(dict.fromkeys(F.FEATURES + F.SEQ_CHANNELS + ["valid"]))
    a = f1.select(cols)[: T + 1].to_numpy()
    b = f2.select(cols)[: T + 1].to_numpy()
    np.testing.assert_array_equal(a, b)
    # sanity: the perturbation does change later rows and the targets of row T
    assert not np.array_equal(f1.select(F.FEATURES)[T + 5:].to_numpy(), f2.select(F.FEATURES)[T + 5:].to_numpy())
    assert f1["y_5"][T] != f2["y_5"][T]

    # forecasts: fit small rungs on the unperturbed features, predict on both versions
    ok = f1["valid"].to_numpy()
    x1 = f1.select(F.FEATURES).to_numpy().astype(np.float32)
    x2 = f2.select(F.FEATURES).to_numpy().astype(np.float32)
    y = np.nan_to_num(f1.select(F.TARGETS).to_numpy().astype(np.float32))
    scaler = M.Scaler(x1[ok])
    rungs = [M.SignOFI(), M.LinearOFI().fit(x1[ok], y[ok], scaler), M.Ridge().fit(x1[ok], y[ok], scaler, x1[ok], y[ok])]
    torch.manual_seed(0)
    mlp = M.MLP()
    mlp.scaler, mlp.ysd = scaler, np.ones(len(config.HORIZONS), np.float32)
    mlp.net = M.MLPNet(len(F.FEATURES), config.MLP_HIDDEN, len(config.HORIZONS)).eval()
    rungs.append(mlp)
    rungs.append(M.PerSymbol("ridge", {0: rungs[2]}))
    rows = np.arange(config.WARMUP, T + 1)
    sid = np.zeros(len(rows), dtype=np.int8)
    for r in rungs:
        np.testing.assert_array_equal(r.predict(x1[rows], sid), r.predict(x2[rows], sid))
    # sequence rung: windows ending at t <= T only see rows <= T
    cnn = M.CNNLSTM()
    cnn.net = M.SeqNet(len(F.SEQ_CHANNELS), 4, 8, len(config.HORIZONS)).eval()
    cnn.mu, cnn.sd = np.zeros(len(F.SEQ_CHANNELS), np.float32), np.ones(len(F.SEQ_CHANNELS), np.float32)
    cnn.ysd = np.ones(len(config.HORIZONS), np.float32)
    c1 = np.nan_to_num(f1.select(F.SEQ_CHANNELS).to_numpy().astype(np.float32))
    c2 = np.nan_to_num(f2.select(F.SEQ_CHANNELS).to_numpy().astype(np.float32))
    np.testing.assert_array_equal(cnn.predict_rows(c1, rows), cnn.predict_rows(c2, rows))
