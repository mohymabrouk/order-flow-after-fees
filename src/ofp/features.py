"""Build stage: 1-second bars -> features (known at the end of bucket t), targets (mid change over t -> t+h) and the quote
columns the trading simulator needs. One parquet per (symbol, day); every day is a self-contained session.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import polars as pl

from ofp import config
from ofp.download import bar_path

log = logging.getLogger("features")

FEATURES = (
    ["ofi_n", "ofi_n_l1", "ofi_n_l2"]
    + [f"ofi_s{w}" for w in config.OFI_SUMS]
    + ["ti_1"] + [f"ti_{w}" for w in config.FLOW_WINDOWS]
    + ["sv_1"] + [f"sv_{w}" for w in config.FLOW_WINDOWS]
    + ["qi", "spread_bps"]
    + [f"ret_{w}" for w in config.RET_WINDOWS]
    + [f"vol_{w}" for w in config.VOL_WINDOWS]
    + ["act_60"]
)
SEQ_CHANNELS = ["ofi_n", "ti_1", "sv_1", "qi", "spread_bps", "ret_1", "lupd", "ltrd"]
TARGETS = [f"y_{h}" for h in config.HORIZONS]
SIM_COLS = ["bid", "ask", "bid100", "ask100", "tmin", "tmax"]
STALE_SECONDS = 10


def feature_path(symbol: str, day):
    return config.FEATURES / symbol / f"{symbol}_{day.isoformat()}.parquet"


def compute(bars: pl.DataFrame) -> pl.DataFrame:
    """All features use rolling/shift operations that look backwards only (bars <= t)."""
    b = bars.sort("k")
    mid = (pl.col("bid") + pl.col("ask")) / 2
    b = b.with_columns(
        mid.alias("mid"),
        pl.col("depth_mean").rolling_mean(config.DEPTH_WINDOW, min_samples=1).alias("depth_ref"),
        (pl.col("buy_n") - pl.col("sell_n")).alias("_tn"),
        (pl.col("buy_n") + pl.col("sell_n")).alias("_tt"),
        (pl.col("buy_qty") - pl.col("sell_qty")).alias("_sv"),
        pl.when(pl.col("n_upd") > 0).then(pl.col("k")).otherwise(None).forward_fill().alias("_last_upd"),
    )
    lm = pl.col("mid").log() * 1e4
    b = b.with_columns((pl.col("ofi") / pl.col("depth_ref")).alias("ofi_n"), (lm - lm.shift(1)).alias("ret_1"))
    exprs = [pl.col("ofi_n").shift(1).alias("ofi_n_l1"), pl.col("ofi_n").shift(2).alias("ofi_n_l2")]
    exprs += [(pl.col("ofi").rolling_sum(w) / pl.col("depth_ref")).alias(f"ofi_s{w}") for w in config.OFI_SUMS]
    safe = lambda num, den: pl.when(den > 0).then(num / den).otherwise(0.0)  # noqa: E731
    exprs.append(safe(pl.col("_tn").cast(pl.Float64), pl.col("_tt").cast(pl.Float64)).alias("ti_1"))
    exprs += [safe(pl.col("_tn").rolling_sum(w).cast(pl.Float64), pl.col("_tt").rolling_sum(w).cast(pl.Float64)).alias(f"ti_{w}")
              for w in config.FLOW_WINDOWS]
    exprs.append((pl.col("_sv") / pl.col("depth_ref")).alias("sv_1"))
    exprs += [(pl.col("_sv").rolling_sum(w) / pl.col("depth_ref")).alias(f"sv_{w}") for w in config.FLOW_WINDOWS]
    exprs.append(((pl.col("bidq") - pl.col("askq")) / (pl.col("bidq") + pl.col("askq"))).alias("qi"))
    exprs.append(((pl.col("ask") - pl.col("bid")) / pl.col("mid") * 1e4).alias("spread_bps"))
    exprs += [(lm - lm.shift(w)).alias(f"ret_{w}") for w in config.RET_WINDOWS if w != 1]
    exprs += [(pl.col("ret_1") ** 2).rolling_mean(w).sqrt().alias(f"vol_{w}") for w in config.VOL_WINDOWS]
    exprs.append(pl.col("n_upd").cast(pl.Float64).rolling_mean(60).log1p().alias("act_60"))
    exprs.append(pl.col("n_upd").cast(pl.Float64).log1p().alias("lupd"))
    exprs.append(pl.col("_tt").cast(pl.Float64).log1p().alias("ltrd"))
    # Targets: strictly future mid changes (bps of the mid at the end of bucket t).
    exprs += [((pl.col("mid").shift(-h) - pl.col("mid")) / pl.col("mid") * 1e4).alias(f"y_{h}") for h in config.HORIZONS]
    b = b.with_columns(exprs)
    feat_ok = pl.all_horizontal([pl.col(c).is_not_null() & pl.col(c).is_finite() for c in FEATURES + SEQ_CHANNELS])
    valid = (
        (pl.col("k") >= config.WARMUP)
        & feat_ok
        & (pl.col("ask") > pl.col("bid"))
        & ((pl.col("k") - pl.col("_last_upd")) <= STALE_SECONDS).fill_null(False)
    )
    b = b.with_columns(valid.fill_null(False).alias("valid"))
    keep = ["k", "valid", "ofi", "mid"] + FEATURES + ["lupd", "ltrd"] + TARGETS + SIM_COLS
    out = b.select(keep)
    f32 = [c for c in FEATURES + ["lupd", "ltrd"] + TARGETS]
    return out.with_columns([pl.col(c).cast(pl.Float32) for c in f32])


def build_all(symbols=None) -> None:
    from ofp.download import manifest_rows

    rows = manifest_rows()
    symbols = symbols or list(config.SYMBOLS)
    n = 0
    t0 = time.time()
    for r in rows:
        if r["symbol"] not in symbols:
            continue
        from datetime import date

        day = date.fromisoformat(r["day"])
        out = feature_path(r["symbol"], day)
        if out.exists():
            continue
        bars = pl.read_parquet(bar_path(r["symbol"], day))
        f = compute(bars)
        out.parent.mkdir(parents=True, exist_ok=True)
        f.write_parquet(out, compression="zstd")
        n += 1
        if n % 20 == 0:
            log.info("built %d feature files (%.0fs)", n, time.time() - t0)
    log.info("build done: %d new feature files in %.0fs", n, time.time() - t0)


def available(split: str, symbols=None) -> list[tuple[str, object]]:
    """(symbol, day) pairs with a feature file, in date order, for one split."""
    out = []
    for d in config.sampled_days():
        if config.day_split(d) != split:
            continue
        for s in symbols or config.SYMBOLS:
            if feature_path(s, d).exists():
                out.append((s, d))
    return out


def load_panel(split: str, columns: list[str], symbols=None, only_valid: bool = False, subsample: float | None = None,
               seed: int = config.SEED) -> tuple[pl.DataFrame, list[tuple[str, object]]]:
    """Concatenate (symbol, day) feature files of a split. Full days keep all 86,400 rows (needed by the simulator)."""
    rng = np.random.default_rng(seed)
    frames, keys = [], []
    for i, (s, d) in enumerate(available(split, symbols)):
        lf = pl.scan_parquet(feature_path(s, d)).select(["k", "valid"] + [c for c in columns if c not in ("k", "valid")])
        if only_valid:
            lf = lf.filter(pl.col("valid"))
        df = lf.collect()
        if subsample is not None:
            df = df.filter(pl.Series(rng.random(df.height) < subsample))
        frames.append(df.with_columns(pl.lit(i, dtype=pl.Int32).alias("seg"),
                                      pl.lit(config.SYMBOLS.index(s), dtype=pl.Int8).alias("sid")))
        keys.append((s, d))
    return pl.concat(frames, how="vertical"), keys
