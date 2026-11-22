"""Thin slice: first 6 sampled BTCUSDT days (2 weeks), CKS OFI linear model fitted on the first 4, taker simulation on the
last 2, for every horizon/latency/fee level. Smoke test of the whole chain before widening. Output: results/thin_slice.csv
"""

import time

from ofp import config  # noqa: F401
import numpy as np
import pandas as pd
import polars as pl

from ofp import features as F
from ofp import sim
from ofp.download import bar_path

t0 = time.time()
days = [d for d in config.sampled_days()][:6]
frames = []
for i, d in enumerate(days):
    out = F.feature_path("BTCUSDT", d)
    if not out.exists():
        f = F.compute(pl.read_parquet(bar_path("BTCUSDT", d)))
        out.parent.mkdir(parents=True, exist_ok=True)
        f.write_parquet(out, compression="zstd")
    frames.append(pl.read_parquet(out).with_columns(pl.lit(i).alias("seg")))
panel = pl.concat(frames)
print("panel", panel.shape, "valid", panel["valid"].sum(), f"{time.time()-t0:.0f}s")
fit = panel.filter(pl.col("seg") < 4)
ev = panel.filter(pl.col("seg") >= 4)
book = sim.Book(*(ev[c].to_numpy() for c in ["bid", "ask", "bid100", "ask100", "tmin", "tmax"]))
seg_day = np.array([0, 0, 0, 0, 0, 1])  # segs 4,5 -> eval days 0,1 (after re-basing)
eval_days = [str(days[4]), str(days[5])]
valid = ev["valid"].to_numpy()
x_ev = ev["ofi_n"].to_numpy().astype(float)
spread = float(fit.filter(pl.col("valid"))["spread_bps"].mean())
rows = []
for h in config.HORIZONS:
    f = fit.filter(pl.col("valid") & pl.col(f"y_{h}").is_not_null())
    x, y = f["ofi_n"].to_numpy().astype(float), f[f"y_{h}"].to_numpy().astype(float)
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x_ev
    yv = ev[f"y_{h}"].to_numpy().astype(float)
    m = valid & np.isfinite(yv)
    r2 = 1 - np.sum((yv[m] - pred[m]) ** 2) / np.sum(yv[m] ** 2)
    nz = m & (yv != 0)
    acc = np.mean(np.sign(pred[nz]) == np.sign(yv[nz]))
    for lat in config.LATENCIES_MS:
        for fee_name, (ft, fm) in config.FEES.items():
            cost = 2 * ft * 1e4 + spread
            for mult in (0.0, 1.0):
                tr = sim.simulate(pred, valid, book, h, lat, "taker", ft, fm, mult * cost)
                # seg index within ev panel is idx // 86400 -> 0 or 1
                sm = sim.summarize(tr, np.array([0, 1]), eval_days, n_segments=2)
                rows.append({"h": h, "beta": b, "r2": r2, "acc": acc, "latency": lat, "fee": fee_name, "mult": mult,
                             **{k: sm[k] for k in ["trades", "gross_bps", "spread_bps", "fee_bps", "net_bps", "hit_rate"]}})
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df[(df.fee != "cheap")].round(3).to_string())
config.RESULTS.mkdir(exist_ok=True)
df.to_csv(config.RESULTS / "thin_slice.csv", index=False)
print(f"done in {time.time()-t0:.0f}s; mean train spread {spread:.4f} bps")
