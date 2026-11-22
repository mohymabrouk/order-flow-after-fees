"""Execution-timing overlay: a NEW analysis run after the main study's one-shot test.

It uses the frozen models' cached forecasts (no retraining, no refit) and the cached bars/features. A parent order (buy or
sell one unit within W seconds) is executed at the first second whose forecast is favourable (predicted up-move for a buy,
down-move for a sell), else at the window end. Taker fill after 100 ms, as in the main study. Benchmarks: execute at the
window start ("now") and at a uniformly random second (exact expectation). Saving = benchmark cost - rule cost, in bp of
the arrival mid; fees are identical across alternatives and excluded. Design pre-declared in PLAN.md.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone

from ofp import config  # noqa: F401  (thread pins first)

import numpy as np
import pandas as pd

from ofp import features as F

log = logging.getLogger("overlay")
WINDOWS = (5, 10, 30, 60)
FORECAST_H = (1, 5)
START_STEP = 60
N_BOOT = 5000
H = list(config.HORIZONS)
COLS = ["bid", "ask", "bid100", "ask100"]
LEARNED = ("linear_ofi", "ridge", "lgbm", "mlp", "cnn_lstm")


# ------------------------------------------------------------------------------------------------- core
def next_true(mask: np.ndarray) -> np.ndarray:
    """nxt[i] = smallest j >= i with mask[j] (len(mask) if none). Uses only rows >= i, so no look-back issue."""
    n = len(mask)
    idx = np.where(mask, np.arange(n), n)
    return np.minimum.accumulate(idx[::-1])[::-1]


def order_starts(valid: np.ndarray, k: np.ndarray, seg_len: int = config.SECONDS_PER_DAY) -> np.ndarray:
    """Start rows: every START_STEP seconds after the warm-up, valid bucket, longest window fits inside the day."""
    ok = (k >= config.WARMUP) & ((k - config.WARMUP) % START_STEP == 0) & (k + max(WINDOWS) - 1 < seg_len) & valid
    return np.flatnonzero(ok)


def simulate_overlay(signal, valid, bid, ask, bid100, ask100, starts, W: int) -> dict:
    """Per parent order (both sides at every start): decision row, costs of the rule and the two benchmarks (bp)."""
    sig = np.where(np.isfinite(signal), signal, 0.0)
    m0 = (bid[starts] + ask[starts]) / 2
    out = {k: [] for k in ("start", "side", "exec_row", "forced", "cost_rule", "cost_now", "cost_random")}
    for side in (1, -1):
        fav = valid & (side * sig > 0)
        nxt = next_true(fav)
        last = starts + W - 1
        t = np.minimum(nxt[starts], last)
        q = ask100 if side == 1 else bid100  # taker: buy at the ask, sell at the bid, 100 ms after the decision
        rand = np.zeros(len(starts))
        for j in range(W):
            rand += q[starts + j]
        rand /= W
        cost = lambda px: side * (px - m0) / m0 * 1e4  # noqa: E731
        out["start"].append(starts)
        out["side"].append(np.full(len(starts), side))
        out["exec_row"].append(t)
        out["forced"].append(nxt[starts] > last - 1)  # no favourable second before the window end
        out["cost_rule"].append(cost(q[t]))
        out["cost_now"].append(cost(q[starts]))
        out["cost_random"].append(cost(rand))
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["save_now"] = res["cost_now"] - res["cost_rule"]
    res["save_random"] = res["cost_random"] - res["cost_rule"]
    return res


def summarize(res: dict, order_day: np.ndarray, n_days: int, seed: int = config.SEED) -> dict:
    """Pooled mean savings with a daily block bootstrap (resample days) and the share of days with a positive mean."""
    rng = np.random.default_rng(seed)
    cnt = np.bincount(order_day, minlength=n_days).astype(float)
    out = {"orders": int(len(order_day)), "days": n_days,
           "exec_at_start_pct": 100 * float(np.mean(res["exec_row"] == res["start"])),
           "forced_pct": 100 * float(np.mean(res["forced"])),
           "mean_delay_s": float(np.mean(res["exec_row"] - res["start"])),
           "cost_rule_bp": float(np.mean(res["cost_rule"])), "cost_now_bp": float(np.mean(res["cost_now"])),
           "cost_random_bp": float(np.mean(res["cost_random"]))}
    draws = rng.integers(0, n_days, size=(N_BOOT, n_days))
    for b in ("now", "random"):
        x = res[f"save_{b}"]
        s = np.bincount(order_day, weights=x, minlength=n_days)
        day_mean = np.divide(s, cnt, out=np.zeros_like(s), where=cnt > 0)
        boot = s[draws].sum(1) / cnt[draws].sum(1)
        out[f"save_vs_{b}_bp"] = float(x.mean())
        out[f"save_vs_{b}_ci_lo"] = float(np.quantile(boot, 0.025))
        out[f"save_vs_{b}_ci_hi"] = float(np.quantile(boot, 0.975))
        out[f"pos_days_vs_{b}_pct"] = 100 * float(np.mean(day_mean[cnt > 0] > 0))
    return out


# ------------------------------------------------------------------------------------------------- data
class Book:
    def __init__(self, split: str):
        t0 = time.time()
        df, self.keys = F.load_panel(split, COLS)
        self.valid = df["valid"].to_numpy()
        self.k = df["k"].to_numpy()
        self.bid, self.ask, self.bid100, self.ask100 = (df[c].to_numpy().astype(np.float64) for c in COLS)
        self.days = sorted({d for _, d in self.keys})
        self.seg_day = np.array([self.days.index(d) for _, d in self.keys])
        self.starts = order_starts(self.valid, self.k)
        self.n = df.height
        log.info("%s book: %d segments, %d rows, %d order starts (%.0fs)", split, len(self.keys), self.n, len(self.starts),
                 time.time() - t0)

    def preds(self, split: str, name: str) -> np.ndarray:
        p = np.load(config.PREDS / split / f"{name}.npy", mmap_mode="r")
        assert p.shape[0] == self.n, f"cached {split} predictions of {name} are not aligned with the panel"
        return p

    def run(self, signal: np.ndarray, W: int, breakdown: bool = False):
        res = simulate_overlay(signal, self.valid, self.bid, self.ask, self.bid100, self.ask100, self.starts, W)
        seg = res["start"] // config.SECONDS_PER_DAY
        day = self.seg_day[seg]
        out = summarize(res, day, len(self.days))
        if not breakdown:
            return out
        sym = np.array([s for s, _ in self.keys])[seg]
        parts = []
        for name, m in [(f"symbol {s}", sym == s) for s in sorted(set(sym))] + [("buys", res["side"] == 1),
                                                                                ("sells", res["side"] == -1)]:
            sub = {k: v[m] for k, v in res.items()}
            parts.append({"part": name, **summarize(sub, day[m], len(self.days))})
        return out, parts


def _signal(p: np.ndarray, h: int) -> np.ndarray:
    return np.asarray(p[:, H.index(h)], dtype=np.float64)


def _grid(name: str):
    hs = (1,) if name == "sign_ofi" else FORECAST_H  # sign(OFI) is the same for every horizon
    return [(h, W) for h in hs for W in WINDOWS]


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------- stages
def validate() -> None:
    """Score the 44-config grid on validation, choose per model, freeze."""
    t0 = time.time()
    book = Book("val")
    rows = []
    for name in config.MODELS_LADDER:
        p = book.preds("val", name)
        for h, W in _grid(name):
            s = book.run(_signal(p, h), W)
            rows.append({"model": name, "h": h, "W": W, **s})
            log.info("val %s h=%d W=%d: save vs now %.4f bp, vs random %.4f bp", name, h, W, s["save_vs_now_bp"],
                     s["save_vs_random_bp"])
    g = pd.DataFrame(rows)
    g.to_csv(config.RESULTS / "overlay_validation.csv", index=False)
    best = g.loc[g.groupby("model")["save_vs_now_bp"].idxmax()].set_index("model")
    learned = best.loc[list(LEARNED)].sort_values("save_vs_now_bp", ascending=False)
    chosen = ["sign_ofi"] + list(learned.index[:2])
    frozen = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule": "per model, (h, W) with the highest mean validation saving vs 'now'; test = sign_ofi + top-2 learned rungs",
        "grid": {"windows": WINDOWS, "forecast_h": FORECAST_H, "start_step_s": START_STEP, "latency_ms": 100},
        "n_configs": int(len(g)),
        "choices": [{"model": m, "h": int(best.loc[m, "h"]), "W": int(best.loc[m, "W"]),
                     "val_save_vs_now_bp": float(best.loc[m, "save_vs_now_bp"]),
                     "val_save_vs_random_bp": float(best.loc[m, "save_vs_random_bp"])} for m in chosen],
        "val_best_per_model": {m: {"h": int(r["h"]), "W": int(r["W"]), "save_vs_now_bp": float(r["save_vs_now_bp"])}
                               for m, r in best.iterrows()},
    }
    frozen["hash"] = _hash(frozen)
    (config.RESULTS / "overlay_frozen.json").write_text(json.dumps(frozen, indent=1))
    log.info("overlay validation done in %.0fs; frozen %s: %s", time.time() - t0, frozen["hash"], frozen["choices"])


def test(force: bool = False) -> None:
    from ofp.pipeline import code_hash

    log_path = config.RESULTS / "overlay_test_log.jsonl"
    if log_path.exists() and any(json.loads(x).get("event") == "done" for x in log_path.read_text().splitlines() if x.strip()):
        if not force:
            raise SystemExit("overlay test already run once (results/overlay_test_log.jsonl); --force reruns it (logged)")
    frozen = json.loads((config.RESULTS / "overlay_frozen.json").read_text())
    saved = frozen.pop("hash")
    if _hash(frozen) != saved:
        raise SystemExit("overlay frozen config hash mismatch")
    t0 = time.time()
    base = {"frozen_hash": saved, "code_hash": code_hash(), "forced_rerun": bool(force)}

    def _log(ev):
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({**ev, "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")

    _log({"event": "start", **base})
    book = Book("test")
    prim, byw, parts = [], [], []
    for c in frozen["choices"]:
        p = book.preds("test", c["model"])
        sig = _signal(p, c["h"])
        for W in WINDOWS:
            s, br = book.run(sig, W, breakdown=True)
            row = {"model": c["model"], "h": c["h"], "W": W, "primary": W == c["W"], **s}
            byw.append(row)
            if W == c["W"]:
                prim.append({**row, "val_save_vs_now_bp": c["val_save_vs_now_bp"]})
                parts += [{"model": c["model"], "h": c["h"], "W": W, **x} for x in br]
        log.info("test %s done", c["model"])
    pd.DataFrame(prim).to_csv(config.RESULTS / "overlay_test.csv", index=False)
    pd.DataFrame(byw).to_csv(config.RESULTS / "overlay_test_by_window.csv", index=False)
    pd.DataFrame(parts).to_csv(config.RESULTS / "overlay_test_breakdown.csv", index=False)
    _log({"event": "done", **base, "n_days": len(book.days), "orders_per_config": prim[0]["orders"] if prim else 0,
          "seconds": round(time.time() - t0)})
    log.info("overlay test done in %.0fs", time.time() - t0)


def report() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ofp.report import LABEL, md_table

    R = config.RESULTS
    v = pd.read_csv(R / "overlay_validation.csv")
    t = pd.read_csv(R / "overlay_test.csv")
    w = pd.read_csv(R / "overlay_test_by_window.csv")
    frozen = json.loads((R / "overlay_frozen.json").read_text())
    tl = [json.loads(x) for x in (R / "overlay_test_log.jsonl").read_text().splitlines() if x.strip()]
    out = ["# Execution-timing overlay (new analysis with the frozen models; no retraining)\n",
           "Saving = benchmark cost − rule cost, bp of the arrival mid, positive = the rule executed at a better price. "
           "Taker fill 100 ms after the decision; fees identical across alternatives and excluded. CI: 95 % daily block "
           "bootstrap (5,000 resamples of days). Orders: one buy and one sell every 60 s per symbol-day.\n"]

    def fmt_rows(df):
        recs = []
        for _, r in df.iterrows():
            recs.append({"model": LABEL[r["model"]], "h (s)": int(r["h"]), "W (s)": int(r["W"]), "orders": int(r["orders"]),
                         "saving vs now bp [95% CI]": f"{r['save_vs_now_bp']:+.3f} [{r['save_vs_now_ci_lo']:+.3f}, {r['save_vs_now_ci_hi']:+.3f}]",
                         "days > 0 (vs now) %": r["pos_days_vs_now_pct"],
                         "saving vs random bp [95% CI]": f"{r['save_vs_random_bp']:+.3f} [{r['save_vs_random_ci_lo']:+.3f}, {r['save_vs_random_ci_hi']:+.3f}]",
                         "days > 0 (vs random) %": r["pos_days_vs_random_pct"],
                         "cost rule / now / random bp": f"{r['cost_rule_bp']:.3f} / {r['cost_now_bp']:.3f} / {r['cost_random_bp']:.3f}",
                         "exec at start %": r["exec_at_start_pct"], "forced %": r["forced_pct"], "mean delay s": r["mean_delay_s"]})
        return pd.DataFrame(recs)

    fm = {"days > 0 (vs now) %": 0, "days > 0 (vs random) %": 0, "exec at start %": 0, "forced %": 1, "mean delay s": 2}
    out.append(f"\n## Test (primary: validation-chosen h and W; frozen {frozen['hash']})\n")
    out.append(md_table(fmt_rows(t), fm))
    bd = pd.read_csv(R / "overlay_test_breakdown.csv")
    out.append("\n## Test decomposition of the primary configs by symbol and by side (descriptive)\n")
    recs = [{"model": LABEL[r["model"]], "part": r["part"], "orders": int(r["orders"]),
             "saving vs now bp [95% CI]": f"{r['save_vs_now_bp']:+.3f} [{r['save_vs_now_ci_lo']:+.3f}, {r['save_vs_now_ci_hi']:+.3f}]",
             "days > 0 %": r["pos_days_vs_now_pct"], "cost now bp": r["cost_now_bp"], "cost rule bp": r["cost_rule_bp"]}
            for _, r in bd.iterrows()]
    out.append(md_table(pd.DataFrame(recs), {"days > 0 %": 0, "cost now bp": 3, "cost rule bp": 3}))
    out.append("\n## Test, every window W for the same three models (descriptive; same single run)\n")
    out.append(md_table(fmt_rows(w), fm))
    out.append(f"\n## Validation grid ({len(v)} configurations; choices made here)\n")
    out.append(md_table(fmt_rows(v.sort_values(["model", "h", "W"])), fm))
    out.append("\n## Overlay test log\n")
    for e in tl:
        out.append(f"- {e['event']} {e['timestamp_utc']} frozen {e['frozen_hash']} code {e['code_hash']} forced={e['forced_rerun']}")
    (R / "overlay_tables.md").write_text("\n".join(out) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    colors = dict(zip(config.MODELS_LADDER, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]))
    for ax, b in zip(axes, ("now", "random")):
        for i, (m, g) in enumerate(w.groupby("model", sort=False)):
            g = g.sort_values("W")
            x = np.arange(len(g)) + (i - 1) * 0.12
            y = g[f"save_vs_{b}_bp"].to_numpy()
            err = np.vstack([y - g[f"save_vs_{b}_ci_lo"].to_numpy(), g[f"save_vs_{b}_ci_hi"].to_numpy() - y])
            ax.errorbar(x, y, yerr=err, fmt="o-", color=colors[m], capsize=3, label=f"{LABEL[m]} (h={int(g['h'].iloc[0])} s)")
            prim = g["primary"].to_numpy()
            ax.plot(x[prim], y[prim], "s", ms=11, mfc="none", color=colors[m])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(WINDOWS)), [f"W={x}s" for x in WINDOWS])
        ax.set_title(f"saving vs executing {'at the window start' if b == 'now' else 'at a random second'}", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean saving per order, bp of arrival mid (test)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Execution timing with frozen order-flow models, test 2024-01-11 to 2024-03-30, 3 symbols "
                 "(square = validation-chosen W; bars = 95% daily block bootstrap)", fontsize=10)
    fig.tight_layout()
    fig.savefig(R / "fig_overlay.png", dpi=130)
    plt.close(fig)
    _summary_figure(plt, LABEL, t)
    log.info("overlay report written")


def _summary_figure(plt, LABEL, t: pd.DataFrame) -> None:
    """One figure for the README top: compute cost vs (a) net bp per trade as a strategy, (b) bp saved per order as an
    execution timer. Reads the main study's frozen test results and the overlay test results."""
    R = config.RESULTS
    info = json.loads((R / "train_info.json").read_text())
    tr = pd.read_csv(R / "test_results.csv")

    def us(name, h):
        ti = info[name]["timing"]
        return (ti[H.index(int(h))] if name == "lgbm" else ti[0])["us_single"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    for fee, col, lab in (("vip0", "#c0392b", "VIP0 fees (5 bp/side)"), ("cheap", "#e67e22", "3 bp/side"),
                          ("zero", "#2471a3", "zero fees")):
        c = tr[(tr["mode"] == "taker") & (tr["fee"] == fee) & (tr["latency_ms"] == 100)].copy()
        c["us"] = [us(m, h) for m, h in zip(c["model"], c["h"])]
        c = c.sort_values("us")
        ax.plot(c["us"], c["test_net_bps"], "o-", color=col, label=lab)
    for _, r in tr[(tr["mode"] == "taker") & (tr["fee"] == "vip0") & (tr["latency_ms"] == 100)].iterrows():
        ax.annotate(LABEL[r["model"]].split(" ", 1)[1], (us(r["model"], r["h"]), r["test_net_bps"]),
                    textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("CPU µs per prediction (1 row, 2 threads, log)")
    ax.set_ylabel("net bp per trade (test)")
    ax.set_title("(a) as a trading strategy (taker, 100 ms): nothing pays significantly with fees", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    for _, r in t.iterrows():
        x = us(r["model"], r["h"])
        y = r["save_vs_now_bp"]
        ax.errorbar([x], [y], yerr=[[y - r["save_vs_now_ci_lo"]], [r["save_vs_now_ci_hi"] - y]], fmt="o", capsize=4,
                    color="#1e8449")
        ax.annotate(f"{LABEL[r['model']].split(' ', 1)[1]} (W={int(r['W'])} s): {y:+.2f} bp", (x, y),
                    textcoords="offset points", xytext=(6, -12), fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(3, 200)
    ax.set_ylim(0, 0.4)
    ax.set_xlabel("CPU µs per prediction (1 row, 2 threads, log)")
    ax.set_ylabel("bp saved per order vs executing at once (test)")
    ax.set_title("(b) as an execution timer for a trade that must happen: every rung tested saves", fontsize=10)
    ax.grid(alpha=0.3)
    fig.suptitle("Same frozen order-flow models, Binance BTC/ETH/SOL perps, test 2024-01-11 to 2024-03-30 "
                 "(error bars: 95 % daily block bootstrap)", fontsize=10)
    fig.tight_layout()
    fig.savefig(R / "fig_summary.png", dpi=130)
    plt.close(fig)
