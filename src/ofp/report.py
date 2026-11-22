"""Report stage: tables (results/tables.md, CSVs) and figures from the results files only."""

from __future__ import annotations

import json
import logging

from ofp import config  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

log = logging.getLogger("report")
H = list(config.HORIZONS)
LABEL = {"sign_ofi": "1 sign(OFI)", "linear_ofi": "2 linear CKS OFI", "ridge": "3 ridge", "lgbm": "4 LightGBM",
         "mlp": "5 MLP", "cnn_lstm": "6 CNN+LSTM"}
FEE_LABEL = {"vip0": "VIP0 5.0/2.0 bp", "cheap": "cheap 3.0/1.0 bp", "zero": "zero"}


def _fmt(v, nd=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return f"{v:,.{nd}f}"


def md_table(df: pd.DataFrame, fmts: dict | None = None) -> str:
    fmts = fmts or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, str):
                cells.append(v)
            else:
                cells.append(_fmt(v, fmts.get(c, 2)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _psr(daily: pd.DataFrame, name: str, mode: str, fee: str, lat: int) -> float:
    """Probabilistic Sharpe ratio vs 0 (the copied deflated_sharpe with a single trial), from the saved daily P&L."""
    from ofp.metrics import deflated_sharpe

    col = f"{name}|{mode}|{fee}|{lat}"
    return deflated_sharpe(daily[col], 1, 0.0) if col in daily.columns else np.nan


def run() -> None:
    R = config.RESULTS
    info = json.loads((R / "train_info.json").read_text())
    frozen = json.loads((R / "frozen_config.json").read_text())
    tr = pd.read_csv(R / "test_results.csv")
    fm = pd.read_csv(R / "test_forecast_metrics.csv")
    ph = pd.read_csv(R / "test_per_horizon.csv")
    daily = pd.read_csv(R / "test_daily_pnl.csv", index_col=0)
    test_log = [json.loads(x) for x in (R / "test_log.jsonl").read_text().splitlines() if x.strip()]
    out = []
    mode, fee, lat = config.MAIN_SCENARIO
    main = tr[(tr["mode"] == mode) & (tr["fee"] == fee) & (tr["latency_ms"] == lat)].copy()
    rows = []
    for _, r in main.iterrows():
        name = r["model"]
        hi = H.index(int(r["h"]))
        ti = info[name]
        timing = ti["timing"][hi] if name == "lgbm" else ti["timing"][0]
        f = fm[(fm["model"] == name) & (fm["h"] == r["h"])].iloc[0]
        rows.append({
            "model": LABEL[name], "params": int(ti["n_params"][hi]), "µs/pred (1 row)": timing["us_single"],
            "µs/pred (batch)": timing["us_batch"], "train s": ti["train_secs"], "h (s)": int(r["h"]),
            "thr x cost": r["mult"], "R² (test)": f["r2"] * 100 if np.isfinite(f["r2"]) else np.nan,
            "acc (test)": f["acc"] * 100, "trades": int(r["test_trades"]), "gross bps": r["test_gross_bps"],
            "net bps": r["test_net_bps"], "hit %": r["test_hit_rate"] * 100, "daily net bps": r["test_daily_mean_bps"],
            "Sharpe (ann.)": r["test_sharpe_ann"], "PSR": _psr(daily, name, mode, fee, lat), "DSR": r["test_dsr"],
            "DSR all rungs": r["test_dsr_all_rungs"],
            "val net bps": r["val_net_bps"],
        })
    main_tbl = pd.DataFrame(rows)
    main_tbl.to_csv(R / "main_table.csv", index=False)
    tk = frozen["trials"][f"{mode}|{fee}|{lat}"]
    out.append(f"## Main table: taker, {FEE_LABEL[fee]} (taker/maker), latency {lat} ms, test period\n")
    out.append(f"R² is out-of-sample vs a zero forecast, in %; accuracy is directional on non-zero mid moves, in %. "
               f"DSR: deflated Sharpe with the rung's own validation grid as the trials (30 configs; ridge and LightGBM "
               f"count their hyperparameter grid too; sign(OFI) 5), trial Sharpe variance from that rung's validation "
               f"configs with >= {config.MIN_VAL_TRADES} trades. DSR all rungs: the whole cell's family, "
               f"{tk['trials']} trials, variance {tk['trial_sharpe_var']:.4g} (per day). PSR: probabilistic Sharpe vs 0 "
               f"(no deflation). "
               f"`thr x cost` is the validation-chosen threshold multiple of expected cost.\n")
    out.append(md_table(main_tbl, {"µs/pred (1 row)": 1, "µs/pred (batch)": 3, "train s": 0, "thr x cost": 2, "R² (test)": 3,
                                   "acc (test)": 1, "hit %": 1, "PSR": 3, "DSR": 3, "DSR all rungs": 3, "Sharpe (ann.)": 2}))

    # ---- latency x fee sensitivity, per mode: net bps per trade, trades, Sharpe ----
    for m in ("taker", "maker"):
        sub = tr[tr["mode"] == m]
        out.append(f"\n## Sensitivity, {m}{' (OPTIMISTIC fill model)' if m == 'maker' else ''}: test net bps per trade "
                   f"[trades] (annualised Sharpe); each cell has its own validation-chosen horizon and threshold\n")
        recs = []
        for name in config.MODELS_LADDER:
            if name not in set(sub["model"]):
                continue
            rec = {"model": LABEL[name]}
            for fee_ in config.FEES:
                for lat_ in config.LATENCIES_MS:
                    c = sub[(sub["model"] == name) & (sub["fee"] == fee_) & (sub["latency_ms"] == lat_)]
                    if len(c):
                        c = c.iloc[0]
                        rec[f"{fee_} / {lat_}ms"] = (f"{_fmt(c['test_net_bps'])} [{int(c['test_trades']):,}] "
                                                     f"({_fmt(c['test_sharpe_ann'], 1)}) h={int(c['h'])}")
            recs.append(rec)
        t = pd.DataFrame(recs)
        t.to_csv(R / f"sensitivity_{m}.csv", index=False)
        out.append(md_table(t))

    # ---- daily P&L sums table ----
    sens = tr[["mode", "fee", "latency_ms", "model", "h", "mult", "pays_on_val", "test_trades", "test_net_bps",
               "test_total_net_bps", "test_sharpe_ann", "test_dsr", "test_turnover"]].copy()
    sens.to_csv(R / "sensitivity_long.csv", index=False)

    # ---- per-horizon ----
    for (m, fe, la), g in ph.groupby(["mode", "fee", "latency_ms"], sort=False):
        out.append(f"\n## Per horizon, {m}, {FEE_LABEL[fe]}, {la} ms (threshold chosen on validation per horizon): "
                   f"R² %, accuracy %, trades, net bps, Sharpe\n")
        recs = []
        for name in config.MODELS_LADDER:
            gg = g[g["model"] == name]
            if not len(gg):
                continue
            rec = {"model": LABEL[name]}
            for _, r in gg.sort_values("h").iterrows():
                r2 = "–" if not np.isfinite(r["r2"]) else f"{r['r2'] * 100:.2f}"
                rec[f"h={int(r['h'])}s"] = (f"R² {r2} / acc {r['acc'] * 100:.1f} / {int(r['test_trades']):,} tr / "
                                           f"{_fmt(r['test_net_bps'])} bp / SR {_fmt(r['test_sharpe_ann'], 1)}")
            recs.append(rec)
        out.append(md_table(pd.DataFrame(recs)))

    # ---- break-even fee: edge per trade at zero fees vs the fee it would have to pay ----
    tk = tr[(tr["mode"] == "taker") & (tr["latency_ms"] == 100)]
    if len(tk):
        out.append("\n## Edge before fees (test, taker, 100 ms): gross − spread, bps per trade, of the strategy chosen on "
                   "validation for each fee level [trades]. Half of it is the largest per-side fee that strategy could pay "
                   "(VIP0 charges 5.0 bp per side, the cheap tier 3.0 bp).\n")
        recs = []
        for name in config.MODELS_LADDER:
            rec = {"model": LABEL[name]}
            for fee_ in config.FEES:
                c = tk[(tk["model"] == name) & (tk["fee"] == fee_)]
                if not len(c):
                    continue
                c = c.iloc[0]
                pre = c["test_gross_bps"] - c["test_spread_bps"]
                rec[f"chosen for {fee_}"] = f"{_fmt(pre, 3)} [{int(c['test_trades']):,}] h={int(c['h'])}"
                rec[f"break-even/side ({fee_})"] = pre / 2 if np.isfinite(pre) else np.nan
            recs.append(rec)
        be = pd.DataFrame(recs)
        be.to_csv(R / "edge_before_fees.csv", index=False)
        out.append(md_table(be, {c: 3 for c in be.columns if c.startswith("break-even")}))

    # ---- forecast metrics, all horizons ----
    fmw = fm.pivot(index="model", columns="h", values=["r2", "acc"])
    out.append("\n## Forecast quality on test (all valid buckets): R² % vs zero forecast | directional accuracy %\n")
    recs = []
    for name in config.MODELS_LADDER:
        if name not in fmw.index:
            continue
        rec = {"model": LABEL[name]}
        for h in H:
            r2 = fmw.loc[name, ("r2", h)]
            rec[f"h={h}s"] = f"{'–' if not np.isfinite(r2) else f'{r2 * 100:.2f}'} | {fmw.loc[name, ('acc', h)] * 100:.1f}"
        recs.append(rec)
    out.append(md_table(pd.DataFrame(recs)))

    # ---- per-symbol at main scenario ----
    sym_cols = [c for c in main.columns if c.startswith("test_net_bps_") or c.startswith("test_trades_")]
    if sym_cols:
        out.append("\n## Main scenario by symbol (test): net bps per trade [trades]\n")
        recs = []
        syms = sorted({c.split("_")[-1] for c in sym_cols})
        for _, r in main.iterrows():
            rec = {"model": LABEL[r["model"]]}
            for s in syms:
                rec[s] = f"{_fmt(r.get(f'test_net_bps_{s}'))} [{int(r.get(f'test_trades_{s}', 0)):,}]"
            recs.append(rec)
        out.append(md_table(pd.DataFrame(recs)))

    # ---- uncertainty: bootstrap over test days of the mean daily net P&L ----
    rng = np.random.default_rng(config.SEED)
    recs = []
    for c in daily.columns:
        x = daily[c].to_numpy(dtype=float)
        bs = rng.choice(x, size=(5000, len(x)), replace=True).mean(axis=1)
        name, scen = c.split("|", 1)
        recs.append({"model": LABEL[name], "scenario": scen.replace("|", ", ") + " ms", "mean daily net bps": x.mean(),
                     "95% CI low": np.quantile(bs, 0.025), "95% CI high": np.quantile(bs, 0.975),
                     "positive days %": 100 * np.mean(x > 0), "PSR": _psr(daily, name, *scen.split("|")[:2], int(scen.split("|")[2])),
                     "days": len(x)})
    ci = pd.DataFrame(recs)
    ci.to_csv(R / "daily_pnl_bootstrap.csv", index=False)
    out.append("\n## Mean daily net P&L on test with a 95% bootstrap CI over days (5,000 resamples; bps of notional, "
               "summed over symbols)\n")
    out.append(md_table(ci, {"positive days %": 0, "PSR": 3}))

    out.append("\n## Test log\n")
    for e in test_log:
        out.append(f"- {e['event']} {e['timestamp_utc']} config {e['config_hash']} code {e['code_hash']} forced={e['forced_rerun']}")
    (R / "tables.md").write_text("\n".join(out) + "\n", encoding="utf-8")

    # ---- figures ----
    _frontier(main_tbl, tr, info)
    _daily(daily)
    _accuracy_vs_profit(ph)
    log.info("report written: %s", R / "tables.md")


def _frontier(main_tbl, tr, info):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = {"vip0": "#c0392b", "cheap": "#e67e22", "zero": "#2471a3"}
    for ax, metric, ylab in ((axes[0], "test_net_bps", "net bps per trade (test)"),
                             (axes[1], "test_daily_mean_bps", "mean daily net P&L, bps of notional (test)")):
        for fee in config.FEES:
            xs, ys, names = [], [], []
            for name in config.MODELS_LADDER:
                c = tr[(tr["mode"] == "taker") & (tr["fee"] == fee) & (tr["latency_ms"] == 100) & (tr["model"] == name)]
                if not len(c):
                    continue
                c = c.iloc[0]
                hi = H.index(int(c["h"]))
                t = info[name]["timing"][hi] if name == "lgbm" else info[name]["timing"][0]
                xs.append(t["us_single"])
                ys.append(c[metric])
                names.append(name)
            order = np.argsort(xs)
            xs_, ys_ = np.array(xs)[order], np.array(ys, dtype=float)[order]
            notrade = ~np.isfinite(ys_)
            ys_[notrade] = 0.0  # a rung that never trades earns exactly 0
            ax.plot(xs_, ys_, "-", color=colors[fee], alpha=0.7)
            ax.plot(xs_[~notrade], ys_[~notrade], "o", color=colors[fee], label=f"taker, {FEE_LABEL[fee]}")
            if notrade.any():
                ax.plot(xs_[notrade], ys_[notrade], "o", mfc="none", color=colors[fee], ms=9,
                        label=f"no trade (taker, {FEE_LABEL[fee]})")
            for x, y, n in zip(xs, ys, names):
                if fee == "zero":
                    ax.annotate(LABEL[n].split(" ", 1)[1], (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xscale("log")
        if "daily" in metric:
            ax.set_yscale("symlog", linthresh=30)
        ax.set_xlabel("CPU µs per prediction (single row, 2 threads, log scale)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Compute cost vs net profit, latency 100 ms (each point: validation-chosen horizon and threshold)")
    fig.tight_layout()
    fig.savefig(config.RESULTS / "fig_frontier.png", dpi=130)
    plt.close(fig)


def _daily(daily: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    panels = ((axes[0], ("taker|zero|100",), "zero fees (taker, 100 ms): every rung"),
              (axes[1], ("taker|vip0|100", "taker|cheap|100", "maker|vip0|100"),
               "with fees (100 ms), rungs 3-6 (rungs 1-2 lose 400-8,500 bp/day)"))
    styles = {"taker|vip0|100": "-", "taker|cheap|100": "--", "maker|vip0|100": ":", "taker|zero|100": "-"}
    colors = dict(zip(config.MODELS_LADDER, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]))
    for ax, scens, title in panels:
        for c in daily.columns:
            name, scen = c.split("|", 1)
            if scen not in scens or (len(scens) > 1 and name in ("sign_ofi", "linear_ofi")):
                continue
            ax.plot(range(len(daily)), daily[c].cumsum(), styles[scen], color=colors[name],
                    label=f"{LABEL[name]} ({scen.replace('|', ', ')} ms)")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("test day (sampled Tue/Thu/Sat, 2024-01-11 to 2024-03-30)")
        ax.set_ylabel("cumulative net P&L, bps of notional (sum of 3 symbols)")
        ax.legend(fontsize=6, ncol=1 if len(scens) == 1 else 2)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.RESULTS / "fig_cum_pnl.png", dpi=130)
    plt.close(fig)


def _accuracy_vs_profit(ph: pd.DataFrame):
    g = ph[(ph["fee"] == "vip0")]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for name in config.MODELS_LADDER:
        gg = g[g["model"] == name]
        if len(gg):
            ax.scatter(gg["acc"] * 100, gg["test_net_bps"], label=LABEL[name], s=25)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("directional accuracy on test, % (non-zero moves)")
    ax.set_ylabel("net bps per trade, taker VIP0, 100 ms")
    ax.set_title("Accuracy is not profit (one point per model x horizon)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.RESULTS / "fig_accuracy_vs_profit.png", dpi=130)
    plt.close(fig)
