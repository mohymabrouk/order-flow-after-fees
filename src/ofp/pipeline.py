"""Validate, freeze and test stages.

validate: forecasts of every rung on the validation panel, then the full simulation grid
          (mode x fee x latency x rung x horizon x threshold multiple) -> results/validation_grid.csv
freeze:   per (rung, scenario) pick (horizon, multiple) on validation only -> results/frozen_config.json (+ SHA-256)
test:     one run on the test panel with the frozen config -> results/test_*.csv, logged to results/test_log.jsonl
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone

from ofp import config  # noqa: F401

import numpy as np
import pandas as pd
import polars as pl

from ofp import features as F
from ofp import models as M
from ofp import sim
from ofp.metrics import deflated_sharpe

log = logging.getLogger("pipeline")
H = list(config.HORIZONS)
MODES = ("taker", "maker")
PER_H_SCENARIOS = (config.MAIN_SCENARIO, ("taker", "zero", 100))


def scenarios():
    for mode in MODES:
        for fee in config.FEES:
            for lat in config.LATENCIES_MS:
                yield mode, fee, lat


def scen_key(mode, fee, lat) -> str:
    return f"{mode}|{fee}|{lat}"


def expected_cost_bps(mode: str, fee: str, spread_bps: float) -> float:
    taker, maker = config.FEES[fee]
    if mode == "taker":
        return 2 * taker * 1e4 + spread_bps
    return (maker + taker) * 1e4 + 0.5 * spread_bps


def code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((config.ROOT / "src" / "ofp").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def json_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------- data
def train_spreads() -> dict:
    """Mean quoted spread (bps) per symbol on valid train buckets: the spread part of the expected cost."""
    out = {}
    for s in config.SYMBOLS:
        vals = []
        for sym, d in F.available("train", [s]):
            vals.append(pl.scan_parquet(F.feature_path(sym, d)).filter(pl.col("valid")).select("spread_bps").collect())
        if vals:
            out[s] = float(pl.concat(vals)["spread_bps"].mean())
    return out


class Panel:
    def __init__(self, split: str):
        t0 = time.time()
        cols = F.FEATURES + F.TARGETS + F.SIM_COLS + ["lupd", "ltrd"]
        df, self.keys = F.load_panel(split, cols)
        assert df.height == len(self.keys) * config.SECONDS_PER_DAY
        self.n = df.height
        self.valid = df["valid"].to_numpy()
        self.k = df["k"].to_numpy()
        self.x = df.select(F.FEATURES).to_numpy().astype(np.float32)
        self.y = df.select(F.TARGETS).to_numpy().astype(np.float32)
        ch = df.select(F.SEQ_CHANNELS).to_numpy().astype(np.float32)
        self.ch = np.nan_to_num(ch, nan=0.0, posinf=0.0, neginf=0.0)
        self.book = sim.Book(*(df[c].to_numpy().astype(np.float64) for c in F.SIM_COLS))
        self.symbol = np.array([s for s, _ in self.keys])
        self.days = sorted({d for _, d in self.keys})
        self.seg_day = np.array([self.days.index(d) for _, d in self.keys])
        self.sid = df["sid"].to_numpy()
        del df
        log.info("%s panel: %d segments (%d days x symbols %s), %d rows, %d valid (%.0fs)", split, len(self.keys),
                 len(self.days), sorted(set(self.symbol)), self.n, int(self.valid.sum()), time.time() - t0)

    def predict(self, rung) -> np.ndarray:
        out = np.full((self.n, len(H)), np.nan, dtype=np.float32)
        if rung.sequence:
            rows = np.flatnonzero(self.valid & (self.k >= config.SEQ_LEN))
            out[rows] = rung.predict_rows(self.ch, rows)
        else:
            rows = np.flatnonzero(self.valid)
            for s in range(0, len(rows), 1_000_000):
                r = rows[s:s + 1_000_000]
                out[r] = rung.predict(self.x[r], self.sid[r])
        return out


def forecast_metrics(panel: Panel, preds: dict, split: str) -> pd.DataFrame:
    rows = []
    for name, p in preds.items():
        rung_has_mag = name != "sign_ofi"
        for i, h in enumerate(H):
            m = panel.valid
            rows.append({"split": split, "model": name, "h": h,
                         "r2": M.r2_zero(panel.y[m, i], p[m, i]) if rung_has_mag else np.nan,
                         "acc": M.accuracy(panel.y[m, i], p[m, i])})
    return pd.DataFrame(rows)


def run_sim(panel: Panel, pred: np.ndarray, name: str, mode: str, fee: str, lat: int, h: int, mult: float,
            spreads: dict, trials=None, trial_var=None, return_trades=False):
    taker, maker = config.FEES[fee]
    seg_cost = np.array([expected_cost_bps(mode, fee, spreads[s]) for s in panel.symbol])  # per (symbol, day) segment
    thr = 0.0 if name == "sign_ofi" else np.repeat(seg_cost, config.SECONDS_PER_DAY) * mult
    sig = pred[:, H.index(h)]
    sig = np.where(np.isfinite(sig), sig, 0.0)
    trades = sim.simulate(sig, panel.valid, panel.book, h, lat, mode, taker, maker, thr)
    summ = sim.summarize(trades, panel.seg_day, panel.days, n_segments=len(panel.keys), trials=trials, trial_var=trial_var)
    return (summ, trades) if return_trades else summ


def _mults(name: str):
    return (0.0,) if name == "sign_ofi" else config.THRESHOLD_MULTIPLES


def predictions(panel: Panel, split: str, names) -> dict:
    out = {}
    (config.PREDS / split).mkdir(parents=True, exist_ok=True)
    for name in names:
        t0 = time.time()
        rung = M.load_rung(name)
        p = panel.predict(rung)
        np.save(config.PREDS / split / f"{name}.npy", p)
        out[name] = p
        log.info("%s predictions on %s in %.0fs", name, split, time.time() - t0)
    return out


def available_rungs() -> list[str]:
    return [n for n in config.MODELS_LADDER if n == "sign_ofi" or M.model_path(n).exists()]


# ------------------------------------------------------------------------------------------------- stages
def validate() -> None:
    t0 = time.time()
    spreads = train_spreads()
    (config.RESULTS / "train_spreads.json").write_text(json.dumps(spreads, indent=1))
    log.info("train mean spreads (bps): %s", spreads)
    panel = Panel("val")
    names = available_rungs()
    preds = predictions(panel, "val", names)
    forecast_metrics(panel, preds, "val").to_csv(config.RESULTS / "val_forecast_metrics.csv", index=False)
    rows = []
    for mode, fee, lat in scenarios():
        t1 = time.time()
        for name in names:
            for h in H:
                for mult in _mults(name):
                    s = run_sim(panel, preds[name], name, mode, fee, lat, h, mult, spreads)
                    rows.append({"mode": mode, "fee": fee, "latency_ms": lat, "model": name, "h": h, "mult": mult, **s})
        log.info("validation grid %s done (%.0fs)", scen_key(mode, fee, lat), time.time() - t1)
        pd.DataFrame(rows).to_csv(config.RESULTS / "validation_grid.csv", index=False)
    log.info("validate done in %.0fs (%d grid rows)", time.time() - t0, len(rows))


def _n_hyper(name: str) -> int:
    return {"ridge": len(config.RIDGE_ALPHAS), "lgbm": len(config.LGB_LEAVES_GRID)}.get(name, 1)


def freeze() -> None:
    g = pd.read_csv(config.RESULTS / "validation_grid.csv")
    spreads = json.loads((config.RESULTS / "train_spreads.json").read_text())
    train_info = json.loads((config.RESULTS / "train_info.json").read_text())
    selections, per_h, trials = [], [], {}
    for (mode, fee, lat), gs in g.groupby(["mode", "fee", "latency_ms"], sort=False):
        key = scen_key(mode, fee, lat)
        n_trials = int(sum(_n_hyper(m) * len(H) * len(_mults(m)) for m in gs["model"].unique()))
        traded = gs[gs["trades"] >= config.MIN_VAL_TRADES]  # the configs that were real candidates
        if len(traded) < 2:
            traded = gs[gs["trades"] > 0]
        trials[key] = {"trials": n_trials, "trial_sharpe_var": float(traded["sharpe_pp"].var(ddof=1)) if len(traded) > 1 else 0.0,
                       "n_simulated": int(len(gs)), "n_in_variance": int(len(traded))}
        for name, gm in gs.groupby("model", sort=False):
            tg = gm[gm["trades"] >= config.MIN_VAL_TRADES]
            if len(tg) < 2:
                tg = gm[gm["trades"] > 0]
            trials[f"{key}|{name}"] = {"trials": int(_n_hyper(name) * len(H) * len(_mults(name))),
                                       "trial_sharpe_var": float(tg["sharpe_pp"].var(ddof=1)) if len(tg) > 1 else 0.0,
                                       "n_in_variance": int(len(tg))}
            elig = gm[gm["trades"] >= config.MIN_VAL_TRADES]
            pool = elig if len(elig) else gm
            best = pool.sort_values(["total_net_bps", "trades"], ascending=[False, False]).iloc[0]
            selections.append({"mode": mode, "fee": fee, "latency_ms": int(lat), "model": name, "h": int(best["h"]),
                               "mult": float(best["mult"]), "val_trades": int(best["trades"]),
                               "val_net_bps": float(best["net_bps"]), "val_total_net_bps": float(best["total_net_bps"]),
                               "val_sharpe_ann": float(best["sharpe_ann"]), "pays_on_val": bool(best["total_net_bps"] > 0),
                               "eligible": bool(len(elig) > 0)})
            if (mode, fee, int(lat)) in PER_H_SCENARIOS:
                for h, gh in gm.groupby("h"):
                    e = gh[gh["trades"] >= config.MIN_VAL_TRADES]
                    b = (e if len(e) else gh).sort_values(["total_net_bps", "trades"], ascending=[False, False]).iloc[0]
                    per_h.append({"mode": mode, "fee": fee, "latency_ms": int(lat), "model": name, "h": int(h),
                                  "mult": float(b["mult"]), "val_trades": int(b["trades"]), "val_net_bps": float(b["net_bps"])})
    frozen = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection_rule": f"max total validation net P&L with >= {config.MIN_VAL_TRADES} validation trades (else max over all)",
        "config": config.as_dict(),
        "train_spreads_bps": spreads,
        "hyperparameters": {k: {a: v[a] for a in ("alpha", "leaves") if a in v} for k, v in train_info.items() if not k.startswith("_")},
        "trials": trials,
        "selections": selections,
        "per_horizon_main": per_h,
    }
    frozen["hash"] = json_hash(frozen)
    (config.RESULTS / "frozen_config.json").write_text(json.dumps(frozen, indent=1))
    pd.DataFrame(selections).to_csv(config.RESULTS / "frozen_selections.csv", index=False)
    log.info("frozen config hash %s: %d selections", frozen["hash"], len(selections))


def _log_event(ev: dict) -> None:
    with open(config.RESULTS / "test_log.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(ev) + "\n")


def test(force: bool = False) -> None:
    log_path = config.RESULTS / "test_log.jsonl"
    if log_path.exists() and any(json.loads(line).get("event") == "done" for line in log_path.read_text().splitlines() if line.strip()):
        if not force:
            raise SystemExit("test already run once (see results/test_log.jsonl); use --force to rerun (it is logged)")
    frozen = json.loads((config.RESULTS / "frozen_config.json").read_text())
    h_saved = frozen.pop("hash")
    if json_hash(frozen) != h_saved:
        raise SystemExit("frozen config hash mismatch: the config was edited after freezing")
    t0 = time.time()
    base = {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "config_hash": h_saved,
            "code_hash": code_hash(), "forced_rerun": bool(force)}
    _log_event({"event": "start", **base})
    spreads = frozen["train_spreads_bps"]
    panel = Panel("test")
    names = sorted({s["model"] for s in frozen["selections"]}, key=list(config.MODELS_LADDER).index)
    preds = predictions(panel, "test", names)
    fm = forecast_metrics(panel, preds, "test")
    fm.to_csv(config.RESULTS / "test_forecast_metrics.csv", index=False)
    rows, daily = [], {}
    for sel in frozen["selections"]:
        key = scen_key(sel["mode"], sel["fee"], sel["latency_ms"])
        tr = frozen["trials"][key]
        s, trades = run_sim(panel, preds[sel["model"]], sel["model"], sel["mode"], sel["fee"], sel["latency_ms"], sel["h"],
                            sel["mult"], spreads, trials=tr["trials"], trial_var=tr["trial_sharpe_var"], return_trades=True)
        rows.append({**sel, **{f"test_{k}": v for k, v in s.items()}})
        rows[-1]["test_dsr_all_rungs"] = rows[-1].pop("test_dsr")
        rows[-1]["trials_all_rungs"], rows[-1]["trial_var_all_rungs"] = tr["trials"], tr["trial_sharpe_var"]
        trr = frozen["trials"][f"{key}|{sel['model']}"]
        pnl = sim.daily_pnl(trades, panel.seg_day, panel.days)
        rows[-1]["test_dsr"] = deflated_sharpe(pnl, trr["trials"], trr["trial_sharpe_var"])
        rows[-1]["trials_rung"], rows[-1]["trial_var_rung"] = trr["trials"], trr["trial_sharpe_var"]
        if (sel["mode"], sel["fee"], sel["latency_ms"]) in (config.MAIN_SCENARIO, ("taker", "zero", 100), ("taker", "cheap", 100),
                                                             ("maker", "vip0", 100)):
            daily[f"{sel['model']}|{key}"] = sim.daily_pnl(trades, panel.seg_day, panel.days)
            if (sel["mode"], sel["fee"], sel["latency_ms"]) == config.MAIN_SCENARIO:
                sym = panel.symbol[trades["idx"] // config.SECONDS_PER_DAY]
                for symb in sorted(set(panel.symbol)):
                    m = sym == symb
                    rows[-1][f"test_trades_{symb}"] = int(m.sum())
                    rows[-1][f"test_net_bps_{symb}"] = float(np.mean(trades["net_bps"][m])) if m.any() else np.nan
    pd.DataFrame(rows).to_csv(config.RESULTS / "test_results.csv", index=False)
    pd.DataFrame(daily).to_csv(config.RESULTS / "test_daily_pnl.csv")
    ph = []
    for item in frozen["per_horizon_main"]:
        tr = frozen["trials"][scen_key(item["mode"], item["fee"], item["latency_ms"])]
        s = run_sim(panel, preds[item["model"]], item["model"], item["mode"], item["fee"], item["latency_ms"], item["h"],
                    item["mult"], spreads, trials=tr["trials"], trial_var=tr["trial_sharpe_var"])
        ph.append({**item, **{f"test_{k}": v for k, v in s.items()}})
    pd.DataFrame(ph).merge(fm, on=["model", "h"], how="left").to_csv(config.RESULTS / "test_per_horizon.csv", index=False)
    _log_event({"event": "done", **base, "n_segments": len(panel.keys), "n_days": len(panel.days),
                "first_day": str(panel.days[0]), "last_day": str(panel.days[-1]), "seconds": round(time.time() - t0)})
    log.info("test done in %.0fs", time.time() - t0)
