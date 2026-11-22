"""Train stage: the model ladder. Every rung maps the same features (known at the end of bucket t) to forecasts of the mid
change in bps over the horizons in config.HORIZONS. Rung objects are pickled to models/<name>.pkl.
"""

from __future__ import annotations

import json
import logging
import pickle
import time

from ofp import config  # noqa: F401  (thread pins first)

import lightgbm as lgb
import numpy as np
import polars as pl
import torch
from torch import nn

from ofp import features as F

torch.set_num_threads(config.THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

log = logging.getLogger("models")
H = list(config.HORIZONS)
NH = len(H)
CLIP = 10.0


# ------------------------------------------------------------------------------------------------- helpers
class Scaler:
    def __init__(self, x: np.ndarray):
        self.mu = np.nanmean(x, axis=0).astype(np.float32)
        sd = np.nanstd(x, axis=0).astype(np.float32)
        self.sd = np.where(sd > 0, sd, 1.0).astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mu) / self.sd, -CLIP, CLIP).astype(np.float32)


def r2_zero(y: np.ndarray, p: np.ndarray) -> float:
    """Out-of-sample R^2 against a zero forecast (the natural benchmark for returns)."""
    m = np.isfinite(y) & np.isfinite(p)
    return float(1 - np.sum((y[m] - p[m]) ** 2) / np.sum(y[m] ** 2))


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    """Directional accuracy on rows where the mid actually moved and the forecast is non-zero."""
    m = np.isfinite(y) & np.isfinite(p) & (y != 0) & (p != 0)
    return float(np.mean(np.sign(y[m]) == np.sign(p[m]))) if m.any() else float("nan")


def winsor_limits(y: np.ndarray) -> np.ndarray:
    lo, hi = config.TARGET_WINSOR
    return np.stack([np.nanquantile(y, lo, axis=0), np.nanquantile(y, hi, axis=0)])


# ------------------------------------------------------------------------------------------------- rungs
class Rung:
    name = "base"
    sequence = False
    has_magnitude = True

    def predict(self, x: np.ndarray, sid=None) -> np.ndarray:  # (n, n_features) raw features -> (n, NH) bps
        raise NotImplementedError

    def n_params(self, h_idx: int) -> int:
        raise NotImplementedError


class SignOFI(Rung):
    name = "sign_ofi"
    has_magnitude = False

    def predict(self, x, sid=None):
        return np.repeat(np.sign(x[:, [F.FEATURES.index("ofi_n")]]), NH, axis=1).astype(np.float32)

    def n_params(self, h_idx):
        return 0


class LinearOFI(Rung):
    name = "linear_ofi"

    def fit(self, x, y, scaler):
        j = F.FEATURES.index("ofi_n")
        self.mu, self.sd = scaler.mu[j], scaler.sd[j]
        z = np.clip((x[:, j] - self.mu) / self.sd, -CLIP, CLIP)
        self.a, self.b = np.zeros(NH), np.zeros(NH)
        for i in range(NH):
            m = np.isfinite(y[:, i])
            self.b[i], self.a[i] = np.polyfit(z[m], y[m, i], 1)
        return self

    def predict(self, x, sid=None):
        z = np.clip((x[:, F.FEATURES.index("ofi_n")] - self.mu) / self.sd, -CLIP, CLIP)
        return (self.a[None, :] + z[:, None] * self.b[None, :]).astype(np.float32)

    def n_params(self, h_idx):
        return 2


class Ridge(Rung):
    name = "ridge"

    def fit(self, x, y, scaler, xv, yv):
        self.scaler = scaler
        z = scaler(x)
        zm, ym = z.mean(0), np.nanmean(y, 0)
        zc = (z - zm).astype(np.float64)
        gram = zc.T @ zc
        self.W = np.zeros((z.shape[1], NH))
        self.alpha = [None] * NH
        zv = scaler(xv)
        self.grid = []
        for i in range(NH):
            m = np.isfinite(y[:, i])
            xty = zc[m].T @ (y[m, i] - ym[i])
            best = None
            for a in config.RIDGE_ALPHAS:
                w = np.linalg.solve(gram + a * np.eye(len(gram)), xty)
                pv = ym[i] + (zv - zm) @ w
                r2 = r2_zero(yv[:, i], pv)
                self.grid.append({"h": H[i], "alpha": a, "val_r2": r2})
                if best is None or r2 > best[0]:
                    best = (r2, a, w)
            self.W[:, i], self.alpha[i] = best[2], best[1]
        self.b = ym - zm @ self.W
        return self

    def predict(self, x, sid=None):
        return (self.scaler(x) @ self.W + self.b).astype(np.float32)

    def n_params(self, h_idx):
        return self.W.shape[0] + 1


class PerSymbol(Rung):
    """One copy of a linear rung per symbol, as Cont-Kukanov-Stoikov estimate one price-impact slope per instrument
    (in bps, the impact of a unit of OFI/depth scales with the tick size, which differs a lot across symbols)."""

    def __init__(self, name: str, inner: dict):
        self.name, self.inner = name, inner

    def predict(self, x, sid=None):
        out = np.full((len(x), NH), np.nan, dtype=np.float32)
        for s, r in self.inner.items():
            m = sid == s
            if m.any():
                out[m] = r.predict(x[m])
        return out

    def n_params(self, h_idx):
        return int(sum(r.n_params(h_idx) for r in self.inner.values()))

    @property
    def grid(self):
        return [{"sym": config.SYMBOLS[s], **g} for s, r in self.inner.items() for g in getattr(r, "grid", [])]

    @property
    def alpha(self):
        return {config.SYMBOLS[s]: r.alpha for s, r in self.inner.items() if hasattr(r, "alpha")}


class LGBM(Rung):
    name = "lgbm"

    def fit(self, x, y, xv, yv):
        self.boosters, self.leaves, self.grid = [], [], []
        for i in range(NH):
            m, mv = np.isfinite(y[:, i]), np.isfinite(yv[:, i])
            best = None
            for leaves in config.LGB_LEAVES_GRID:
                params = {"objective": "regression", "learning_rate": config.LGB_LR, "num_leaves": leaves,
                          "min_data_in_leaf": config.LGB_MIN_LEAF, "feature_fraction": 0.8, "bagging_fraction": 0.5,
                          "bagging_freq": 1, "lambda_l2": 10.0, "max_bin": 63, "num_threads": config.THREADS,
                          "verbose": -1, "seed": config.SEED, "deterministic": True, "force_row_wise": True}
                dtr = lgb.Dataset(x[m], y[m, i], free_raw_data=True)
                dva = lgb.Dataset(xv[mv], yv[mv, i], reference=dtr)
                t0 = time.time()
                bst = lgb.train(params, dtr, num_boost_round=config.LGB_MAX_TREES, valid_sets=[dva],
                                callbacks=[lgb.early_stopping(30, verbose=False)])
                pv = bst.predict(xv, num_threads=config.THREADS, num_iteration=bst.best_iteration)
                r2 = r2_zero(yv[:, i], pv)
                self.grid.append({"h": H[i], "leaves": leaves, "trees": bst.best_iteration, "val_r2": r2,
                                  "secs": round(time.time() - t0, 1)})
                log.info("lgbm h=%d leaves=%d trees=%d val R2 %.5f (%.0fs)", H[i], leaves, bst.best_iteration, r2,
                         time.time() - t0)
                if best is None or r2 > best[0]:
                    best = (r2, leaves, bst)
            bst = best[2]
            self.boosters.append(bst.model_to_string(num_iteration=bst.best_iteration))
            self.leaves.append(best[1])
        self._load()
        return self

    def _load(self):
        self._b = [lgb.Booster(model_str=s) for s in self.boosters]

    def __getstate__(self):
        d = dict(self.__dict__)
        d.pop("_b", None)
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._load()

    def predict(self, x, sid=None, horizons=None):
        idx = range(NH) if horizons is None else horizons
        out = np.full((len(x), NH), np.nan, dtype=np.float32)
        for i in idx:
            out[:, i] = self._b[i].predict(x, num_threads=config.THREADS)
        return out

    def n_params(self, h_idx):
        dump = self._b[h_idx].dump_model()
        return int(sum(2 * t["num_leaves"] - 1 for t in dump["tree_info"]))


class MLPNet(nn.Module):
    def __init__(self, n_in, hidden, n_out):
        super().__init__()
        layers, d = [], n_in
        for hdim in hidden:
            layers += [nn.Linear(d, hdim), nn.ReLU()]
            d = hdim
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _train_torch(model, make_batch, n_train, xv_t, yv_t, epochs, batch, lr, seed, tag):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    best, best_state, history = np.inf, None, []
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n_train)
        t0, tot, nb = time.time(), 0.0, 0
        for s in range(0, n_train, batch):
            xb, yb = make_batch(perm[s:s + batch])
            m = torch.isfinite(yb)
            loss = ((model(xb) - torch.nan_to_num(yb)) ** 2 * m).sum() / m.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        model.eval()
        with torch.no_grad():
            vl = []
            for s in range(0, len(yv_t), 65536):
                pv = model(xv_t[s:s + 65536]) if not callable(xv_t) else model(xv_t(s, s + 65536))
                yb = yv_t[s:s + 65536]
                m = torch.isfinite(yb)
                vl.append((((pv - torch.nan_to_num(yb)) ** 2 * m).sum() / m.sum()).item())
            v = float(np.mean(vl))
        history.append({"epoch": ep + 1, "train_loss": tot / nb, "val_loss": v, "secs": round(time.time() - t0, 1)})
        log.info("%s epoch %d train %.5f val %.5f (%.0fs)", tag, ep + 1, tot / nb, v, time.time() - t0)
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
        else:
            break  # early stopping, patience 1
    model.load_state_dict(best_state)
    return history


class MLP(Rung):
    name = "mlp"

    def fit(self, x, y, scaler, xv, yv):
        self.scaler = scaler
        lim = winsor_limits(y)
        self.ysd = np.nanstd(np.clip(y, lim[0], lim[1]), axis=0).astype(np.float32)
        z = torch.from_numpy(scaler(x))
        yt = torch.from_numpy((np.clip(y, lim[0], lim[1]) / self.ysd).astype(np.float32))
        zv = torch.from_numpy(scaler(xv))
        yvt = torch.from_numpy((yv / self.ysd).astype(np.float32))
        self.net = MLPNet(z.shape[1], config.MLP_HIDDEN, NH)
        self.history = _train_torch(self.net, lambda idx: (z[idx], yt[idx]), len(z), zv, yvt, config.MLP_EPOCHS,
                                    config.MLP_BATCH, config.MLP_LR, config.SEED, "mlp")
        self.net.eval()
        return self

    @torch.no_grad()
    def predict(self, x, sid=None):
        out = []
        for s in range(0, len(x), 262144):
            out.append(self.net(torch.from_numpy(self.scaler(x[s:s + 262144]))).numpy())
        return (np.concatenate(out) * self.ysd).astype(np.float32)

    def n_params(self, h_idx):
        return int(sum(p.numel() for p in self.net.parameters()))


class SeqNet(nn.Module):
    """DeepLOB-style: two temporal convolutions, then an LSTM, then a linear head (all horizons at once)."""

    def __init__(self, c_in, ch, hid, n_out):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(c_in, ch, 3), nn.LeakyReLU(0.01), nn.Conv1d(ch, ch, 3), nn.LeakyReLU(0.01))
        self.lstm = nn.LSTM(ch, hid, batch_first=True)
        self.head = nn.Linear(hid, n_out)

    def forward(self, x):  # (B, K, C)
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        _, (h, _) = self.lstm(z)
        return self.head(h[-1])


def windows(arr: np.ndarray, rows: np.ndarray, k: int = config.SEQ_LEN) -> np.ndarray:
    """(len(rows), k, C) windows ending at each row (inclusive)."""
    return arr[rows[:, None] + np.arange(-k + 1, 1)[None, :]]


class CNNLSTM(Rung):
    name = "cnn_lstm"
    sequence = True

    def fit(self, ch_tr, y_tr, rows_tr, ch_va, y_va, rows_va):
        self.mu = ch_tr[rows_tr].mean(0).astype(np.float32)
        sd = ch_tr[rows_tr].std(0)
        self.sd = np.where(sd > 0, sd, 1).astype(np.float32)
        lim = winsor_limits(y_tr[rows_tr])
        self.ysd = np.nanstd(np.clip(y_tr[rows_tr], lim[0], lim[1]), axis=0).astype(np.float32)
        ztr = torch.from_numpy(self._z(ch_tr))
        ytr = torch.from_numpy((np.clip(y_tr, lim[0], lim[1]) / self.ysd).astype(np.float32))
        zva = torch.from_numpy(self._z(ch_va))
        yva = torch.from_numpy((y_va[rows_va] / self.ysd).astype(np.float32))
        rt, rv = torch.from_numpy(rows_tr), torch.from_numpy(rows_va)
        off = torch.arange(-config.SEQ_LEN + 1, 1)

        def make_batch(idx):
            r = rt[idx]
            return ztr[r[:, None] + off], ytr[r]

        def val_x(s, e):
            r = rv[s:e]
            return zva[r[:, None] + off]

        self.net = SeqNet(len(F.SEQ_CHANNELS), config.CNN_CHANNELS, config.CNN_HIDDEN, NH)
        self.history = _train_torch(self.net, make_batch, len(rows_tr), val_x, yva, config.CNN_EPOCHS, config.CNN_BATCH,
                                    config.CNN_LR, config.SEED, "cnn_lstm")
        self.net.eval()
        return self

    def _z(self, ch):
        return np.clip((ch - self.mu) / self.sd, -CLIP, CLIP).astype(np.float32)

    @torch.no_grad()
    def predict_rows(self, ch: np.ndarray, rows: np.ndarray) -> np.ndarray:
        z = torch.from_numpy(self._z(ch))
        off = torch.arange(-config.SEQ_LEN + 1, 1)
        out = []
        for s in range(0, len(rows), 16384):
            r = torch.from_numpy(rows[s:s + 16384])
            out.append(self.net(z[r[:, None] + off]).numpy())
        return (np.concatenate(out) * self.ysd).astype(np.float32) if out else np.zeros((0, NH), np.float32)

    @torch.no_grad()
    def predict_window(self, win: np.ndarray) -> np.ndarray:  # (B, K, C) raw channels
        return (self.net(torch.from_numpy(self._z(win))).numpy() * self.ysd).astype(np.float32)

    def n_params(self, h_idx):
        return int(sum(p.numel() for p in self.net.parameters()))


# ------------------------------------------------------------------------------------------------- timing
def time_predictions(rung: Rung, x: np.ndarray, ch: np.ndarray | None = None, rows: np.ndarray | None = None,
                     h_idx: int = 0, n_single: int = 2000, n_batch: int = 10_000, sid=None) -> dict:
    """CPU microseconds per prediction: single-row calls and amortised over a 10,000-row batch (2 threads).
    A per-symbol rung is timed through one symbol's model, as a live per-symbol process would call it."""
    if isinstance(rung, PerSymbol):
        first = sorted(rung.inner)[0]
        rung, x = rung.inner[first], (x if sid is None else x[sid == first])
        sid = None
    if rung.sequence:
        wins = windows(ch, rows[: max(n_single, n_batch)])
        single = lambda i: rung.predict_window(wins[i:i + 1])  # noqa: E731
        batch = lambda: rung.predict_window(wins[:n_batch])  # noqa: E731
    elif isinstance(rung, LGBM):
        single = lambda i: rung.predict(x[i:i + 1], horizons=[h_idx])  # noqa: E731
        batch = lambda: rung.predict(x[:n_batch], horizons=[h_idx])  # noqa: E731
    else:
        sid = np.zeros(len(x), dtype=np.int8) if sid is None else sid
        single = lambda i: rung.predict(x[i:i + 1], sid[i:i + 1])  # noqa: E731
        batch = lambda: rung.predict(x[:n_batch], sid[:n_batch])  # noqa: E731
    for i in range(50):
        single(i)
    t0 = time.perf_counter()
    for i in range(n_single):
        single(i)
    us_single = (time.perf_counter() - t0) / n_single * 1e6
    batch()
    reps = 3
    t0 = time.perf_counter()
    for _ in range(reps):
        batch()
    us_batch = (time.perf_counter() - t0) / reps / n_batch * 1e6
    return {"us_single": us_single, "us_batch": us_batch}


# ------------------------------------------------------------------------------------------------- data
def load_xy(split: str, subsample: float | None, max_rows: int | None = None, seed: int = config.SEED):
    df, _ = F.load_panel(split, F.FEATURES + F.TARGETS, only_valid=True, subsample=subsample, seed=seed)
    if max_rows and df.height > max_rows:
        df = df.sample(n=max_rows, seed=seed)
    x = df.select(F.FEATURES).to_numpy().astype(np.float32)
    y = df.select(F.TARGETS).to_numpy().astype(np.float32)
    return x, y, df["sid"].to_numpy()


def load_seq(split: str, every_other_day: bool, max_windows: int | None, seed: int = config.SEED):
    """Full-day channel arrays + target arrays and the rows usable as window ends (valid, all targets finite)."""
    df, keys = F.load_panel(split, F.SEQ_CHANNELS + F.TARGETS)
    if every_other_day:
        df = df.filter(pl.col("seg") % 2 == 0)
    ch = df.select(F.SEQ_CHANNELS).to_numpy().astype(np.float32)
    ch = np.nan_to_num(ch, nan=0.0, posinf=0.0, neginf=0.0)
    y = df.select(F.TARGETS).to_numpy().astype(np.float32)
    ok = df["valid"].to_numpy() & np.isfinite(y).all(1) & (df["k"].to_numpy() >= config.SEQ_LEN)
    rows = np.flatnonzero(ok)
    if max_windows and len(rows) > max_windows:
        rows = np.sort(np.random.default_rng(seed).choice(rows, max_windows, replace=False))
    return ch, y, rows


# ------------------------------------------------------------------------------------------------- stage
def model_path(name: str):
    return config.MODELS / f"{name}.pkl"


def load_rung(name: str) -> Rung:
    if name == "sign_ofi":
        return SignOFI()
    with open(model_path(name), "rb") as fh:
        return pickle.load(fh)


def train_all(only=None) -> None:
    config.MODELS.mkdir(parents=True, exist_ok=True)
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    names = only or list(config.MODELS_LADDER)
    t0 = time.time()
    x, y, sid = load_xy("train", subsample=config.TRAIN_SUBSAMPLE)
    xv, yv, sidv = load_xy("val", subsample=None, max_rows=config.VAL_EARLYSTOP_ROWS)
    log.info("train rows %d, val early-stopping rows %d (%.0fs)", len(x), len(xv), time.time() - t0)
    lim = winsor_limits(y)
    yw = np.clip(y, lim[0], lim[1])  # winsorised targets for fitting only
    scaler = Scaler(x)
    info_path = config.RESULTS / "train_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info["_data"] = {"train_rows": int(len(x)), "val_es_rows": int(len(xv)), "features": F.FEATURES,
                     "winsor_limits": lim.tolist()}
    for name in names:
        t1 = time.time()
        if name == "sign_ofi":
            rung = SignOFI()
        elif name == "linear_ofi":
            rung = PerSymbol(name, {s: LinearOFI().fit(x[sid == s], yw[sid == s], scaler) for s in np.unique(sid)})
        elif name == "ridge":
            rung = PerSymbol(name, {s: Ridge().fit(x[sid == s], yw[sid == s], scaler, xv[sidv == s], yv[sidv == s])
                                    for s in np.unique(sid)})
        elif name == "lgbm":
            pick = np.sort(np.random.default_rng(config.SEED).choice(len(x), min(len(x), config.LGB_TRAIN_ROWS), replace=False))
            rung = LGBM().fit(x[pick], yw[pick], xv, yv)
        elif name == "mlp":
            rung = MLP().fit(x, yw, scaler, xv, yv)
        elif name == "cnn_lstm":
            ch, ys, rows = load_seq("train", every_other_day=True, max_windows=config.CNN_TRAIN_WINDOWS)
            chv, ysv, rowsv = load_seq("val", every_other_day=False, max_windows=config.VAL_EARLYSTOP_ROWS // 4)
            log.info("cnn windows: train %d, val %d", len(rows), len(rowsv))
            rung = CNNLSTM().fit(ch, ys, rows, chv, ysv, rowsv)
        else:
            raise ValueError(name)
        secs = time.time() - t1
        if name != "sign_ofi":
            with open(model_path(name), "wb") as fh:
                pickle.dump(rung, fh)
        # forecast quality on the validation early-stopping sample + timing
        if rung.sequence:
            pv = rung.predict_rows(chv, rowsv)
            yq = ysv[rowsv]
            timing = time_predictions(rung, None, ch=chv, rows=rowsv)
            del ch, ys
        else:
            pv = rung.predict(xv, sidv)
            yq = yv
            timing = {}
        entry = {"train_secs": round(secs, 1), "n_params": [rung.n_params(i) for i in range(NH)],
                 "val_r2": [r2_zero(yq[:, i], pv[:, i]) if rung.has_magnitude else None for i in range(NH)],
                 "val_acc": [accuracy(yq[:, i], pv[:, i]) for i in range(NH)]}
        if not rung.sequence:
            entry["timing"] = [time_predictions(rung, xv, h_idx=i, sid=sidv) for i in (range(NH) if isinstance(rung, LGBM) else [0])]
        else:
            entry["timing"] = [timing]
        for attr in ("grid", "history", "alpha", "leaves"):
            if hasattr(rung, attr):
                entry[attr] = getattr(rung, attr)
        info[name] = entry
        info_path.write_text(json.dumps(info, indent=1, default=float))
        log.info("%s trained in %.0fs; params %s; val R2 %s; acc %s; timing %s", name, secs, entry["n_params"],
                 np.round([v if v is not None else np.nan for v in entry["val_r2"]], 5), np.round(entry["val_acc"], 3),
                 entry["timing"][0])
    log.info("train stage done in %.0fs", time.time() - t0)
