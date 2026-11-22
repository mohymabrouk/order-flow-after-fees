"""All constants of the study in one place. Every upper-case constant is part of the config hash in the test log.

Importing this module pins every numerical library to 2 threads (shared laptop budget).
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

from datetime import date, timedelta  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("P4_CACHE", Path.home() / ".cache" / "quant-research" / "p4"))
RAW_TMP = CACHE / "raw_tmp"  # zips live here only until their parquet is verified
BARS = CACHE / "bars"
FEATURES = CACHE / "features"
PREDS = CACHE / "preds"
MANIFEST = CACHE / "manifest.csv"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"

THREADS = 2
USER_AGENT = "Mozilla/5.0 (compatible; research-data-client)"
BASE_URL = "https://data.binance.vision/data/futures/um/daily"

# ---- data window and sampling --------------------------------------------------------------------
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
FIRST_DAY = date(2023, 5, 16)  # first USDT-M bookTicker daily file
LAST_DAY = date(2024, 3, 30)  # last USDT-M bookTicker daily file (S3 listing, 2026-11-21)
SAMPLE_WEEKDAYS = (1, 3, 5)  # Tuesday, Thursday, Saturday (Monday = 0)
TRAIN_END = date(2023, 11, 21)
GAP_1 = date(2023, 11, 22)
VAL_START = date(2023, 11, 23)
VAL_END = date(2024, 1, 8)
GAP_2 = date(2024, 1, 9)
TEST_START = date(2024, 1, 10)
RAW_BUDGET_GB = 60.0

# ---- bars, features, targets ---------------------------------------------------------------------
BUCKET_MS = 1000
SECONDS_PER_DAY = 86_400
LATENCY_OFFSET_MS = 100  # extra quote snapshot at bucket close + 100 ms
HORIZONS = (1, 5, 30, 60, 300)  # seconds
DEPTH_WINDOW = 300  # trailing window (s) of mean depth used to normalise OFI and signed volume
WARMUP = 300  # first seconds of each day dropped (rolling windows not full)
OFI_SUMS = (5, 30, 60, 300)
FLOW_WINDOWS = (5, 30)
RET_WINDOWS = (1, 5, 30, 60, 300)
VOL_WINDOWS = (60, 300)
SEQ_LEN = 30  # buckets seen by the sequence model

# ---- models ----------------------------------------------------------------------------------------
SEED = 20260401
TRAIN_SUBSAMPLE = 0.25  # fraction of train buckets used to fit rungs 2-5
LGB_TRAIN_ROWS = 1_500_000
LGB_LEAVES_GRID = (15, 63)
LGB_LR = 0.1
LGB_MAX_TREES = 300
LGB_MIN_LEAF = 2000
RIDGE_ALPHAS = (1.0, 100.0, 10_000.0)
TARGET_WINSOR = (0.0005, 0.9995)
MLP_HIDDEN = (64, 32)
MLP_EPOCHS = 6
MLP_BATCH = 4096
MLP_LR = 2e-3
CNN_CHANNELS = 16
CNN_HIDDEN = 32
CNN_TRAIN_WINDOWS = 1_500_000
CNN_EPOCHS = 8
CNN_BATCH = 1024
CNN_LR = 2e-3
VAL_EARLYSTOP_ROWS = 1_000_000

# ---- trading ----------------------------------------------------------------------------------------
LATENCIES_MS = (0, 100, 1000)
FEES = {  # (taker, maker) as fractions of notional. Levels as given in the brief, 2026-11-21.
    "vip0": (0.0005, 0.0002),
    "cheap": (0.0003, 0.0001),
    "zero": (0.0, 0.0),
}
FEE_SCHEDULE_NOTE = (
    "Binance USDT-M VIP0 taker 0.05% / maker 0.02% and a cheaper tier taker 0.03% / maker 0.01%, as specified in the task "
    "brief on 2026-11-21 (Binance's fee page is JS-rendered and could not be fetched to re-check); applies to the 2023-24 data."
)
THRESHOLD_MULTIPLES = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
MIN_VAL_TRADES = 50
MAIN_SCENARIO = ("taker", "vip0", 100)
DAYS_PER_YEAR = 365
MODELS_LADDER = ("sign_ofi", "linear_ofi", "ridge", "lgbm", "mlp", "cnn_lstm")


def day_split(d: date) -> str | None:
    if FIRST_DAY <= d <= TRAIN_END:
        return "train"
    if VAL_START <= d <= VAL_END:
        return "val"
    if TEST_START <= d <= LAST_DAY:
        return "test"
    return None  # gap days


def sampled_days() -> list[date]:
    out, d = [], FIRST_DAY
    while d <= LAST_DAY:
        if d.weekday() in SAMPLE_WEEKDAYS and day_split(d) is not None:
            out.append(d)
        d += timedelta(days=1)
    return out


def as_dict() -> dict:
    """Every upper-case constant (used for the config hash)."""
    skip = {"ROOT", "CACHE", "RAW_TMP", "BARS", "FEATURES", "PREDS", "MANIFEST", "RESULTS", "MODELS", "LOGS"}
    return {k: (str(v) if isinstance(v, (Path, date)) else v) for k, v in globals().items() if k.isupper() and k not in skip}
