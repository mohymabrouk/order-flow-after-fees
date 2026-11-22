"""Download stage: per (symbol, day) fetch bookTicker + aggTrades zips with their .CHECKSUM, verify SHA-256, build the
1-second bar parquet, verify it, then delete the zips. Resumable: a day whose verified parquet exists is skipped.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import polars as pl
import requests

from ofp import bars as B
from ofp import config

log = logging.getLogger("download")
SESSION = requests.Session()
SESSION.headers["User-Agent"] = config.USER_AGENT

MANIFEST_FIELDS = [
    "symbol", "day", "split", "bt_bytes", "at_bytes", "n_quotes_raw", "n_quotes_out_of_day", "n_quotes_in_bars",
    "n_trades_raw", "n_trades_out_of_day", "n_trades_in_bars", "qty_raw", "qty_in_day", "qty_in_bars", "n_empty_buckets",
    "uid_sorted", "ts_sorted", "secs_download", "secs_build", "written_utc",
]


def bar_path(symbol: str, day) -> "config.Path":
    return config.BARS / symbol / f"{symbol}_{day.isoformat()}.parquet"


def _fetch(url: str, dest, retries: int = 4) -> int:
    for attempt in range(retries):
        try:
            with SESSION.get(url, stream=True, timeout=60) as r:
                if r.status_code == 404:
                    raise FileNotFoundError(url)
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                n = 0
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                        n += len(chunk)
                tmp.replace(dest)
                return n
        except FileNotFoundError:
            raise
        except Exception as exc:  # network hiccup: retry with backoff
            log.warning("fetch failed (%s), attempt %d: %s", url, attempt + 1, exc)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"giving up on {url}")


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_verified_zip(kind: str, symbol: str, day) -> tuple["config.Path", int]:
    name = f"{symbol}-{kind}-{day.isoformat()}.zip"
    url = f"{config.BASE_URL}/{kind}/{symbol}/{name}"
    dest = config.RAW_TMP / name
    cks = SESSION.get(url + ".CHECKSUM", timeout=60)
    cks.raise_for_status()
    expected = cks.text.split()[0].strip().lower()
    if not (dest.exists() and _sha256(dest) == expected):
        _fetch(url, dest)
    got = _sha256(dest)
    if got != expected:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {name}: {got} != {expected}")
    return dest, dest.stat().st_size


def _read_zip(path) -> bytes:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        return zf.read(names[0])


def manifest_rows() -> list[dict]:
    if not config.MANIFEST.exists():
        return []
    with open(config.MANIFEST, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _append_manifest(row: dict) -> None:
    new = not config.MANIFEST.exists()
    with open(config.MANIFEST, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in MANIFEST_FIELDS})


def fetch_day(symbol: str, day) -> tuple:
    t0 = time.time()
    bt = _get_verified_zip("bookTicker", symbol, day)
    at = _get_verified_zip("aggTrades", symbol, day)
    return bt, at, time.time() - t0


def process_day(symbol: str, day, fetched=None) -> dict:
    (bt_zip, bt_bytes), (at_zip, at_bytes), secs_dl = fetched or fetch_day(symbol, day)
    t1 = time.time()
    quotes = B.read_quotes_csv(_read_zip(bt_zip))
    trades = B.read_trades_csv(_read_zip(at_zip))
    day_start = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)
    bars, stats = B.make_bars(quotes, trades, day_start)
    del quotes, trades
    out = bar_path(symbol, day)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    bars.write_parquet(tmp, compression="zstd")
    # ---- verify the parquet before deleting the raw zips ----
    back = pl.read_parquet(tmp)
    ok = (
        back.height == config.SECONDS_PER_DAY
        and int(back["n_upd"].sum()) == stats["n_quotes_raw"] - stats["n_quotes_out_of_day"]
        and int(back["buy_n"].sum() + back["sell_n"].sum()) == stats["n_trades_raw"] - stats["n_trades_out_of_day"]
        and abs(float(back["buy_qty"].sum() + back["sell_qty"].sum()) - stats["qty_in_day"]) <= 1e-9 * max(1.0, stats["qty_in_day"])
        and stats["n_quotes_out_of_day"] < 0.001 * stats["n_quotes_raw"] + 10
    )
    if not ok:
        raise RuntimeError(f"bar verification failed for {symbol} {day}: {stats}")
    tmp.replace(out)
    bt_zip.unlink()
    at_zip.unlink()
    row = {"symbol": symbol, "day": day.isoformat(), "split": config.day_split(day), "bt_bytes": bt_bytes,
           "at_bytes": at_bytes, **stats, "secs_download": round(secs_dl, 1), "secs_build": round(time.time() - t1, 1),
           "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _append_manifest(row)
    return row


def run(symbols=None, max_days: int | None = None, days=None) -> None:
    config.RAW_TMP.mkdir(parents=True, exist_ok=True)
    symbols = symbols or list(config.SYMBOLS)
    done = {(r["symbol"], r["day"]) for r in manifest_rows()}
    used = sum(int(r["bt_bytes"]) + int(r["at_bytes"]) for r in manifest_rows())
    all_days = days or config.sampled_days()
    for symbol in symbols:
        todo = [d for d in all_days if (symbol, d.isoformat()) not in done or not bar_path(symbol, d).exists()]
        if max_days:
            todo = todo[:max_days]
        log.info("%s: %d days to process (%d sampled in total); raw GB so far %.2f", symbol, len(todo), len(all_days), used / 1e9)
        pool = ThreadPoolExecutor(max_workers=1)  # prefetch the next day's zips while this day is parsed
        nxt = pool.submit(fetch_day, symbol, todo[0]) if todo else None
        for i, d in enumerate(todo):
            if used / 1e9 > config.RAW_BUDGET_GB:
                log.error("raw download budget of %.0f GB reached; stopping", config.RAW_BUDGET_GB)
                pool.shutdown(wait=True)
                return
            fut = nxt
            nxt = pool.submit(fetch_day, symbol, todo[i + 1]) if i + 1 < len(todo) else None
            try:
                row = process_day(symbol, d, fetched=fut.result())
            except FileNotFoundError as exc:
                log.warning("missing file, day skipped: %s", exc)
                continue
            used += int(row["bt_bytes"]) + int(row["at_bytes"])
            log.info("%s %s [%d/%d] quotes %d trades %d empty %d dl %.0fs build %.0fs raw GB %.2f", symbol, d, i + 1,
                     len(todo), row["n_quotes_raw"], row["n_trades_raw"], row["n_empty_buckets"], row["secs_download"],
                     row["secs_build"], used / 1e9)
