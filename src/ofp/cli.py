"""Command line: python -m ofp.cli <stage>.

Stages, in order: download, build, train, validate, freeze, test, report; then the execution-timing overlay
(a later, separate analysis on the frozen models): overlay_validate, overlay_test, overlay_report. Each writes its outputs to files, so a stage can
be rerun alone. `test` runs once: it refuses to run again unless --force (which is logged).
"""

import argparse
import logging
import sys

from ofp import config  # noqa: F401  (pins thread counts before numpy/polars/torch load)


def setup_logging(stage: str) -> None:
    config.LOGS.mkdir(parents=True, exist_ok=True)
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(config.LOGS / f"{stage}.log", encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", handlers=handlers, force=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="ofp")
    p.add_argument("stage", choices=["download", "build", "train", "validate", "freeze", "test", "report",
                                      "overlay_validate", "overlay_test", "overlay_report"])
    p.add_argument("--symbols", default=None, help="comma-separated symbols (default: all in config)")
    p.add_argument("--max-days", type=int, default=None, help="download/build: only the first N sampled days (thin slice)")
    p.add_argument("--only", default=None, help="train: comma-separated rungs to (re)train")
    p.add_argument("--force", action="store_true", help="test stage: allow a second (logged) test run")
    a = p.parse_args(argv)
    setup_logging(a.stage)
    symbols = a.symbols.split(",") if a.symbols else None
    if a.stage == "download":
        from ofp import download
        download.run(symbols=symbols, max_days=a.max_days)
    elif a.stage == "build":
        from ofp import features
        features.build_all(symbols=symbols)
    elif a.stage == "train":
        from ofp import models
        models.train_all(only=a.only.split(",") if a.only else None)
    elif a.stage == "validate":
        from ofp import pipeline
        pipeline.validate()
    elif a.stage == "freeze":
        from ofp import pipeline
        pipeline.freeze()
    elif a.stage == "test":
        from ofp import pipeline
        pipeline.test(force=a.force)
    elif a.stage == "report":
        from ofp import report
        report.run()
    elif a.stage == "overlay_validate":  # new analysis after the main test: frozen models, cached forecasts
        from ofp import overlay
        overlay.validate()
    elif a.stage == "overlay_test":
        from ofp import overlay
        overlay.test(force=a.force)
    elif a.stage == "overlay_report":
        from ofp import overlay
        overlay.report()


if __name__ == "__main__":
    main()
