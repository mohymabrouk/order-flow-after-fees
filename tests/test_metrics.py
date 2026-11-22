"""Test 6: the copied Sharpe and deflated Sharpe match the Alpha Audit functions on fixed inputs.

The Alpha Audit package is never imported: the test reads its source file, extracts the two function definitions with
`ast` and executes them in an isolated namespace, then compares with our copies.
"""

import ast
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from ofp import metrics

SRC = Path("C:/dev/quant-agent-lab/backend/src/qal/lab/metrics.py")


def _original_functions():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    wanted = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("sharpe", "deflated_sharpe")]
    consts = [n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("EULER_GAMMA", "_NORMAL") for t in n.targets)]
    mod = ast.Module(body=consts + wanted, type_ignores=[])
    ns = {"math": math, "np": np, "pd": pd, "NormalDist": NormalDist}
    exec(compile(mod, str(SRC), "exec"), ns)
    return ns["sharpe"], ns["deflated_sharpe"]


@pytest.mark.skipif(not SRC.exists(), reason="Alpha Audit source not on this machine")
def test_same_numbers_as_alpha_audit():
    sharpe0, dsr0 = _original_functions()
    rng = np.random.default_rng(7)
    for i in range(5):
        r = pd.Series(rng.standard_t(4, 250) * 0.01 + 0.0005 * i)
        r.iloc[3] = np.nan
        assert metrics.sharpe(r, 365) == sharpe0(r, 365)
        assert metrics.sharpe(r, 252) == sharpe0(r, 252)
        for trials, var in [(1, 0.0), (10, 0.002), (205, 0.01)]:
            assert metrics.deflated_sharpe(r, trials, var) == dsr0(r, trials, var)
    flat = pd.Series([0.0] * 10)
    assert metrics.sharpe(flat, 365) == sharpe0(flat, 365) == 0.0
    assert metrics.deflated_sharpe(flat, 5, 0.1) == dsr0(flat, 5, 0.1) == 0.0


def test_fixed_values():
    r = pd.Series([0.01, -0.005, 0.02, 0.0, 0.004, -0.01, 0.015])
    x = r.to_numpy()
    assert metrics.sharpe(r, 365) == pytest.approx(x.mean() / x.std(ddof=1) * math.sqrt(365))
    assert 0.0 < metrics.deflated_sharpe(r, 1, 0.0) < 1.0
    assert metrics.deflated_sharpe(r, 100, 0.05) < metrics.deflated_sharpe(r, 1, 0.0)
