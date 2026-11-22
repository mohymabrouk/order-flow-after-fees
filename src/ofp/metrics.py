"""Sharpe and deflated Sharpe.

`sharpe` and `deflated_sharpe` are COPIED (not imported) from Alpha Audit,
C:/dev/quant-agent-lab/backend/src/qal/lab/metrics.py (the Alpha Audit project), unchanged except for this header.
The test suite checks that they return the same numbers as the originals on fixed inputs.
"""

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


# ---- copied from qal/lab/metrics.py -------------------------------------------------------------------
def sharpe(returns: pd.Series, periods_per_year: int) -> float:
    r = returns.dropna()
    sd = r.std(ddof=1)
    if len(r) < 2 or sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(r.mean() / sd * math.sqrt(periods_per_year))


def deflated_sharpe(returns: pd.Series, trials: int, trial_sharpe_variance: float) -> float:
    """Probability that the strategy's true Sharpe is above what the best of `trials` random strategies would show
    (Bailey & López de Prado, 2014). Sharpe ratios here are per period (not annualised), as in the paper.

    `trial_sharpe_variance` is the variance of the per-period Sharpe ratios of all trials in the family.
    With fewer than two trials the benchmark is zero, which is the probabilistic Sharpe ratio.
    """
    x = returns.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 3 or x.std(ddof=1) == 0:
        return 0.0
    sr = float(x.mean() / x.std(ddof=1))
    # Sample skewness and (non-excess) kurtosis, as in the paper.
    z_scores = (x - x.mean()) / x.std()
    skew = float(np.mean(z_scores**3))
    kurtosis = float(np.mean(z_scores**4))
    if trials >= 2 and trial_sharpe_variance > 0:
        expected_max = math.sqrt(trial_sharpe_variance) * (
            (1 - EULER_GAMMA) * _NORMAL.inv_cdf(1 - 1 / trials)
            + EULER_GAMMA * _NORMAL.inv_cdf(1 - 1 / (trials * math.e))
        )
    else:
        expected_max = 0.0
    denominator = 1 - skew * sr + (kurtosis - 1) / 4 * sr**2
    if denominator <= 0:
        return 0.0
    z = (sr - expected_max) * math.sqrt(n - 1) / math.sqrt(denominator)
    return float(_NORMAL.cdf(z))
# ---- end of copied code ------------------------------------------------------------------------------------


def per_period_sharpe(returns: pd.Series) -> float:
    x = returns.dropna().to_numpy(dtype=float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / x.std(ddof=1))
