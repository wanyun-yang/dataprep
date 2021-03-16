"""Common types and functionalities for compute(...)."""

from typing import Any, Dict, List, Optional, Tuple, Union, cast

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde as gaussian_kde_
from scipy.stats import ks_2samp as ks_2samp_
from scipy.stats import normaltest as normaltest_
from scipy.stats import skewtest as skewtest_

# Dictionary for mapping the time unit to its formatting. Each entry is of the
# form unit:(unit code for pd.Grouper freq parameter, pandas to_period strftime
# formatting for line charts, pandas to_period strftime formatting for box plot,
# label format).
DTMAP = {
    "year": ("Y", "%Y", "%Y", "Year"),
    "quarter": ("Q", "Q%q %Y", "Q%q %Y", "Quarter"),
    "month": ("M", "%B %Y", "%b %Y", "Month"),
    "week": ("W-SAT", "%d %B, %Y", "%d %b, %Y", "Week of"),
    "day": ("D", "%d %B, %Y", "%d %b, %Y", "Date"),
    "hour": ("H", "%d %B, %Y, %I %p", "%d %b, %Y, %I %p", "Hour"),
    "minute": ("T", "%d %B, %Y, %I:%M %p", "%d %b, %Y, %I:%M %p", "Minute"),
    "second": ("S", "%d %B, %Y, %I:%M:%S %p", "%d %b, %Y, %I:%M:%S %p", "Second"),
}


def _get_timeunit(min_time: pd.Timestamp, max_time: pd.Timestamp, dflt: int) -> str:
    """Auxillary function to find an appropriate time unit. Will find the
    time unit such that the number of time units are closest to dflt."""

    dt_secs = {
        "year": 60 * 60 * 24 * 365,
        "quarter": 60 * 60 * 24 * 91,
        "month": 60 * 60 * 24 * 30,
        "week": 60 * 60 * 24 * 7,
        "day": 60 * 60 * 24,
        "hour": 60 * 60,
        "minute": 60,
        "second": 1,
    }

    time_rng_secs = (max_time - min_time).total_seconds()
    prev_bin_cnt, prev_unit = 0, "year"
    for unit, secs_in_unit in dt_secs.items():
        cur_bin_cnt = time_rng_secs / secs_in_unit
        if abs(prev_bin_cnt - dflt) < abs(cur_bin_cnt - dflt):
            return prev_unit
        prev_bin_cnt = cur_bin_cnt
        prev_unit = unit

    return prev_unit


@dask.delayed(name="scipy-normaltest", pure=True, nout=2)  # pylint: disable=no-value-for-parameter
def normaltest(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Delayed version of scipy normaltest. Due to the dask version will
    trigger a compute."""
    return cast(Tuple[np.ndarray, np.ndarray], normaltest_(arr))


@dask.delayed(name="scipy-ks_2samp", pure=True, nout=2)  # pylint: disable=no-value-for-parameter
def ks_2samp(data1: np.ndarray, data2: np.ndarray) -> Tuple[float, float]:
    """Delayed version of scipy ks_2samp."""
    return cast(Tuple[float, float], ks_2samp_(data1, data2))


@dask.delayed(  # pylint: disable=no-value-for-parameter
    name="scipy-gaussian_kde", pure=True, nout=2
)
def gaussian_kde(arr: np.ndarray) -> Tuple[float, float]:
    """Delayed version of scipy gaussian_kde."""
    return cast(Tuple[np.ndarray, np.ndarray], gaussian_kde_(arr))


@dask.delayed(name="scipy-skewtest", pure=True, nout=2)  # pylint: disable=no-value-for-parameter
def skewtest(arr: np.ndarray) -> Tuple[float, float]:
    """Delayed version of scipy skewtest."""
    return cast(Tuple[float, float], skewtest_(arr))
