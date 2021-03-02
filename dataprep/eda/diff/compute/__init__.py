"""Computations for plot_diff([df...])."""

from typing import Optional, Union, List, Dict, Any
import dask.dataframe as dd
import pandas as pd
from ...intermediate import Intermediate
from ...utils import to_dask
from .multiple_df import compare_multiple_df
from .singular import *

__all__ = ["compute"]

def compute(
    df: Union[List[Union[pd.DataFrame, dd.DataFrame]], Union[pd.DataFrame, dd.DataFrame]],
    x: Optional[str] = None,
    window: Optional[List[str]] = None,
    cfg: Optional[Dict[str, Any]] = None
) -> Intermediate:
    """
    blablabla
    """

    if isinstance(df, list):

        assert len(df) >= 2

        label = cfg.diff.label
        if not label:
            cfg.diff.label = [f"df{i+1}" for i in range(len(df))]
        elif len(df) != len(label):
            raise ValueError("Number of the given label doesn't match the number of DataFrames.")

        df_list = list(map(to_dask, df))
        if x:
            # return compare_multiple_on_column(df_list, x)
            pass
        else:
            return compare_multiple_df(df_list, cfg)

    # else:
    #     df = to_dask(df)
    #     df.columns = df.columns.astype(str)
    #     df = string_dtype_to_object(df)
