"""Computations for plot_diff([df...])."""

from typing import Optional, Union, List
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
) -> Intermediate:
    """
    blablabla
    """

    if isinstance(df, list):

        assert len(df) >= 2

        # df_list = list(map(to_dask, df))
        df_list = df
        if x:
            # return compare_multiple_on_column(df_list, x)
            pass
        else:
            return compare_multiple_df(df_list)


    # else:
    #     df = to_dask(df)
    #     df.columns = df.columns.astype(str)
    #     df = string_dtype_to_object(df)
