# type: ignore
"""Computations for plot_diff([df1, df2, ..., dfn])."""
from operator import index
import re
import ast
import pandas as pd
import numpy as np
import dask
import dask.array as da
import dask.dataframe as dd
from collections import UserList
from typing import Any, Callable, Dict, List, Tuple, Union, Optional

from pandas.core import base
from .common import DTMAP, _get_timeunit
from ...intermediate import Intermediate
from ...dtypes import (
    DType,
    Nominal,
    Continuous,
    DateTime,
    detect_dtype,
    get_dtype_cnts_and_num_cols,
    is_dtype,
    drop_null
)
from ....errors import UnreachableError
from ...configs import Config

# from dataprep.eda.diff.compute.common import DTMAP, _get_timeunit
# from dataprep.eda.configs import Config
# from dataprep.eda.intermediate import Intermediate
# from dataprep.eda.dtypes import DType, Nominal, Continuous, DateTime, detect_dtype, get_dtype_cnts_and_num_cols, is_dtype, drop_null
# from dataprep.errors import UnreachableError


class Dfs(UserList):
    """
    This class implements a sequence of DataFrames
    """

    def __init__(self, dfs: List[dd.DataFrame]) -> None:
        self.data = dfs

    def __getattr__(self, attr: str) -> UserList:
        output = []
        for df in self.data:
            output.append(getattr(df, attr))
        return Dfs(output)

    def apply(self, method: str) -> UserList:
        """
        Apply the same method for all elements in the list.
        """
        params = re.search(r"\((.*?)\)", method)
        if params:
            params = params.group(1)
        else:
            params = ""

        output = []
        for df in self.data:
            if len(params) > 0:
                method = method.replace(params, "").replace("()", "")
                try:
                    params = ast.literal_eval(params)
                except SyntaxError:
                    pass
                output.append(getattr(df, method)(params))
            else:
                output.append(getattr(df, method)())
        return Dfs(output)

    def getidx(self, ind: Union[str, int]) -> List[Any]:
        """
        Get the specified index for all elements in the list.
        """
        output = []
        for data in self.data:
            output.append(data[ind])
        return output

    def self_map(self, func: Callable[[dd.Series], Any], **kwargs: Any) -> List[Any]:
        """
        Map the data to the given function.
        """
        return [func(df, **kwargs) for df in self.data]


class Srs(UserList):
    """
    This class **separates** the columns with the same name into individual series.
    """

    def __init__(self, srs: Union[dd.DataFrame, List[Any]], agg: bool = False) -> None:
        if agg:
            self.data = srs
        else:
            if len(srs.shape) > 1:
                self.data: List[dd.Series] = [srs.iloc[:, loc] for loc in range(srs.shape[1])]
            else:
                self.data: List[dd.Series] = [srs]

    def __getattr__(self, attr: str) -> UserList:
        output = []
        for srs in self.data:
            output.append(getattr(srs, attr))
        return Srs(output, agg=True)

    def apply(self, method: str) -> UserList:
        """
        Apply the same method for all elements in the list.
        """
        params = re.search(r"\((.*?)\)", method)
        if params:
            params = str(params.group(1))
        else:
            params = ""

        output = []
        for srs in self.data:
            if len(params) > 0:
                method = method.replace(params, "").replace("()", "")
                if isinstance(params, str) and "=" not in params:
                    output.append(getattr(srs, method)(eval(params)))  # is it the only choice?
                else:
                    output.append(getattr(srs, method)(params))
            else:
                output.append(getattr(srs, method)())
        return Srs(output, agg=True)

    def getidx(self, ind: Union[str, int]) -> List[Any]:
        """
        Get the specified index for all elements in the list.
        """
        output = []
        for data in self.data:
            output.append(data[ind])

        return output

    def getmask(
        self, mask: Union[List[dd.Series], UserList], inverse: bool = False
    ) -> List[dd.Series]:
        """
        Return rows based on a boolean mask.
        """
        output = []
        for data, cond in zip(self.data, mask):
            if inverse:
                output.append(data[~cond])
            else:
                output.append(data[cond])

        return output

    def self_map(self, func: Callable[[dd.Series], Any], **kwargs: Any) -> List[Any]:
        """
        Map the data to the given function.
        """
        return [func(srs, **kwargs) for srs in self.data]


def compare_multiple_df(
    df_list: List[dd.DataFrame], cfg: Optional[Dict[str, Any]] = None
) -> Intermediate:
    """
    Compute function for plot_diff([df...])

    Parameters
    ----------
    dfs
        Dataframe sequence to be compared.
    """
    dfs = Dfs(df_list)
    baseline: int = cfg.diff.baseline

    data: List[Any] = []
    aligned_dfs = dd.concat(df_list, axis=1, copy=False)
    all_columns = set().union(*dfs.columns)

    # extract the first rows for checking if a column contains a mutable type
    first_rows = aligned_dfs.head()  # dd.DataFrame.head triggers a (small) data read

    for col in all_columns:
        srs = Srs(aligned_dfs[col])
        col_dtype = srs.self_map(detect_dtype)
        if len(col_dtype) > 1:
            col_dtype = col_dtype[baseline]
        else:
            col_dtype = col_dtype[0]

        if is_dtype(col_dtype, Continuous()) and cfg.hist.enable:
            data.append((col, Continuous(), _cont_calcs(srs.apply("dropna"), cfg)))
        elif is_dtype(col_dtype, Nominal()) and cfg.bar.enable:
            if len(first_rows[col].shape) > 1:  # exception for singular column
                try:
                    first_rows[col].iloc[:, baseline].apply(hash)
                except TypeError:
                    srs = srs.apply("dropna").apply("astype(str)")
            else:
                try:
                    first_rows[col].apply(hash)
                except TypeError:
                    srs = srs.apply("dropna").apply("astype(str)")
            data.append((col, Nominal(), _nom_calcs(srs.apply("dropna"), cfg)))
        elif is_dtype(col_dtype, DateTime()) and cfg.line.enable:
            data.append((col, DateTime(), dask.delayed(_calc_line_dt)(srs, col, cfg.line.unit)))
    stats = calc_stats(dfs, cfg)
    data = [dask.compute(*i) for i in data]
    stats = {k: list(dask.compute(*v)) for k, v in stats.items()}
    plot_data = []

    for col, dtp, datum in data:
        if is_dtype(dtp, Continuous()):
            if cfg.hist.enable:
                plot_data.append((col, dtp, datum["hist"]))
        elif is_dtype(dtp, Nominal()):
            if cfg.bar.enable:
                plot_data.append(
                    (col, dtp, (dask.compute(*datum["bar"]), len(aligned_dfs)))
                )
        elif is_dtype(dtp, DateTime()):
            plot_data.append((col, dtp, (dask.compute(*datum[0]), datum[1]))) # workaround
    return Intermediate(
        data=plot_data, stats=stats, visual_type="comparison_grid"
    )


def calc_stats(dfs: Dfs, cfg: Config) -> Dict[str, List[str]]:
    """
    Calculate the statistics for plot_diff([df1, df2, ..., dfn])

    Params
    ------
    dfs
        DataFrames to be compared
    """
    stats: Dict[str, List[Any]] = {"nrows": dfs.shape.getidx(0)}
    dtype_cnts = []
    num_cols = []
    if cfg.stats.enable:
        for df in dfs:
            temp = get_dtype_cnts_and_num_cols(df, dtype=None)
            dtype_cnts.append(temp[0])
            num_cols.append(temp[1])

        stats["ncols"] = dfs.shape.getidx(1)
        stats["npresent_cells"] = dfs.apply("count").apply("sum")
        stats["nrows_wo_dups"] = dfs.apply("drop_duplicates").shape.getidx(0)
        stats["mem_use"] = dfs.apply("memory_usage(deep=True)").apply("sum")
        stats["dtype_cnts"] = dtype_cnts

    return stats


def _cont_calcs(srs: Srs, cfg: Config) -> Dict[str, List[Any]]:
    """
    Computations for a continuous column in plot_diff([df1, df2, ..., dfn])
    """

    data: Dict[str, List[Any]] = {}

    # drop infinite values
    mask = srs.apply("isin({np.inf, -np.inf})")
    srs = Srs(srs.getmask(mask, inverse=True), agg=True)

    # histogram
    data["hist"] = srs.self_map(
        da.histogram,
        bins=cfg.hist.bins,
        range=(min(dask.compute(*srs.apply("min"))), max(dask.compute(*srs.apply("max")))),
    )

    return data


def _nom_calcs(srs: Srs, cfg: Config) -> Dict[str, List[Any]]:
    """
    Computations for a nominal column in plot_diff([df1, df2, ..., dfn])
    """
    # dictionary of data for the bar chart and related insights
    data: Dict[str, List[Any]] = {}
    data["bar"] = []
    baseline: int = cfg.diff.baseline

    # value counts for barchart and uniformity insight

    if cfg.bar.enable:
        if len(srs) > 1:
            grps = srs.apply("value_counts(sort=False)")
            # select the largest or smallest groups
            ngrp = grps[baseline].nlargest(cfg.bar.bars) if cfg.bar.sort_descending else grps[baseline].nsmallest(cfg.bar.bars)

            # pick data using indices from the baseline
            for i, grp in enumerate(grps):
                # if baseline has indices which are not in others
                if i != baseline:
                    tmp_srs = pd.Series(name=ngrp.name, dtype=int)
                    for tar_idx in ngrp.index:
                        try:
                            tmp_srs.loc[tar_idx] = dask.compute(*grp.loc[tar_idx])[0]
                        except: # bug?: can't intercept KeyError so we catch all errors
                            tmp_srs.loc[tar_idx] = 0

                    data["bar"].append(tmp_srs.to_frame())
            data["bar"].insert(baseline, ngrp.to_frame())
        else: # singular column
            grps = srs.apply("value_counts(sort=False)")[0]
            ngrp = grps.nlargest(cfg.bar.bars) if cfg.bar.sort_descending else grps.nsmallest(cfg.bar.bars)
            data["bar"].append(ngrp.to_frame())
    return data


def _calc_line_dt(
    srs: Srs,
    x: str,
    unit: str
) -> Tuple[List[pd.DataFrame], Dict[str, int]]:
    """
    Calculate a line or multiline chart with date on the x axis. If df contains
    one datetime column, it will make a line chart of the frequency of values.

    Parameters
    ----------
    df
        A dataframe
    x
        The column name
    unit
        The unit of time over which to group the values
    """
    # pylint: disable=too-many-locals
    unit = _get_timeunit(min(dask.compute(*srs.apply("min"))), min(dask.compute(*srs.apply("max"))), 100) if unit == "auto" else unit
    dfs = Dfs(srs.apply('to_frame').self_map(drop_null))
    if unit not in DTMAP.keys():
        raise ValueError
    grouper = pd.Grouper(key=x, freq=DTMAP[unit][0])  # for grouping the time values
    # miss_pct = round(df[x].isna().sum() / len(df) * 100, 1)
    for i in range(len(dfs)): # Dfs can't handle callable params
        dfs[i] = dfs[i].groupby(grouper)
    dfr = dfs.apply('size').apply('reset_index')

    for df in dfr:
        df.columns = [x, "freq"]
        df["pct"] = df["freq"] / len(df) * 100

        df[x] = df[x] - pd.to_timedelta(6, unit="d") if unit == "week" else df[x]
        df["lbl"] = df[x].dt.to_period("S").dt.strftime(DTMAP[unit][1])

    return (dfr, DTMAP[unit][3])

if __name__ == "__main__":
    df1 = pd.read_csv(
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    )
    df2 = df1.copy()

    df2["Age"] = df1["Age"] + 10
    df2["Extra"] = df1["Sex"]
    df3 = df1.iloc[:800, :]
    df = [df1, df2, df3]

    from dataprep.eda.utils import to_dask

    itmdt = compare_multiple_df(list(map(to_dask, df)), cfg=Config())
    print("EOF")
