# type: ignore
"""Computations for plot_diff([df1, df2, ..., dfn])."""
import re
import ast
import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
from ...intermediate import Intermediate
from ...dtypes import DType, Nominal, detect_dtype, get_dtype_cnts_and_num_cols, is_dtype
from typing import Any, Callable, Dict, List
from collections import UserList

class Dfs(UserList[Any]):
    """
    This class implements a sequence of DataFrames
    """
    def __init__(self, dfs: List[dd.DataFrame]) -> None:
        self.data = dfs

    def __getattr__(self, attr: str) -> UserList[Any]:
        output = []
        for df in self.data:
            output.append(getattr(df, attr))
        return Dfs(output)

    def apply(self, method: str) -> UserList[Any]:
        """
        Apply the same method for all elements in the list.
        """
        params = re.search(r'\((.*?)\)', method)
        if params:
            params = params.group(1)
        else:
            params = ''

        output = []
        for df in self.data:
            if len(params) > 0:
                method = method.replace(params, '').replace('()', '')
                try:
                    params = ast.literal_eval(params)
                except SyntaxError:
                    pass
                output.append(getattr(df, method)(params))
            else:
                output.append(getattr(df, method)())
        return Dfs(output)

    def getidx(self, ind: int) -> List[Any]:
        """
        Get the specified index for all elements in the list.
        """
        output = []
        for data in self.data:
            output.append(data[ind])
        return output


class Srs(UserList[Any]):
    """
    This class separates the columns with the same name into individual series.
    """
    def __init__(self, srs: dd.DataFrame) -> None:
        self.data: List[dd.Series] = [srs.iloc[:, loc] for loc in range(srs.shape[1])]

    def __getattr__(self, attr: str) -> List[Any]:
        output = []
        for srs in self.data:
            output.append(getattr(srs, attr))
        return output

    def apply(self, method: str) -> List[Any]:
        """
        Apply the same method for all elements in the list.
        """
        params = re.search(r'\((.*?)\)', method)
        if params:
            params = params.group(1)
        else:
            params = ''

        output = []
        for srs in self.data:
            if len(params) > 0:
                method = method.replace(params, '').replace('()', '')
                try:
                    params = ast.literal_eval(params)
                except SyntaxError:
                    pass
                output.append(getattr(srs, method)(params))
            else:
                output.append(getattr(srs, method)())
        return output

    def self_map(self, func: Callable[dd.Series, Any], **kwargs: Any) -> Any:
        """
        Map the data to the given function.
        """
        return [func(srs, **kwargs) for srs in self.data]


def compare_multiple_df(dfs: List[dd.DataFrame]) -> Intermediate:
    """
    Compute function for plot_diff([df...])

    Parameters
    ----------
    dfs
        Dataframe sequence to be compared.
    """
    dfs = Dfs(dfs)
    # extract the first rows for checking if a column contains a mutable type
    first_rows = dfs.apply("head")  # dd.DataFrame.head triggers a (small) data read

    datas: List[Any] = []
    col_names_dtypes: List[Typle[str, DType]] = []
    aligned_dfs = pd.concat(Dfs(dfs),axis=1, copy=False)
    all_columns = set().union(*dfs.columns)
    for col in all_columns:
        srs = Srs(aligned_dfs[col])
        col_dtype = srs.self_map(detect_dtype)[0] # todo: use the first dtype for now
        if is_dtype(col_dtype, Nominal()):
            try:
                first_rows[col].apply(hash)
            except TypeError:
                srs = df[col] = srs.astype(str)
            datas.append(calc_nom_col(srs.dropna(), first_rows[col], ngroups, largest))
            col_names_dtypes.append((col, Nominal()))
        elif is_dtype(col_dtype, Continuous()):
            ## if cfg.hist_enable or cfg.any_insights("hist"):
            datas.append(calc_cont_col(srs.dropna(), bins))
            col_names_dtypes.append((col, Continuous()))
        elif is_dtype(col_dtype, DateTime()):
            datas.append(dask.delayed(_calc_line_dt)(df[[col]], timeunit))
            col_names_dtypes.append((col, DateTime()))
        else:
            raise UnreachableError

    stats = calc_stats_mul(dfs)




    stats = dask.compute(stats)


    return Intermediate(
        stats = stats
    )


def calc_stats_mul(dfs: Dfs) -> Dict[str, List[str]]:
    """
    Calculate the statistics for plot_diff([df1, df2, ..., dfn])

    Params
    ------
    dfs
        DataFrames to be compared
    """
    dtype_cnts = []
    num_cols = []
    for df in dfs:
        temp = get_dtype_cnts_and_num_cols(df, dtype=None)
        dtype_cnts.append(temp[0])
        num_cols.append(temp[1])

    stats: Dict[str, List[Any]] = {"nrows": dfs.shape.getidx(0)}

    stats["ncols"] = dfs.shape.getidx(1)
    stats["npresent_cells"] = dfs.apply("count")
    stats["nrows_wo_dups"] = dfs.apply("drop_duplicates").shape.getidx(0)
    stats["mem_use"] = dfs.apply("memory_usage(deep=True)").apply("sum")
    stats["dtype_cnts"] = dtype_cnts

    return stats


def calc_plot_data(dfs: List[dd.DataFrame]) -> List[Any]:
    pass

