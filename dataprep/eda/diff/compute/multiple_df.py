#type: ignore
"""Computations for plot_diff([df1, df2, ..., dfn])."""
import re
import ast
import pandas as pd
import dask
import dask.dataframe as dd
from collections import UserList
from typing import Any, Callable, Dict, List, Tuple
# from ...intermediate import Intermediate
# from ...dtypes import DType, Nominal, Continuous, DateTime, detect_dtype, get_dtype_cnts_and_num_cols, is_dtype
# from ....errors import UnreachableError

from dataprep.eda.intermediate import Intermediate
from dataprep.eda.dtypes import DType, Nominal, Continuous, DateTime, detect_dtype, get_dtype_cnts_and_num_cols, is_dtype
from dataprep.errors import UnreachableError

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


class Srs(UserList):
    """
    This class separates the columns with the same name into individual series.
    """
    def __init__(self, srs: dd.DataFrame) -> None:
        if len(srs.shape) > 1:
            self.data: List[dd.Series] = [srs.iloc[:, loc] for loc in range(srs.shape[1])]
        else:
            self.data: List[dd.Series] = [srs]

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
                except:
                    pass
                output.append(getattr(srs, method)(params))
            else:
                output.append(getattr(srs, method)())
        return output

    def self_map(self, func: Callable[[dd.Series], Any], **kwargs: Any) -> Any:
        """
        Map the data to the given function.
        """
        return [func(srs, **kwargs) for srs in self.data]


def compare_multiple_df(df_list: List[dd.DataFrame]) -> Intermediate:
    """
    Compute function for plot_diff([df...])

    Parameters
    ----------
    dfs
        Dataframe sequence to be compared.
    """
    dfs = Dfs(df_list)
    candidate_rank_idx = _get_candidate(dfs)


    datas: List[Any] = []
    aligned_dfs = pd.concat(Dfs(dfs),axis=1, copy=False)
    all_columns = set().union(*dfs.columns)

    # extract the first rows for checking if a column contains a mutable type
    first_rows = aligned_dfs.head()  # dd.DataFrame.head triggers a (small) data read

    for col in all_columns:
        srs = Srs(aligned_dfs[col])
        col_dtype = srs.self_map(detect_dtype)
        if len(col_dtype) > 1:
            col_dtype = col_dtype[candidate_rank_idx[1]] # use secondary for now
        else:
            col_dtype = col_dtype[0]

        if is_dtype(col_dtype, Nominal()):
            try:
                first_rows[col].apply(hash)
            except TypeError:
                srs = srs.apply("astype(str)")
            datas.append(calc_nom_col(srs.apply("dropna"), first_rows[col]))
        elif is_dtype(col_dtype, Continuous()):
            ## if cfg.hist_enable or cfg.any_insights("hist"):
            # datas.append(calc_cont_col(srs.apply("dropna"), bins))
            print(f"continues: {col}")
        elif is_dtype(col_dtype, DateTime()):
            # datas.append(dask.delayed(_calc_line_dt)(df[[col]], timeunit))
            print(f"dt: {col}")
        else:
            raise UnreachableError

    stats = calc_stats(dfs)

    stats = dask.compute(stats)


    return Intermediate(
        stats = stats,
        visual_type = "compare_multiple_dfs"
    )


def calc_stats(dfs: Dfs) -> Dict[str, List[str]]:
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


def calc_plot_data(df_list: List[dd.DataFrame]) -> List[Any]:
    pass


def _get_candidate(dfs: Dfs) -> List[int]:
    """
    The the index of major df from the candidates to determine the base for calculation.
    """
    dfs = dfs.apply('dropna')
    candidates = []

    dim: Dfs = dfs.shape
    major_candidate = dim.getidx(0)
    secondary_candidate = dim.getidx(1)

    candidates.append(major_candidate.index(max(major_candidate)))
    candidates.append(secondary_candidate.index(max(secondary_candidate)))

    #todo: there might be a better way to do this
    return candidates

if __name__ == "__main__":
    df1 = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    df2 = df1.copy()

    df2['Age'] = df1['Age'] + 10
    df2['Extra'] = df1['Sex']
    df3 = df1.iloc[:800, :]

    compare_multiple_df([df1, df2, df3])
