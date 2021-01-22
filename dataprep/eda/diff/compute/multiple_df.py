from ...intermediate import Intermediate
from typing import Any, Dict, List
from collections import UserList
from ...dtypes import get_dtype_cnts_and_num_cols
import dask.dataframe as dd
import re
import ast
import dask


def compare_multiple_df(dfs: List[dd.DataFrame]) -> Intermediate:
    stats = calc_stats_mul(dfs)
    return stats

def calc_stats_mul(dfs: List[dd.DataFrame]) -> Dict[str, List[str]]:
    """
    Calculate the statistics for plot_diff([df1, df2...dfn])

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
    dfs = Dfs(dfs)

    stats = {"nrows": dfs.shape._getidx(0)}

    stats["ncols"] = dfs.shape._getidx(1)
    stats["npresent_cells"] = dfs._call("count")
    stats["nrows_wo_dups"] = dfs._call("drop_duplicates").shape._getidx(0)
    stats["mem_use"] = dfs._call("memory_usage(deep=True)")._call("sum")
    stats["dtype_cnts"] = dtype_cnts

    return stats

class Dfs(UserList[Any]):
    """
    This class implements a sequence of DataFrames
    """
    def __init__(self, dfs: List[Any]) -> None:
        super().__init__(dfs)

    def __getattr__(self, attr: str) -> object:
        output = []
        for df in self.data:
            output.append(getattr(df, attr))
        return Dfs(output)

    def _call(self, method: str) -> List[Any]:
        try:
            params = re.search(r'\((.*?)\)', method).group(1)
        except AttributeError:
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

    def _getidx(self, ind: int) -> Any:
        output = []
        for d in self.data:
            output.append(d[ind])
        return output