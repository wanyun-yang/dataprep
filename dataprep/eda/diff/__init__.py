"""
    This module implements the plot_diff function.
"""

from typing import Optional, Union, List, Dict, Any
import dask.dataframe as dd
import pandas as pd

from ..configs import Config
from ..container import Container
from ..progress_bar import ProgressBar
from .compute import compute
from .render import render

__all__ = ["plot_diff"]

def plot_diff(
    df: Union[List[Union[pd.DataFrame, dd.DataFrame]], Union[pd.DataFrame, dd.DataFrame]],
    x: Optional[str] = None,
    window: Optional[List[str]] = None,
    progress: bool = True,
    config: Optional[Dict[str, Any]] = None,
    display: Optional[List[str]] = None,
) -> Container:
    """
    This function is to generate and render element in a report object.

    Parameters
    ----------
    df
        The DataFrame(s) to be compared.
    x
        The column to be emphasized in the comparision.
    window
        The window for seleceting range on continous and datetime variable,
        or the categories in categorical variable.
    progress
        Whether to show the progress bar.

    Examples
    --------
    >>> import pandas as pd
    >>> from dataprep.eda import plot_diff
    >>> plot_diff([df1, df2])
    >>> plot_diff([df1, df2], x)
    >>> plot_diff(df, x, ["cat_1", "cat_2"])
    >>> plot_diff(df, x, ["[0:100]", "[100:200]"])
    >>> plot_diff(df, x, ["[2020-01-01:2020-07-01]", "[2020-07-01:2021-01-01]"])
    """
    cfg = Config.from_dict(display, config)

    with ProgressBar(minimum=1, disable=not progress):
        intermediate = compute(
            df,
            x=x,
            window=window,
            cfg=cfg
        )
    to_render = render(intermediate, cfg=cfg)
    return Container(to_render, intermediate.visual_type, cfg)
