"""
This module implements the visualization for the plot_diff function.
"""  # pylint: disable=too-many-lines
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from bokeh.layouts import row
from bokeh.models import (
    BasicTicker,
    ColorBar,
    ColumnDataSource,
    CustomJSHover,
    FactorRange,
    FuncTickFormatter,
    HoverTool,
    LayoutDOM,
    Legend,
    LegendItem,
    LinearColorMapper,
    Panel,
    PrintfTickFormatter,
)
from bokeh.plotting import Figure, figure
from bokeh.transform import cumsum, linear_cmap, transform
from bokeh.util.hex import hexbin
from pandas.core import base
from scipy.stats import norm
from wordcloud import WordCloud

from ..configs import Config
from ..dtypes import Continuous, DateTime, Nominal, is_dtype
from ..intermediate import Intermediate
from ..palette import CATEGORY20, PASTEL1, RDBU, VIRIDIS

__all__ = ["render"]


def tweak_figure(
    fig: Figure,
    ptype: Optional[str] = None,
    show_yticks: bool = False,
    max_lbl_len: int = 15,
) -> None:
    """
    Set some common attributes for a figure
    """
    fig.axis.major_label_text_font_size = "9pt"
    fig.title.text_font_size = "10pt"
    fig.axis.minor_tick_line_color = "white"
    if ptype in ["pie", "qq", "heatmap"]:
        fig.ygrid.grid_line_color = None
    if ptype in ["bar", "pie", "hist", "kde", "qq", "heatmap", "line"]:
        fig.xgrid.grid_line_color = None
    if ptype in ["bar", "hist", "line"] and not show_yticks:
        fig.ygrid.grid_line_color = None
        fig.yaxis.major_label_text_font_size = "0pt"
        fig.yaxis.major_tick_line_color = None
    if ptype in ["bar", "nested", "stacked", "heatmap", "box"]:
        fig.xaxis.major_label_orientation = np.pi / 3
        fig.xaxis.formatter = FuncTickFormatter(
            code="""
            if (tick.length > %d) return tick.substring(0, %d-2) + '...';
            else return tick;
        """
            % (max_lbl_len, max_lbl_len)
        )
    if ptype in ["nested", "stacked", "box"]:
        fig.xgrid.grid_line_color = None
    if ptype in ["nested", "stacked"]:
        fig.y_range.start = 0
        fig.x_range.range_padding = 0.03
    if ptype in ["line", "boxnum"]:
        fig.min_border_right = 20
        fig.xaxis.major_label_standoff = 7
        fig.xaxis.major_label_orientation = 0
        fig.xaxis.major_tick_line_color = None


def _make_title(grp_cnt_stats: Dict[str, int], x: str, y: str) -> str:
    """
    Format the title to notify the user of sampled output
    """
    x_ttl, y_ttl = None, None
    if f"{x}_ttl" in grp_cnt_stats:
        x_ttl = grp_cnt_stats[f"{x}_ttl"]
        x_shw = grp_cnt_stats[f"{x}_shw"]
    if f"{y}_ttl" in grp_cnt_stats:
        y_ttl = grp_cnt_stats[f"{y}_ttl"]
        y_shw = grp_cnt_stats[f"{y}_shw"]
    if x_ttl and y_ttl:
        if x_ttl > x_shw and y_ttl > y_shw:
            return f"(top {y_shw} out of {y_ttl}) {y} by (top {x_shw} out of {x_ttl}) {x}"
    elif x_ttl:
        if x_ttl > x_shw:
            return f"{y} by (top {x_shw} out of {x_ttl}) {x}"
    elif y_ttl:
        if y_ttl > y_shw:
            return f"(top {y_shw} out of {y_ttl}) {y} by {x}"
    return f"{y} by {x}"


def _format_ticks(ticks: List[float]) -> List[str]:
    """
    Format the tick values
    """
    formatted_ticks = []
    for tick in ticks:  # format the tick values
        before, after = f"{tick:e}".split("e")
        if float(after) > 1e15 or abs(tick) < 1e4:
            formatted_ticks.append(str(tick))
            continue
        mod_exp = int(after) % 3
        factor = 1 if mod_exp == 0 else 10 if mod_exp == 1 else 100
        value = np.round(float(before) * factor, len(str(before)))
        value = int(value) if value.is_integer() else value
        if abs(tick) >= 1e12:
            formatted_ticks.append(str(value) + "T")
        elif abs(tick) >= 1e9:
            formatted_ticks.append(str(value) + "B")
        elif abs(tick) >= 1e6:
            formatted_ticks.append(str(value) + "M")
        elif abs(tick) >= 1e4:
            formatted_ticks.append(str(value) + "K")

    return formatted_ticks


def _format_axis(fig: Figure, minv: int, maxv: int, axis: str) -> None:
    """
    Format the axis ticks
    """  # pylint: disable=too-many-locals
    # divisor for 5 ticks (5 results in ticks that are too close together)
    divisor = 4.5
    # interval
    gap = (maxv - minv) / divisor
    # get exponent from scientific notation
    _, after = f"{gap:.0e}".split("e")
    # round to this amount
    round_to = -1 * int(after)
    # round the first x tick
    minv = np.round(minv, round_to)
    # round value between ticks
    gap = np.round(gap, round_to)

    # make the tick values
    ticks = [float(minv)]
    while max(ticks) + gap < maxv:
        ticks.append(max(ticks) + gap)
    ticks = np.round(ticks, round_to)
    ticks = [int(tick) if tick.is_integer() else tick for tick in ticks]
    formatted_ticks = _format_ticks(ticks)

    if axis == "x":
        fig.xgrid.ticker = ticks
        fig.xaxis.ticker = ticks
        fig.xaxis.major_label_overrides = dict(zip(ticks, formatted_ticks))
        fig.xaxis.major_label_text_font_size = "10pt"
        fig.xaxis.major_label_standoff = 7
        # fig.xaxis.major_label_orientation = 0
        fig.xaxis.major_tick_line_color = None
    elif axis == "y":
        fig.ygrid.ticker = ticks
        fig.yaxis.ticker = ticks
        fig.yaxis.major_label_overrides = dict(zip(ticks, formatted_ticks))
        fig.yaxis.major_label_text_font_size = "10pt"
        fig.yaxis.major_label_standoff = 5


def _format_bin_intervals(bins_arr: np.ndarray) -> List[str]:
    """
    Auxillary function to format bin intervals in a histogram
    """
    bins_arr = np.round(bins_arr, 3)
    bins_arr = [int(val) if float(val).is_integer() else val for val in bins_arr]
    intervals = [f"[{bins_arr[i]}, {bins_arr[i + 1]})" for i in range(len(bins_arr) - 2)]
    intervals.append(f"[{bins_arr[-2]},{bins_arr[-1]}]")
    return intervals


def _format_values(key: str, value: List[Any]) -> List[str]:
    for i in range(len(value)):
        if not isinstance(value[i], (int, float)):
            # if value is a time
            value[i] = str(value[i])
            continue

        if "Memory" in key:
            # for memory usage
            ind = 0
            unit = dict(enumerate(["B", "KB", "MB", "GB", "TB"], 0))
            while value[i] > 1024:
                value[i] /= 1024
                ind += 1
            value[i] = f"{value[i]:.1f} {unit[ind]}"
            continue

        if (value[i] * 10) % 10 == 0:
            # if value is int but in a float form with 0 at last digit
            val = int(value[i])
            if abs(val) >= 1000000:
                val = f"{val:.5g}"
        elif abs(value[i]) >= 1000000 or abs(value[i]) < 0.001:
            val = f"{value[i]:.5g}"
        elif abs(value[i]) >= 1:
            # eliminate trailing zeros
            pre_value = float(f"{value[i]:.4f}")
            val = int(pre_value) if (pre_value * 10) % 10 == 0 else pre_value
        elif 0.001 <= abs(value[i]) < 1:
            val = f"{value[i]:.4g}"
        else:
            val = str(value[i])

        if "%" in key:
            # for percentage, only use digits before notation sign for extreme small number
            val = f"{float(val):.1%}"
        value[i] = str(val)
        continue
    return value


def _align_cols(col: str, target_cnt: int, df_list: Union[List[pd.DataFrame], pd.DataFrame], nrows: List[int]) -> Tuple[List[pd.DataFrame], List[int]]:
    """
    To make the comparison clearer, we use 0 to fill the non-existing columns and their
    corresponding data from computation
    """
    if isinstance(df_list, list):
        base_cnt = len(df_list)
    else:
        base_cnt = len(nrows)
        df_list = list(df_list)

    if base_cnt < target_cnt:
        diff = target_cnt - base_cnt
        zero_clone = df_list[0].copy()
        zero_clone[col] = 0
        for _ in range(diff):
            df_list.append(zero_clone)
            nrows.append(nrows[0])

    return df_list, nrows


def bar_viz(
    df: Union[List[pd.DataFrame], pd.DataFrame],
    ttl_grps: List[int],
    nrows: List[int],
    col: str,
    yscale: str,
    plot_width: int,
    plot_height: int,
    show_yticks: bool,
    target_cnt: int,
    df_labels: List[str],
) -> Figure:
    """
    Render a bar chart
    """

    df, nrows = _align_cols(col, target_cnt, df, nrows)
    df = pd.concat(df, axis=1, copy=False, ignore_index=True).fillna(0)
    df.columns = df_labels
    # pylint: disable=too-many-arguments
    for i in range(target_cnt):
        df[f"pct{i}"] = df[df_labels[i]] / nrows[i] * 100
    df.index = [str(val) for val in df.index]

    tooltips = [(col, "@col_name"), ("Count", "@count"), ("Percent", "@pct{0.2f}%")]

    if show_yticks:
        if len(df) > 10:
            plot_width = 28 * len(df)

    x_ticks = [(ind, df_label) for ind in df.index for df_label in df_labels]
    count = sum(zip(*[df[f"df{i+1}"] for i in range(target_cnt)]), ())
    col_name = sum(zip(*[df.index for _ in range(target_cnt)]), ())
    pct = sum(zip(*[df[f"pct{i}"] for i in range(target_cnt)]), ()) # index starts from 1
    data = ColumnDataSource(data=dict(x_ticks=x_ticks, count=count, col_name=col_name, pct=pct))

    fig = Figure(
        plot_width=plot_width,
        plot_height=plot_height,
        title=col,
        toolbar_location=None,
        tooltips=tooltips,
        tools="hover",
        x_range=FactorRange(*x_ticks),
        y_axis_type=yscale,
    )
    fig.vbar(x="x_ticks", width=0.8, top='count', bottom=0.01, source=data)
    tweak_figure(fig, "bar", show_yticks)
    fig.yaxis.axis_label = "Count"
    if ttl_grps[0] > len(df):
        fig.xaxis.axis_label = f"Top {len(df)} of {ttl_grps[0]} {col}"
        fig.xaxis.axis_label_standoff = 0

    if show_yticks and yscale == "linear":
        _format_axis(fig, 0, df.max().max(), "y")
    return fig

def hist_viz(
    hist: List[Tuple[np.ndarray, np.ndarray]],
    nrows: int,
    col: str,
    yscale: str,
    plot_width: int,
    plot_height: int,
    show_yticks: bool,
    df_labels: List[str],
) -> Figure:
    """
    Render a histogram
    """
    # pylint: disable=too-many-arguments,too-many-locals

    tooltips = [("Bin", "@intvl"), ("Frequency", "@freq"), ("Percent", "@pct{0.2f}%")]
    fig = Figure(
        plot_height=plot_height,
        plot_width=plot_width,
        title=col,
        toolbar_location=None,
        y_axis_type=yscale,
    )

    for i in range(len(hist)):
        counts, bins = hist[i]
        if sum(counts) == 0:
            fig.rect(x=0, y=0, width=0, height=0)
            continue
        intvls = _format_bin_intervals(bins)
        df = pd.DataFrame({
            "intvl": intvls,
            "left": bins[:-1],
            "right": bins[1:],
            "freq": counts,
            "pct": counts / nrows[i] * 100
        })
        bottom = 0 if yscale == "linear" or df.empty else counts.min() / 2
        fig.quad(
            source=df,
            left="left",
            right="right",
            bottom=bottom,
            alpha=0.5,
            top="freq",
            fill_color="#6baed6",
            legend_label=df_labels[i]
        )
        hover = HoverTool(tooltips=tooltips, mode="vline")
        fig.add_tools(hover)

    tweak_figure(fig, "hist", show_yticks)
    fig.yaxis.axis_label = "Frequency"
    fig.legend.location = "top_center"
    fig.legend.click_policy="hide"
    fig.legend.orientation='horizontal'
    _format_axis(fig, df.iloc[0]["left"], df.iloc[-1]["right"], "x")

    # todo
    if show_yticks:
        fig.xaxis.axis_label = col
        if yscale == "linear":
            _format_axis(fig, 0, df["freq"].max(), "y")

    return fig


def format_ov_stats(stats: Dict[str, List[Any]]) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Render statistics information for distribution grid
    """
    # pylint: disable=too-many-locals
    nrows, ncols, npresent_cells, nrows_wo_dups, mem_use, dtypes_cnt = stats.values()
    ncells = np.multiply(nrows, ncols).tolist()

    data = {
        "Number of Variables": ncols,
        "Number of Rows": nrows,
        "Missing Cells": np.subtract(ncells, npresent_cells).astype(float).tolist(),
        "Missing Cells (%)": np.subtract(1, np.divide(npresent_cells, ncells)).tolist(),
        "Duplicate Rows": np.subtract(nrows, nrows_wo_dups).tolist(),
        "Duplicate Rows (%)": np.subtract(1, np.divide(nrows_wo_dups, nrows)).tolist(),
        "Total Size in Memory": list(map(float, mem_use)),
        "Average Row Size in Memory": np.subtract(mem_use, nrows).tolist(),
    }
    return {k: _format_values(k, v) for k, v in data.items()}, dtypes_cnt


def render_comparison_grid(itmdt: Intermediate, cfg: Config) -> Dict[str, Any]:
    """
    Create visualizations for plot(df)
    """
    # pylint: disable=too-many-locals
    plot_width = cfg.plot.width if cfg.plot.width is not None else 324
    plot_height = cfg.plot.height if cfg.plot.height is not None else 300
    df_labels = ['df1', 'df2', 'df3'] # todo: add to config
    figs: List[Figure] = []
    nrows = itmdt["stats"]["nrows"]
    target_cnt = itmdt["target_cnt"]
    titles: List[str] = []
    for col, dtp, data in itmdt["data"]:
        if is_dtype(dtp, Nominal()):
            df, ttl_grps = data
            fig = bar_viz(
                df,
                ttl_grps,
                nrows,
                col,
                cfg.bar.yscale,
                plot_width,
                plot_height,
                False,
                target_cnt,
                df_labels
            )
        elif is_dtype(dtp, Continuous()):
            fig = hist_viz(data, nrows, col, cfg.hist.yscale, plot_width, plot_height, False, df_labels)
        elif is_dtype(dtp, DateTime()):
            # df, timeunit, miss_pct = data
            # fig = dt_line_viz(
            #     df, col, timeunit, cfg.line.yscale, plot_width, plot_height, False, miss_pct
            # )
            continue
        fig.frame_height = plot_height
        titles.append(fig.title.text)
        fig.title.text = ""
        figs.append(fig)

    if cfg.stats.enable:
        toggle_content = "Stats"
    else:
        toggle_content = None

    return {
        "layout": figs,
        "meta": titles,
        "tabledata": format_ov_stats(itmdt["stats"]) if cfg.stats.enable else None,
        "container_width": plot_width * 3,
        "toggle_content": toggle_content,
    }


def render(itmdt: Intermediate, cfg: Config) -> Dict[str, Any]:
    """
    Render a basic plot

    Parameters
    ----------
    itmdt
        The Intermediate containing results from the compute function.
    cfg
        Config instance
    """

    if itmdt.visual_type == "comparison_grid":
        visual_elem = render_comparison_grid(itmdt, cfg)

    return visual_elem
