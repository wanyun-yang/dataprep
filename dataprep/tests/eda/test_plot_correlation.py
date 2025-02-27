"""
    module for testing plot_corr(df, x, y) function.
"""
from dataprep.eda.data_array import DataArray
import random
from time import time

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from ...eda.correlation import compute_correlation, plot_correlation
from ...eda.correlation.compute.univariate import (
    _kendall_tau_1xn,
    _pearson_1xn,
    _spearman_1xn,
    _corr_filter,
)
from ...eda.correlation.compute.overview import (
    _spearman_nxn,
    _pearson_nxn,
    _kendall_tau_nxn,
)
from ...eda.utils import to_dask
from ...eda.data_array import DataArray
from ...eda.configs import Config


@pytest.fixture(scope="module")  # type: ignore
def simpledf() -> dd.DataFrame:
    df = pd.DataFrame(np.random.rand(100, 3), columns=["a", "b", "c"])
    df = pd.concat([df, pd.Series(["a"] * 100)], axis=1)
    df.columns = ["a", "b", "c", "d"]
    df = to_dask(df)

    return df


def test_sanity_compute_1(simpledf: dd.DataFrame) -> None:
    display = ["Stats", "Pearson"]
    config = {
        "height": 500,
        "width": 500,
    }
    cfg = Config.from_dict(display=display, config=config)
    compute_correlation(simpledf, cfg=cfg)
    plot_correlation(simpledf)


def test_sanity_compute_2(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, cfg=cfg, k=1)
    plot_correlation(simpledf, k=1)


def test_sanity_compute_3(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, x="a", cfg=cfg)
    plot_correlation(simpledf, x="a")


def test_sanity_compute_4(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, x="a", cfg=cfg, value_range=(0.5, 0.8))
    plot_correlation(simpledf, x="a", value_range=(0.5, 0.8))


def test_sanity_compute_5(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, x="a", cfg=cfg, k=1)
    plot_correlation(simpledf, x="a", k=1)


def test_sanity_compute_6(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, x="a", cfg=cfg, k=0)
    plot_correlation(simpledf, x="a", k=0)


def test_sanity_compute_7(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, x="b", cfg=cfg, y="a")
    plot_correlation(simpledf, x="b", y="a")


def test_sanity_compute_8(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, x="b", cfg=cfg, y="a", k=1)
    plot_correlation(simpledf, x="b", y="a", k=1)


def test_sanity_compute_9(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, cfg=cfg, value_range=(0.3, 0.7))
    plot_correlation(simpledf, value_range=(0.3, 0.7))


@pytest.mark.xfail  # type: ignore
def test_sanity_compute_fail_2(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, cfg=cfg, k=3, value_range=(0.3, 0.7))
    plot_correlation(simpledf, k=3, value_range=(0.3, 0.7))


@pytest.mark.xfail  # type: ignore
def test_sanity_compute_fail_3(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, x="a", cfg=cfg, value_range=(0.5, 0.8), k=3)
    plot_correlation(simpledf, x="a", value_range=(0.5, 0.8), k=3)


@pytest.mark.xfail  # type: ignore
def test_sanity_compute_fail_4(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, y="a", cfg=cfg)
    plot_correlation(simpledf, y="a")


@pytest.mark.xfail  # type: ignore
def test_sanity_compute_fail_5(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, x="d", cfg=cfg)
    plot_correlation(simpledf, x="d")


@pytest.mark.xfail  # type: ignore
def test_test_sanity_compute_fail_6(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, x="b", y="a", cfg=cfg, value_range=(0.5, 0.8))
    plot_correlation(simpledf, x="b", y="a", value_range=(0.5, 0.8))


@pytest.mark.xfail  # type: ignore
def test_sanity_compute_fail_7(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, x="b", y="a", cfg=cfg, value_range=(0.5, 0.8), k=3)
    plot_correlation(simpledf, x="b", y="a", value_range=(0.5, 0.8), k=3)


def test_compute_pearson(simpledf: dd.DataFrame) -> None:
    df = DataArray(simpledf)
    df = df.select_num_columns()
    df.compute()
    darray = df.values
    array = darray.compute()

    corr_eda = _pearson_nxn(df).compute()
    corr_pd = pd.DataFrame(data=array).corr("pearson").values
    assert np.isclose(corr_eda, corr_pd).all()

    for i in range(array.shape[1]):
        corr_eda = _pearson_1xn(darray[:, i : i + 1], darray).compute()
        assert np.isclose(_corr_filter(corr_eda)[1], np.sort(corr_pd[:, i])).all()


def test_compute_spearman(simpledf: dd.DataFrame) -> None:
    df = DataArray(simpledf)
    df = df.select_num_columns()
    df.compute()
    darray = df.values
    array = darray.compute()

    corr_eda = _spearman_nxn(df).compute()
    corr_pd = pd.DataFrame(data=array).corr("spearman").values
    assert np.isclose(corr_eda, corr_pd).all()

    for i in range(array.shape[1]):
        corr_eda = _spearman_1xn(darray[:, i : i + 1], darray).compute()
        assert np.isclose(_corr_filter(corr_eda)[1], np.sort(corr_pd[:, i])).all()


def test_compute_kendall(simpledf: dd.DataFrame) -> None:
    df = DataArray(simpledf)
    df = df.select_num_columns()
    df.compute()
    darray = df.values
    array = darray.compute()

    corr_eda = _kendall_tau_nxn(df).compute()
    corr_pd = pd.DataFrame(data=array).corr("kendall").values
    assert np.isclose(corr_eda, corr_pd).all()

    for i in range(array.shape[1]):
        corr_eda = _kendall_tau_1xn(darray[:, i : i + 1], darray).compute()
        assert np.isclose(_corr_filter(corr_eda)[1], np.sort(corr_pd[:, i])).all()


# def test_plot_corr_df() -> None:  # pylint: disable=too-many-locals
#     """
#     :return:
#     """
#     data = np.random.rand(100, 20)
#     df_data = pd.DataFrame(data)

#     start_p_pd = time()
#     res = df_data.corr(method="pearson")
#     end_p_pd = time()
#     print("pd pearson time: ", str(end_p_pd - start_p_pd) + " s")

#     start_p = time()
#     _, intermediate = plot_correlation(df=df_data, return_intermediate=True)
#     end_p = time()
#     print("our pearson time: ", str(end_p - start_p) + " s")
#     assert np.isclose(res, intermediate.result["corr_p"]).all()

#     start_s_pd = time()
#     res = df_data.corr(method="spearman")
#     end_s_pd = time()
#     print("pd spearman time: ", str(end_s_pd - start_s_pd) + " s")

#     start_s = time()
#     _, intermediate = plot_correlation(df=df_data, return_intermediate=True)
#     end_s = time()
#     print("our spearman time: ", str(end_s - start_s) + " s")
#     assert np.isclose(res, intermediate.result["corr_s"]).all()

#     start_k_pd = time()
#     res = df_data.corr(method="kendall")
#     end_k_pd = time()
#     print("pd kendall time: ", str(end_k_pd - start_k_pd) + " s")

#     start_k = time()
#     _, intermediate = plot_correlation(df=df_data, return_intermediate=True)
#     end_k = time()
#     print("our kendall time: ", str(end_k - start_k) + " s")
#     assert np.isclose(res, intermediate.result["corr_k"]).all()


# def test_plot_corr_df_k() -> None:
#     """
#     :return:
#     """
#     data = np.random.rand(100, 20)
#     df_data = pd.DataFrame(data)
#     k = 5
#     res = df_data.corr(method="pearson")
#     row, _ = np.shape(res)
#     res_re = np.reshape(np.triu(res, 1), (row * row,))
#     idx = np.argsort(np.absolute(res_re))
#     mask = np.zeros(shape=(row * row,))
#     for i in range(k):
#         mask[idx[-i - 1]] = 1
#     res = np.multiply(res_re, mask)
#     res = np.reshape(res, (row, row))
#     res = res.T
#     _, intermediate = plot_correlation(df=df_data, return_intermediate=True, k=k)
#     assert np.isclose(intermediate.result["corr_p"], res).all()
#     assert np.isclose(intermediate.result["mask_p"], mask).all()


# def test_plot_corr_df_x_k() -> None:
#     """
#     :return:
#     """
#     df_data = pd.DataFrame({"a": np.random.normal(0, 10, 100)})
#     df_data["b"] = df_data["a"] + np.random.normal(0, 10, 100)
#     df_data["c"] = df_data["a"] + np.random.normal(0, 10, 100)
#     df_data["d"] = df_data["a"] + np.random.normal(0, 10, 100)
#     x_name = "b"
#     k = 3
#     name_list = list(df_data.columns.values)
#     idx_name = name_list.index(x_name)
#     res_p = df_data.corr(method="pearson").values
#     res_p[idx_name][idx_name] = -1
#     res_s = df_data.corr(method="spearman").values
#     res_s[idx_name][idx_name] = -1
#     res_k = df_data.corr(method="kendall").values
#     res_k[idx_name][idx_name] = -1
#     _, _ = plot_correlation(df=df_data, x=x_name, return_intermediate=True, k=k)


# def test_plot_corr_df_x_y_k() -> None:
#     """
#     :return:
#     """
#     df_data = pd.DataFrame({"a": np.random.normal(0, 10, 100)})
#     df_data["b"] = df_data["a"] + np.random.normal(0, 10, 100)
#     df_data["c"] = df_data["a"] + np.random.normal(0, 10, 100)
#     df_data["d"] = df_data["a"] + np.random.normal(0, 10, 100)
#     x_name = "b"
#     y_name = "c"
#     k = 3
#     _ = plot_correlation(
#         df=df_data, x=x_name, y=y_name, return_intermediate=False, k=k,
#     )

#     letters = ["a", "b", "c"]
#     df_data_cat = pd.DataFrame({"a": np.random.normal(0, 10, 100)})
#     df_data_cat["b"] = pd.Categorical([random.choice(letters) for _ in range(100)])
#     df_data_cat["c"] = pd.Categorical([random.choice(letters) for _ in range(100)])
#     _, intermediate = plot_correlation(
#         df=df_data_cat, x="b", y="c", return_intermediate=True
#     )
#     assert np.isclose(
#         pd.crosstab(df_data_cat["b"], df_data_cat["c"]).values,
#         intermediate.result["cross_table"],
#     ).all()
