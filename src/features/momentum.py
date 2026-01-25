# src/features/momentum.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Literal, Optional

import numpy as np
import pandas as pd


def rolling_momentum(point_result: pd.Series, window: int = 15) -> pd.Series:
    """
    Rolling mean 势头：最近 window 分的平均得分优势（Player1 视角）。

    Parameters
    ----------
    point_result : pd.Series
        取值通常为 ±1
    window : int
        滚动窗口大小

    Returns
    -------
    pd.Series
        momentum_rm
    """
    return point_result.rolling(window=window, min_periods=1).mean()


def ewm_momentum(series: pd.Series, span: int = 25) -> pd.Series:
    """
    EWMA 势头：指数加权移动平均。

    Parameters
    ----------
    series : pd.Series
        输入序列（可以是 point_result 或 serve_adj_p1）
    span : int
        EWMA span，越大越平滑

    Returns
    -------
    pd.Series
        momentum_ewm
    """
    return series.ewm(span=span, adjust=False).mean()


def compute_momentum(
    df_match: pd.DataFrame,
    method: Literal["rm", "ewm", "srv_ewm"] = "srv_ewm",
    window: int = 15,
    span: int = 25,
    point_col: str = "point_result",
    serve_adj_col: str = "serve_adj_p1",
) -> pd.Series:
    """
    统一入口：根据 method 计算势头曲线。

    method 说明：
    - "rm"      : rolling mean on point_result
    - "ewm"     : EWMA on point_result
    - "srv_ewm" : EWMA on serve_adj_p1（发球优势校正后的贡献）

    Returns
    -------
    pd.Series
        momentum 序列（与 df_match 同索引）
    """
    if method == "rm":
        if point_col not in df_match.columns:
            raise KeyError(f"Missing '{point_col}' for method=rm")
        return rolling_momentum(df_match[point_col], window=window)

    if method == "ewm":
        if point_col not in df_match.columns:
            raise KeyError(f"Missing '{point_col}' for method=ewm")
        return ewm_momentum(df_match[point_col], span=span)

    if method == "srv_ewm":
        if serve_adj_col not in df_match.columns:
            raise KeyError(f"Missing '{serve_adj_col}' for method=srv_ewm")
        return ewm_momentum(df_match[serve_adj_col], span=span)

    raise ValueError(f"Unknown method: {method}")
