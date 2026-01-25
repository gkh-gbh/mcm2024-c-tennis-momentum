# src/viz/plots.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_momentum_with_swings(
    m: pd.Series,
    swing_flag: Optional[pd.Series] = None,
    title: str = "Detected Momentum Swings",
    out_path: Optional[str | Path] = None,
    figsize=(12, 5),
) -> None:
    """
    画势头曲线，并在 swing 点处画散点；支持保存到文件。

    Parameters
    ----------
    m : pd.Series
        势头曲线（建议 momentum_srv_ewm）
    swing_flag : Optional[pd.Series]
        与 m 同索引的 0/1 标记；若为 None 则只画曲线
    title : str
        图标题
    out_path : Optional[str | Path]
        若提供则保存到该路径，例如 reports/figures/match1701_swings.png
    figsize : tuple
        图尺寸
    """
    plt.figure(figsize=figsize)
    plt.plot(m.values, label="Momentum (EWMA)")

    # 画 0 线（势头优势分界）
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)

    if swing_flag is not None:
        idx = swing_flag[swing_flag == 1].index
        # 转成位置索引便于画 x
        xs = [m.index.get_loc(i) for i in idx]
        ys = m.loc[idx].values
        plt.scatter(xs, ys, s=45, label="Detected swings")

    plt.title(title)
    plt.xlabel("Point Index")
    plt.ylabel("Momentum (Player1 advantage)")
    plt.legend()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", dpi=200)

    plt.show()
