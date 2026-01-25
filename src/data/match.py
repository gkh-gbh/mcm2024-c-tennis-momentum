# src/data/match.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# 数据读取
# -----------------------------
def load_raw_points_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    读取逐分数据（point-by-point）CSV。
    这里不做复杂清洗，只确保读取成功并返回 DataFrame。

    Parameters
    ----------
    csv_path : str | Path
        原始数据路径，例如: data/raw/Wimbledon_featured_matches.csv

    Returns
    -------
    pd.DataFrame
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Loaded dataframe is empty. Check the input file.")
    return df


# -----------------------------
# 单场比赛抽取 + 排序
# -----------------------------
def get_match(df: pd.DataFrame, match_id: str) -> pd.DataFrame:
    """
    从全量 df 中筛选某场 match，并按 set/game/point 顺序排序。

    注意：逐分数据必须排序后，时间序列意义才成立。

    Parameters
    ----------
    df : pd.DataFrame
        全量逐分数据
    match_id : str
        比赛 ID

    Returns
    -------
    df_match : pd.DataFrame
        已排序的单场比赛数据副本
    """
    if "match_id" not in df.columns:
        raise KeyError("Column 'match_id' not found in dataframe.")

    df_match = df[df["match_id"] == match_id].copy()
    if df_match.empty:
        raise ValueError(f"No rows found for match_id={match_id}")

    # 必须存在这些列才能排序
    required = ["set_no", "game_no", "point_no"]
    _assert_cols(df_match, required)

    df_match = df_match.sort_values(by=required).reset_index(drop=True)
    return df_match


# -----------------------------
# 构造 point_result / serve-adjusted
# -----------------------------
def add_point_result(df_match: pd.DataFrame, winner_col: str = "point_victor") -> pd.DataFrame:
    """
    构造 point_result: Player1 赢 = +1, Player2 赢 = -1

    Parameters
    ----------
    df_match : pd.DataFrame
    winner_col : str
        默认使用 point_victor（数值通常为 1/2）

    Returns
    -------
    pd.DataFrame
        原 df_match 增加一列 'point_result'
    """
    _assert_cols(df_match, [winner_col])

    # 约定：赢家为1 -> +1，否则 -> -1
    df_match = df_match.copy()
    df_match["point_result"] = np.where(df_match[winner_col] == 1, 1, -1).astype(int)
    return df_match


def add_serve_adjusted_contrib(
    df_match: pd.DataFrame,
    server_col: str = "server",
    winner_col: str = "point_victor",
    p_server_win: Optional[float] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    构造发球校正后的单分贡献 serve_adj_p1，并返回估计的发球方胜率 p_server_win。

    设计思想（人话）：
    - 发球方赢分本来更容易发生，所以贡献小（1 - p）
    - 接发方赢分更难发生，所以贡献大（-p）
    然后再统一成 Player1 视角（正表示对 Player1 有利）。

    Parameters
    ----------
    df_match : pd.DataFrame
    server_col : str
        发球方标识列（通常为 1/2）
    winner_col : str
        得分方列（通常为 1/2）
    p_server_win : Optional[float]
        若传入则使用固定值；若为 None 则从当前比赛经验估计

    Returns
    -------
    (df_out, p_server_win)
        df_out 新增列:
            - serve_wins (0/1)
            - serve_adj (发球方视角贡献)
            - serve_adj_p1 (Player1 视角贡献)
    """
    _assert_cols(df_match, [server_col, winner_col])

    df_out = df_match.copy()

    # 发球方是否赢分（1=是，0=否）
    df_out["serve_wins"] = (df_out[winner_col] == df_out[server_col]).astype(int)

    # 估计发球方胜率 p
    if p_server_win is None:
        p_server_win = float(df_out["serve_wins"].mean())

    # 发球方视角的“超额表现”贡献
    # 发球方赢：+ (1 - p)；接发赢：- p
    df_out["serve_adj"] = np.where(df_out["serve_wins"] == 1, 1 - p_server_win, -p_server_win)

    # 转换到 Player1 视角：server=1 时同向；server=2 时取反
    df_out["serve_adj_p1"] = np.where(df_out[server_col] == 1, df_out["serve_adj"], -df_out["serve_adj"])

    return df_out, p_server_win


# -----------------------------
# 内部工具
# -----------------------------
def _assert_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
