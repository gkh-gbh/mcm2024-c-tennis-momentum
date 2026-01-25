# src/features/swings.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SwingParams:
    """
    swing 检测参数（建议写进报告/论文方便复现）
    """
    trend_span: int = 5        # 对斜率做 EWMA 平滑的 span（越大越稳）
    amp_q: float = 0.85        # 幅度阈值分位数（越大越严格）
    cool: int = 15             # 冷却时间：相邻 swing 最少间隔多少分
    fill_zero_trend: bool = True  # 是否将 trend=0 用前向填充保持方向


def detect_swings(
    m: pd.Series,
    params: SwingParams = SwingParams(),
) -> Tuple[pd.Series, list[int], float]:
    """
    根据势头曲线 m 检测 swing 点。

    返回：
    - swing_flag: 与 m 同索引的 0/1 序列
    - swing_idx : swing 点的索引列表（整数位置）
    - amp_thr   : 实际使用的幅度阈值（便于记录）

    方法概述：
    1) trend = diff(m) 的 EWMA 平滑 -> 稳定趋势方向
    2) 反转 rev：trend 由正转负 or 由负转正
    3) 幅度过滤：|m| > quantile(amp_q)
    4) 区间压缩：连续满足条件的点视为一次事件，只取 |m| 最大的代表点
    5) 冷却：相邻代表点过近则只保留 |m| 更大的那个
    """
    if m.isna().all():
        raise ValueError("Momentum series is all NaN.")

    # 1) 趋势信号：对斜率做 EWMA 平滑，减少抖动误判
    trend = m.diff().ewm(span=params.trend_span, adjust=False).mean()

    if params.fill_zero_trend:
        trend = trend.replace(0, np.nan).ffill()

    # 2) 反转：上升->下降 或 下降->上升
    rev = (
        ((trend.shift(1) > 0) & (trend <= 0)) |
        ((trend.shift(1) < 0) & (trend >= 0))
    )

    # 3) 幅度阈值（自适应）
    amp_thr = float(m.abs().quantile(params.amp_q))
    candidate = rev & (m.abs() > amp_thr)

    # 4) 区间压缩：连续 candidate 只取 |m| 最大点
    idx = candidate[candidate].index.to_list()

    # 如果 index 不是 RangeIndex（比如保留了原 index），我们用位置索引更稳
    # 这里用 m.index.get_loc 来找位置
    pos_idx = [m.index.get_loc(i) for i in idx]

    keep_pos = _compress_consecutive_by_max_abs(m, pos_idx)

    # 5) 冷却：相邻过近则保留更大 |m| 的
    keep_pos = _apply_cooldown_keep_max_abs(m, keep_pos, cool=params.cool)

    swing_flag = pd.Series(0, index=m.index, dtype=int)
    for p in keep_pos:
        swing_flag.iloc[p] = 1

    return swing_flag, keep_pos, amp_thr


def _compress_consecutive_by_max_abs(m: pd.Series, pos_idx: list[int]) -> list[int]:
    """
    将连续的位置索引合并成一个组，每组只保留 |m| 最大的位置。
    """
    if not pos_idx:
        return []

    pos_idx = sorted(pos_idx)
    keep = []
    i = 0
    while i < len(pos_idx):
        j = i
        group = [pos_idx[i]]
        while j + 1 < len(pos_idx) and pos_idx[j + 1] == pos_idx[j] + 1:
            j += 1
            group.append(pos_idx[j])

        best = max(group, key=lambda p: abs(m.iloc[p]))
        keep.append(best)
        i = j + 1

    return keep


def _apply_cooldown_keep_max_abs(m: pd.Series, keep_pos: list[int], cool: int) -> list[int]:
    """
    冷却：相邻 swing 距离 < cool 时，保留 |m| 更大的点。
    """
    if not keep_pos:
        return []

    keep_pos = sorted(keep_pos)
    final = [keep_pos[0]]

    for p in keep_pos[1:]:
        last = final[-1]
        if p - last >= cool:
            final.append(p)
        else:
            # 太近则保留更大幅度的
            if abs(m.iloc[p]) > abs(m.iloc[last]):
                final[-1] = p

    return final
