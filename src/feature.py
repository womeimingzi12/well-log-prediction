# -*- coding: utf-8 -*-
"""特征工程：原始曲线 + 电阻率对数 + 物理启发特征 + 井内滚动统计 + 井内稳健归一化。

所有特征均"逐井无状态"构建：同一函数同时用于训练井与测试井，
只依赖当前井自身的信息，不引入跨井统计量，天然避免数据泄漏。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

INPUT_CURVES = ["DEPTH", "DEVI", "AZIM", "GR", "SP", "AC", "DEN", "CNL",
                "PE", "RXO", "RT", "CAL", "BIT", "CASE"]
TARGETS = ["POR", "PERM", "SW"]

# 参与滚动统计 / 井内归一化的主要曲线（不含井身常量 BIT/CASE 与角度类）
KEY_CURVES = ["GR", "SP", "AC", "DEN", "CNL", "PE", "RXO", "RT", "CAL"]

# 物理常量（砂泥岩剖面经验值，仅用于特征构造，不作为硬约束）
DEN_MA, DEN_FL = 2.65, 1.0     # 密度骨架/流体 (g/cm3)
AC_MA, AC_FL = 182.0, 620.0    # 声波时差骨架/流体 (μs/m，Wyllie 时间平均式)


def _ensure_columns(df: pd.DataFrame, cols) -> pd.DataFrame:
    """缺失曲线自动补 NaN 列（部分井可能没有全部曲线）。"""
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def build_features(df: pd.DataFrame, rolling_windows=(5, 20),
                   max_depth_gap: float = 1.0) -> pd.DataFrame:
    """输入单井 DataFrame（含输入曲线列），输出逐深度点特征矩阵（含 DEPTH 列）。

    参数：
        rolling_windows: 滚动统计窗口半径列表（单位：采样点数）
        max_depth_gap:   深度断层阈值（m），相邻采样点深度差超过该值视为不同井段，
                         滚动统计与差分不跨段计算
    """
    df = df.copy()
    _ensure_columns(df, INPUT_CURVES)

    # 按深度排序，保证滚动 / 差分特征方向正确
    df = df.sort_values("DEPTH").reset_index(drop=True)

    feat = pd.DataFrame(index=df.index)
    feat["DEPTH"] = df["DEPTH"]

    # ---- 1. 原始曲线 ----
    for c in INPUT_CURVES:
        if c != "DEPTH":
            feat[c] = df[c]

    # ---- 2. 电阻率对数变换（跨数量级变量）----
    for c in ("RXO", "RT"):
        v = df[c].to_numpy(dtype=float)
        feat[f"LOG10_{c}"] = np.log10(np.maximum(v, 1e-5))

    # ---- 3. 物理启发特征 ----
    den = df["DEN"].to_numpy(dtype=float)
    ac = df["AC"].to_numpy(dtype=float)
    cnl = df["CNL"].to_numpy(dtype=float)
    feat["PHI_DEN"] = (DEN_MA - den) / (DEN_MA - DEN_FL)        # 密度孔隙度估计
    feat["PHI_AC"] = (ac - AC_MA) / (AC_FL - AC_MA)             # 声波孔隙度估计(Wyllie)
    feat["PHI_CNL"] = cnl / 100.0                               # 中子孔隙度(%→小数)
    phi_d = np.maximum(feat["PHI_DEN"].to_numpy(dtype=float), 0.0)
    phi_a = np.maximum(feat["PHI_AC"].to_numpy(dtype=float), 0.0)
    feat["PHI_DEN_AC"] = np.sqrt(phi_d * phi_a)                 # 声密组合孔隙度
    feat["LOG_RT_RXO"] = feat["LOG10_RT"] - feat["LOG10_RXO"]   # 泥浆侵入程度
    feat["DEN_CNL_CROSS"] = den * cnl

    # ---- 4. 井斜 / 方位角（圆变量编码）----
    azim = np.deg2rad(df["AZIM"].to_numpy(dtype=float))
    feat["AZIM_SIN"] = np.sin(azim)
    feat["AZIM_COS"] = np.cos(azim)

    # ---- 5. 井内相对深度 ----
    depth = df["DEPTH"].to_numpy(dtype=float)
    d_min, d_max = np.nanmin(depth), np.nanmax(depth)
    rng = d_max - d_min if d_max > d_min else 1.0
    feat["DEPTH_REL"] = (depth - d_min) / rng

    # ---- 6. 井内滚动统计（多尺度、按深度段隔离）+ 一阶差分 ----
    gap = df["DEPTH"].diff().abs()
    seg_id = (gap > max_depth_gap).cumsum()   # 深度断层处开启新段
    for wdw in rolling_windows:
        win = 2 * wdw + 1
        for c in KEY_CURVES:
            gr = df.groupby(seg_id)[c]
            feat[f"ROLLm{wdw}_{c}"] = gr.rolling(
                win, center=True, min_periods=1).mean().reset_index(level=0, drop=True)
            feat[f"ROLLs{wdw}_{c}"] = gr.rolling(
                win, center=True, min_periods=1).std().reset_index(level=0, drop=True)
    for c in KEY_CURVES:
        d_prev = df[c].diff()
        d_prev[gap > max_depth_gap] = np.nan  # 断层处不做差分
        d_next = df[c].diff(-1)
        boundary_next = gap.shift(-1).fillna(0.0) > max_depth_gap
        d_next[boundary_next] = np.nan
        feat[f"DIFF_{c}"] = d_prev
        feat[f"DIFFN_{c}"] = d_next

    # ---- 7. 井内稳健归一化（中位数/四分位距）+ 井内分位数排名 ----
    for c in KEY_CURVES:
        s = df[c]
        med = s.median()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1 if q3 - q1 > 0 else 1.0
        feat[f"WZ_{c}"] = (s - med) / iqr
        feat[f"WRANK_{c}"] = s.rank(pct=True)

    return feat
