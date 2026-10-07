# -*- coding: utf-8 -*-
"""通用工具：配置加载、井文件发现与读取、缺失哨兵值清洗、赛题评分指标。"""
from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd

EPS = 1e-5  # 赛题评分中的极小值 ε


def set_seed(seed: int) -> None:
    """固定 Python / NumPy 随机种子（复现要求：代码内部固定随机种子）。"""
    random.seed(seed)
    np.random.seed(seed)


def load_config(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_well_files(data_dir: str):
    """递归发现目录下全部 .txt 井文件。

    返回 [(logId, path), ...]，logId 为文件名（不含扩展名），
    与 result.json 中的 logId 一一对应；结果按 logId 排序保证确定性。
    """
    items = []
    for root, _dirs, files in os.walk(data_dir):
        for fn in files:
            if fn.lower().endswith(".txt"):
                path = os.path.join(root, fn)
                log_id = os.path.splitext(fn)[0]
                items.append((log_id, path))
    items.sort(key=lambda x: x[0])
    return items


def find_test_subdir(data_dir: str) -> str:
    """若 data_dir 的一级子目录中存在名称含 test/eval/测试 的目录，则进入该目录。

    兼容评测时 --data_dir 指向原始数据根目录（内含 train/ 与 test/）的情况。
    """
    if not os.path.isdir(data_dir):
        return data_dir
    for name in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, name)
        low = name.lower()
        if os.path.isdir(p) and any(k in low for k in ("test", "eval", "测试", "预测")):
            return p
    return data_dir


def read_well_file(path: str, encoding: str = "utf-8") -> pd.DataFrame:
    """读取单井 txt：第一行为曲线名，第二行为单位行（自动识别剔除），后续为采样数据。

    - 兼容逗号 / 空格 / 制表符分隔与 UTF-8 BOM
    - 依次尝试 utf-8-sig / gbk 编码
    - 无法解析为数值的单元格置为 NaN；全空行剔除
    """
    last_err = None
    raw = None
    for enc in (encoding, "utf-8-sig", "gbk"):
        try:
            raw = pd.read_csv(path, sep=r"[,\t; ]+", engine="python",
                              header=None, encoding=enc, skip_blank_lines=True,
                              dtype=str)
            break
        except UnicodeDecodeError as e:  # 仅编码问题才换编码重试
            last_err = e
    if raw is None:
        raise IOError(f"无法读取文件 {path}: {last_err}")

    # 第一行：曲线名
    header = [str(c).strip() for c in raw.iloc[0].tolist()]
    body = raw.iloc[1:].reset_index(drop=True)
    # 第二行：若多数单元格不是数值，视为单位行并剔除
    if len(body) > 0:
        probe = pd.to_numeric(body.iloc[0].replace("", np.nan), errors="coerce")
        if probe.isna().mean() > 0.5:
            body = body.iloc[1:].reset_index(drop=True)

    df = body.apply(pd.to_numeric, errors="coerce")
    df.columns = [h if h else f"col_{i}" for i, h in enumerate(header)]
    keep = [c for c in df.columns if not c.startswith("Unnamed") and not c.startswith("col_")]
    df = df.loc[:, keep]
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def clean_sentinels(df: pd.DataFrame, sentinels) -> pd.DataFrame:
    """把常见缺失哨兵值（如 -999.25）替换为 NaN。"""
    if sentinels:
        df = df.replace(list(sentinels), np.nan)
    return df


# ---------------- 赛题评分指标 ----------------
# PERM 指标实现为 Acc = mean(max(0, 1 - |log10(max(ŷ,ε)/max(y,ε))|))：
# 对真值/预测值均取 ε=1e-5 兜底避免 log(0)；真值 y>0 时与官方公式一致。

def acc_por(y_true, y_pred, eps: float = EPS) -> float:
    """孔隙度（POR）准确率：相对误差阈值 8%。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = np.abs(y_pred - y_true) / (0.08 * (np.abs(y_true) + eps))
    return float(np.mean(np.maximum(0.0, 1.0 - err)))


def acc_perm(y_true, y_pred, eps: float = EPS) -> float:
    """渗透率（PERM）准确率：log10 比值误差阈值 1 个数量级。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ratio_err = np.abs(np.log10(np.maximum(y_pred, eps) / np.maximum(y_true, eps)))
    return float(np.mean(np.maximum(0.0, 1.0 - ratio_err)))


def acc_sw(y_true, y_pred, eps: float = EPS) -> float:
    """含水饱和度（SW）准确率：相对误差阈值 5%。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = np.abs(y_pred - y_true) / (0.05 * (np.abs(y_true) + eps))
    return float(np.mean(np.maximum(0.0, 1.0 - err)))


def total_score(acc_p: float, acc_pe: float, acc_s: float) -> float:
    """综合百分制总分 = (0.3*AccPOR + 0.35*AccPERM + 0.35*AccSW) * 100"""
    return (0.3 * acc_p + 0.35 * acc_pe + 0.35 * acc_s) * 100.0
