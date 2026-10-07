# -*- coding: utf-8 -*-
"""模型封装：POR / PERM / SW 三目标"两段式"模型。

每个目标 = 地板二分类器 + 储层点回归器：
- 分类器：判断深度点是否属于"地板值"（非储层声明值，见 src.feature.FLOOR）；
- 回归器：仅在非地板（储层）样本上训练；
- 推理：分类概率 ≥ 阈值（由训练集折外数据扫描确定）→ 直接输出地板值，否则输出回归值。
- PERM 全程在 log10 空间（与赛题评分的 log10 比值指标对齐）。

LightGBM 开启 deterministic，配合固定种子保证结果可复现。
产物：models/model.pkl（每目标的 clf/reg Booster + 特征列 + 阈值）
      models/scaler.pkl（缺失插补 + 标准化管线）
"""
from __future__ import annotations

import math
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.feature import FLOOR

TARGETS = ["POR", "PERM", "SW"]


def _base_params(seed: int, num_threads: int) -> dict:
    """三个目标共用的正则化与确定性配置（可被 config.yaml 的 lgb_params 覆盖）。"""
    return {
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 30,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "deterministic": True,     # 同数据+同参数+同线程数下结果可复现
        "force_col_wise": True,
        "seed": seed,
        "num_threads": num_threads,
        "verbose": -1,
    }


def make_lgb_params(extra: dict | None, seed: int, num_threads: int) -> dict:
    """储层回归器参数（早停指标由 feval 提供，故 metric=None）。"""
    params = _base_params(seed, num_threads)
    params.update({"objective": "regression", "metric": "None"})
    if extra:
        params.update(extra)
    return params


def make_lgb_clf_params(extra: dict | None, seed: int, num_threads: int) -> dict:
    """地板分类器参数（早停按 AUC，其余与回归器一致）。"""
    params = _base_params(seed, num_threads)
    params.update({"objective": "binary", "metric": "auc"})
    if extra:
        params.update(extra)
    return params


def make_scaler():
    """缺失插补 + 标准化管线（训练后保存为 scaler.pkl）。"""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])


def train_booster(X, y, params, X_val=None, y_val=None,
                  num_boost_round: int = 3000, early_stopping_rounds: int = 200,
                  feature_names=None, feval=None):
    """训练单个 Booster，返回 (booster, best_iteration)。

    feval：可选的赛题指标评估函数（用于折内早停，与 metric=None 配合）。
    """
    dtrain = lgb.Dataset(X, label=y, feature_name=list(feature_names), free_raw_data=False)
    valid_sets = []
    callbacks = [lgb.log_evaluation(0)]
    if X_val is not None and y_val is not None and len(y_val) > 0:
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)
        valid_sets = [dval]
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False,
                                            first_metric_only=True))
    booster = lgb.train(params, dtrain, num_boost_round=num_boost_round,
                        valid_sets=valid_sets, feval=feval, callbacks=callbacks)
    return booster, int(booster.best_iteration or num_boost_round)


class LogParamModel:
    """三目标两段式模型集合 + 预处理管线的统一封装。

    models: {target: {"clf": Booster | None, "reg": Booster, "threshold": float | None}}
    """

    # 推理覆盖用的地板值（PERM 为 log10 空间，与 predict_log 的输出约定一致）
    FLOOR_LOG = {"POR": FLOOR["POR"], "SW": FLOOR["SW"],
                 "PERM": math.log10(FLOOR["PERM"])}

    def __init__(self, feature_cols=None, models=None, scaler=None, meta=None):
        self.feature_cols = list(feature_cols or [])
        self.models = dict(models or {})
        self.scaler = scaler
        self.meta = dict(meta or {})

    # ---------- 推理 ----------
    def predict_log(self, X: pd.DataFrame) -> pd.DataFrame:
        """两段式预测：分类概率 ≥ 阈值 → 地板值，否则回归值。

        返回各目标预测值（注意：PERM 为 log10 空间，需 10^x 还原）。
        """
        X = X.reindex(columns=self.feature_cols)  # 缺列补 NaN、列序对齐
        Xs = self.scaler.transform(X.to_numpy(dtype="float32"))
        out = {}
        for t, m in self.models.items():
            reg = m["reg"].predict(Xs, num_iteration=m["reg"].best_iteration or None)
            clf, th = m.get("clf"), m.get("threshold")
            if clf is not None and th is not None:
                p = clf.predict(Xs, num_iteration=clf.best_iteration or None)
                out[t] = np.where(p >= float(th), self.FLOOR_LOG[t], reg)
            else:
                out[t] = reg
        return pd.DataFrame(out, index=X.index)

    # ---------- 持久化 ----------
    def save(self, model_path: str, scaler_path: str) -> None:
        for p in (model_path, scaler_path):
            d = os.path.dirname(os.path.abspath(p))
            os.makedirs(d, exist_ok=True)
        payload = {
            "feature_cols": self.feature_cols,
            "models": {
                t: {"clf": (m["clf"].model_to_string() if m.get("clf") is not None else None),
                    "reg": m["reg"].model_to_string(),
                    "threshold": m.get("threshold")}
                for t, m in self.models.items()},
            "meta": self.meta,
        }
        joblib.dump(payload, model_path)
        joblib.dump(self.scaler, scaler_path)

    @classmethod
    def load(cls, model_path: str, scaler_path: str) -> "LogParamModel":
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            raise FileNotFoundError(
                f"未找到模型权重 {model_path} / {scaler_path}，请先运行 train.py 完成训练")
        payload = joblib.load(model_path)
        models = {
            t: {"clf": (lgb.Booster(model_str=m["clf"]) if m.get("clf") else None),
                "reg": lgb.Booster(model_str=m["reg"]),
                "threshold": m.get("threshold")}
            for t, m in payload["models"].items()}
        return cls(feature_cols=payload["feature_cols"], models=models,
                   scaler=joblib.load(scaler_path), meta=payload.get("meta", {}))
