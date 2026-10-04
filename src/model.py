# -*- coding: utf-8 -*-
"""模型封装：POR / PERM / SW 三个独立 LightGBM 回归器。

- PERM 在 log10 空间训练与预测（与赛题评分的 log10 比值指标对齐）
- LightGBM 开启 deterministic，配合固定种子保证结果可复现
- 产物：models/model.pkl（三个 Booster + 特征列）
        models/scaler.pkl（缺失插补 + 标准化管线）
"""
from __future__ import annotations

import os

import joblib
import lightgbm as lgb
import pandas as pd

TARGETS = ["POR", "PERM", "SW"]


def make_lgb_params(extra: dict | None, seed: int, num_threads: int) -> dict:
    """构建 LightGBM 参数（内置确定性配置，可被 config.yaml 覆盖）。"""
    params = {
        "objective": "regression",
        "metric": "l1",
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
                  feature_names=None):
    """训练单个目标的 LightGBM Booster，返回 (booster, best_iteration)。"""
    dtrain = lgb.Dataset(X, label=y, feature_name=list(feature_names), free_raw_data=False)
    valid_sets = []
    callbacks = [lgb.log_evaluation(0)]
    if X_val is not None and y_val is not None and len(y_val) > 0:
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)
        valid_sets = [dval]
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
    booster = lgb.train(params, dtrain, num_boost_round=num_boost_round,
                        valid_sets=valid_sets, callbacks=callbacks)
    return booster, int(booster.best_iteration or num_boost_round)


class LogParamModel:
    """三目标回归模型集合 + 预处理管线的统一封装。"""

    def __init__(self, feature_cols=None, boosters=None, scaler=None, meta=None):
        self.feature_cols = list(feature_cols or [])
        self.boosters = dict(boosters or {})   # target -> lgb.Booster
        self.scaler = scaler
        self.meta = dict(meta or {})

    # ---------- 推理 ----------
    def predict_log(self, X: pd.DataFrame) -> pd.DataFrame:
        """返回各目标原始预测值（注意：PERM 为 log10 空间，需 10^x 还原）。"""
        X = X.reindex(columns=self.feature_cols)  # 缺列补 NaN、列序对齐
        Xs = self.scaler.transform(X.to_numpy(dtype="float32"))
        out = {}
        for t, booster in self.boosters.items():
            out[t] = booster.predict(Xs, num_iteration=booster.best_iteration or None)
        return pd.DataFrame(out, index=X.index)

    # ---------- 持久化 ----------
    def save(self, model_path: str, scaler_path: str) -> None:
        for p in (model_path, scaler_path):
            d = os.path.dirname(os.path.abspath(p))
            os.makedirs(d, exist_ok=True)
        payload = {
            "feature_cols": self.feature_cols,
            "boosters": {t: b.model_to_string() for t, b in self.boosters.items()},
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
        boosters = {t: lgb.Booster(model_str=s) for t, s in payload["boosters"].items()}
        scaler = joblib.load(scaler_path)
        return cls(feature_cols=payload["feature_cols"], boosters=boosters,
                   scaler=scaler, meta=payload.get("meta", {}))
