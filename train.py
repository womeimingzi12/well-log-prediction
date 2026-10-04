# -*- coding: utf-8 -*-
"""训练脚本：读取训练井 → 逐井特征工程 → 按井分组交叉验证（赛题指标）
→ 全量重训 → 保存 models/model.pkl 与 models/scaler.pkl。

用法（在本文件所在目录执行）：
    python train.py                                # 使用 config.yaml 默认配置
    python train.py --data_dir ./data/train
    python train.py --fast --max_wells 8           # 小规模冒烟测试（验证流程用）

训练产物：
    models/model.pkl   三个目标（POR / log10(PERM) / SW）的 LightGBM 模型
    models/scaler.pkl  特征插补 + 标准化管线
    cv_report.json     交叉验证报告（本地赛题指标，用于评估效果）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import GroupKFold

from src.feature import TARGETS, build_features
from src.model import LogParamModel, make_lgb_params, make_scaler, train_booster
from src.utils import (acc_perm, acc_por, acc_sw, clean_sentinels,
                       discover_well_files, load_config, read_well_file,
                       set_seed, total_score)

METRIC = {"POR": acc_por, "PERM": acc_perm, "SW": acc_sw}


def parse_args():
    ap = argparse.ArgumentParser(description="测井储层参数预测 - 训练脚本")
    ap.add_argument("--data_dir", default=None,
                    help="训练井 txt 目录（默认取 config.yaml 的 data.train_dir）")
    ap.add_argument("--config", default="config.yaml", help="配置文件路径")
    ap.add_argument("--models_dir", default="models", help="模型输出目录")
    ap.add_argument("--cv_report", default="cv_report.json", help="交叉验证报告输出路径")
    ap.add_argument("--n_folds", type=int, default=None, help="覆盖交叉验证折数")
    ap.add_argument("--num_boost_round", type=int, default=None, help="覆盖最大迭代轮数")
    ap.add_argument("--fast", action="store_true",
                    help="冒烟测试模式：2 折 + 少量轮数，仅用于验证流程")
    ap.add_argument("--max_wells", type=int, default=None,
                    help="仅使用排序后的前 N 口井（冒烟测试用）")
    return ap.parse_args()


def load_train_dataset(data_dir: str, cfg: dict):
    """读取全部训练井，返回 (合并 DataFrame, 井 ID 列表)。"""
    files = discover_well_files(data_dir)
    if not files:
        raise FileNotFoundError(f"{data_dir} 下未发现任何 .txt 井文件")
    frames, well_ids = [], []
    for igo_id, path in files:
        df = read_well_file(path, cfg["data"].get("encoding", "utf-8"))
        df = clean_sentinels(df, cfg["data"].get("missing_sentinels"))
        missing = [t for t in TARGETS if t not in df.columns]
        if missing:
            print(f"[跳过] {igo_id} 缺少目标列 {missing}，不参与训练")
            continue
        df["well_id"] = igo_id
        frames.append(df)
        well_ids.append(igo_id)
    if not frames:
        raise RuntimeError("没有可用的训练井（缺少 POR/PERM/SW 目标列）")
    data = pd.concat(frames, ignore_index=True)
    return data, well_ids


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["model"]["seed"])
    set_seed(seed)

    if args.fast:  # 冒烟测试模式
        cfg["train"]["n_folds"] = 2
        cfg["model"]["num_boost_round"] = 60
        cfg["model"]["early_stopping_rounds"] = 20

    data_dir = args.data_dir or cfg["data"]["train_dir"]
    data, well_ids = load_train_dataset(data_dir, cfg)
    if args.max_wells:
        well_ids = sorted(well_ids)[: args.max_wells]
        data = data[data["well_id"].isin(well_ids)].reset_index(drop=True)
    n_rows = len(data)
    print(f"[数据] 井数={len(well_ids)}  总采样点={n_rows}")

    # ---- 逐井特征工程（井内无状态，无泄漏）----
    feat_frames, tgt_frames, gid_frames = [], [], []
    well_code = {w: i for i, w in enumerate(sorted(well_ids))}
    for wid, sub in data.groupby("well_id", sort=True):
        sub = sub.sort_values("DEPTH").reset_index(drop=True)
        feat_frames.append(
            build_features(sub, rolling_windows=cfg["features"]["rolling_windows"]))
        tgt_frames.append(sub[TARGETS])
        gid_frames.append(np.full(len(sub), well_code[wid], dtype=np.int32))
    X_all = pd.concat(feat_frames, ignore_index=True).astype("float32")
    Y_all = pd.concat(tgt_frames, ignore_index=True)
    groups = np.concatenate(gid_frames)
    feature_cols = list(X_all.columns)  # 含 DEPTH
    print(f"[特征] 特征数={len(feature_cols)}")

    # ---- 训练目标（物理范围外视为脏值剔除；PERM 在 log10 空间与赛题指标对齐）----
    trange = cfg["targets"]
    eps_floor = float(trange["perm_log_floor"])

    def _clean(t: str, arr) -> np.ndarray:
        lo, hi = trange[f"{t.lower()}_range"]
        v = np.array(arr, dtype=float, copy=True)  # 显式拷贝，兼容 pandas 3.x 只读视图
        v[(v < lo) | (v > hi)] = np.nan
        return v

    y_raw = {
        "POR": _clean("POR", Y_all["POR"].to_numpy(dtype=float)),
        "PERM": _clean("PERM", Y_all["PERM"].to_numpy(dtype=float)),
        "SW": _clean("SW", Y_all["SW"].to_numpy(dtype=float)),
    }
    y_log = {
        "POR": y_raw["POR"],
        "PERM": np.log10(np.maximum(y_raw["PERM"], eps_floor)),
        "SW": y_raw["SW"],
    }
    for t in TARGETS:
        n_nan = int(np.isnan(y_log[t]).sum())
        if n_nan:
            print(f"[目标] {t} 缺失 {n_nan} 个深度点（训练时按目标剔除）")

    params = make_lgb_params(cfg["model"].get("lgb_params"), seed,
                             int(cfg["model"]["num_threads"]))
    n_rounds = int(args.num_boost_round or cfg["model"]["num_boost_round"])
    es_rounds = int(cfg["model"]["early_stopping_rounds"])
    n_folds = int(args.n_folds or cfg["train"]["n_folds"])

    # ---- 按井分组 K 折交叉验证（早停 + 赛题指标）----
    oof = {t: np.full(n_rows, np.nan) for t in TARGETS}
    best_iters = {t: [] for t in TARGETS}
    gkf = GroupKFold(n_splits=n_folds)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_all, groups=groups)):
        scaler = make_scaler().fit(X_all.iloc[tr_idx].to_numpy(dtype="float32"))
        Xtr = scaler.transform(X_all.iloc[tr_idx].to_numpy(dtype="float32"))
        Xva = scaler.transform(X_all.iloc[va_idx].to_numpy(dtype="float32"))
        for t in TARGETS:
            m_tr = ~np.isnan(y_log[t][tr_idx])
            m_va = ~np.isnan(y_log[t][va_idx])
            booster, best_it = train_booster(
                Xtr[m_tr], y_log[t][tr_idx][m_tr], params,
                X_val=Xva[m_va], y_val=y_log[t][va_idx][m_va],
                num_boost_round=n_rounds, early_stopping_rounds=es_rounds,
                feature_names=feature_cols)
            oof[t][va_idx[m_va]] = booster.predict(Xva[m_va], num_iteration=best_it)
            best_iters[t].append(best_it)
        print(f"[CV] fold {fold + 1}/{n_folds} 完成")

    # ---- 本地赛题指标（折外预测）----
    accs = {}
    for t in TARGETS:
        m = ~np.isnan(oof[t])
        y_true_t = y_raw[t][m]
        pred_t = oof[t][m] if t != "PERM" else np.power(10.0, oof[t][m])
        accs[t] = METRIC[t](y_true_t, pred_t)
    score = total_score(accs["POR"], accs["PERM"], accs["SW"])
    print(f"[CV] AccPOR={accs['POR']:.4f}  AccPERM={accs['PERM']:.4f}  "
          f"AccSW={accs['SW']:.4f}  Total={score:.2f}")

    # ---- 全量重训（轮数取各目标 CV 最优轮数中位数，固定轮数保证确定性）----
    final_scaler = make_scaler().fit(X_all.to_numpy(dtype="float32"))
    X_full = final_scaler.transform(X_all.to_numpy(dtype="float32"))
    boosters = {}
    for t in TARGETS:
        rounds_t = int(np.median(best_iters[t]))
        m = ~np.isnan(y_log[t])
        booster, _ = train_booster(X_full[m], y_log[t][m], params,
                                   num_boost_round=rounds_t,
                                   feature_names=feature_cols)
        boosters[t] = booster
        print(f"[最终模型] {t}: 轮数={rounds_t}")

    meta = {
        "seed": seed,
        "n_folds": n_folds,
        "n_wells": len(well_ids),
        "n_rows": n_rows,
        "n_features": len(feature_cols),
        "best_iterations": {t: int(np.median(best_iters[t])) for t in TARGETS},
        "cv_accuracy": accs,
        "cv_total_score": score,
        "perm_log_floor": eps_floor,
    }
    model = LogParamModel(feature_cols=feature_cols, boosters=boosters,
                          scaler=final_scaler, meta=meta)
    model.save(os.path.join(args.models_dir, "model.pkl"),
               os.path.join(args.models_dir, "scaler.pkl"))
    print(f"[保存] {args.models_dir}/model.pkl, {args.models_dir}/scaler.pkl")

    with open(args.cv_report, "w", encoding="utf-8") as f:
        json.dump({"cv_accuracy": accs, "cv_total_score": score,
                   "best_iterations": meta["best_iterations"],
                   "n_wells": len(well_ids), "n_rows": n_rows,
                   "n_features": len(feature_cols)},
                  f, ensure_ascii=False, indent=2)
    print(f"[保存] {args.cv_report}")


if __name__ == "__main__":
    main()
