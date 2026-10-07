# -*- coding: utf-8 -*-
"""主预测脚本：加载训练好的模型，对测试井逐井预测并输出 result.json。

用法（在本文件所在目录执行）：
    python predict.py --data_dir ./data --output ./result.json
    python predict.py --data_dir ./data/test --output ./result.json
（若 --data_dir 指向含 train/ 与 test/ 的原始数据根目录，会自动进入 test 子目录）

参数说明：
    --data_dir   测试数据目录，目录结构须与赛题下发的原始数据完全一致
    --output     预测结果 result.json 的保存路径
    --config     配置文件路径（默认 config.yaml）
    --models_dir 模型权重目录（默认 models/，由 train.py 生成）

输出 JSON 格式固定为赛题《提交规范》结构（字段禁止增删、修改）：
    {"resultData": [{"logId": "<井次ID>",
                     "predictions": [{"DEPTH","POR","PERM","SW"}]}]}
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.feature import TARGETS, build_features
from src.model import LogParamModel
from src.utils import (clean_sentinels, discover_well_files, find_test_subdir,
                       load_config, read_well_file)


def parse_args():
    ap = argparse.ArgumentParser(description="测井储层参数预测 - 推理脚本")
    ap.add_argument("--data_dir", default=None,
                    help="测试数据目录（默认取 config.yaml 的 data.test_dir）")
    ap.add_argument("--output", default="result.json", help="预测结果保存路径")
    ap.add_argument("--config", default="config.yaml", help="配置文件路径")
    ap.add_argument("--models_dir", default="models", help="模型权重目录")
    return ap.parse_args()


def format_result(rows_per_well) -> dict:
    """按赛题《提交规范》组装 result.json：字段固定，禁止增删、修改。

    {"resultData": [{"logId": <井次ID>,
                     "predictions": [{"DEPTH","POR","PERM","SW"}, ...]}, ...]}
    """
    return {"resultData": [
        {"logId": log_id,
         "predictions": [{"DEPTH": r["DEPTH"], "POR": r["POR"],
                          "PERM": r["PERM"], "SW": r["SW"]} for r in rows]}
        for log_id, rows in rows_per_well]}


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model = LogParamModel.load(os.path.join(args.models_dir, "model.pkl"),
                               os.path.join(args.models_dir, "scaler.pkl"))

    data_dir = find_test_subdir(args.data_dir or cfg["data"]["test_dir"])
    wells = discover_well_files(data_dir)
    if not wells:
        raise FileNotFoundError(f"{data_dir} 下未发现任何 .txt 井文件")
    print(f"[数据] 发现 {len(wells)} 口测试井（目录：{data_dir}）")

    sentinels = cfg["data"].get("missing_sentinels")
    clamp = cfg["predict_clamp"]

    rows_per_well = []
    for k, (log_id, path) in enumerate(wells, 1):
        df = read_well_file(path, cfg["data"].get("encoding", "utf-8"))
        df = clean_sentinels(df, sentinels)
        # 测试井应只含输入曲线；若误传入训练井（含有效目标值）则跳过
        has_target = any(t in df.columns and df[t].notna().any() for t in TARGETS)
        if has_target:
            print(f"[跳过] {log_id} 含目标列（疑似训练井），不参与预测")
            continue
        if "DEPTH" not in df.columns:
            raise KeyError(f"{log_id} 缺少 DEPTH 列")

        feat = build_features(df, rolling_windows=cfg["features"]["rolling_windows"],
                              max_depth_gap=float(cfg["features"]["max_depth_gap"]))
        pred = model.predict_log(feat)

        # 物理范围裁剪（POR/SW 为 %，PERM 由 log10 空间还原为 mD）
        pred["POR"] = pred["POR"].clip(lower=0.0, upper=float(clamp["por_upper"]))
        pred["SW"] = pred["SW"].clip(lower=0.0, upper=float(clamp["sw_upper"]))
        pred["PERM"] = np.power(10.0, pred["PERM"]).clip(
            lower=float(clamp["perm_lower"]), upper=float(clamp["perm_upper"]))

        rows = [{"DEPTH": round(float(d), 4),
                 "POR": round(float(p), 6),
                 "PERM": round(float(pm), 6),
                 "SW": round(float(s), 6)}
                for d, p, pm, s in zip(feat["DEPTH"].to_numpy(),
                                       pred["POR"].to_numpy(),
                                       pred["PERM"].to_numpy(),
                                       pred["SW"].to_numpy())]
        rows_per_well.append((log_id, rows))
        print(f"[预测] ({k}/{len(wells)}) {log_id}: {len(rows)} 个深度点")

    result = format_result(rows_per_well)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[完成] 结果已写入 {args.output}（共 {len(rows_per_well)} 口井，格式：提交规范 resultData/logId/DEPTH）")


if __name__ == "__main__":
    main()
