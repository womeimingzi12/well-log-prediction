# 基于测井多尺度特征工程与 LightGBM 的储层参数预测模型（POR / PERM / SW）

## 1. 算法名称

基于 LightGBM 梯度提升树与测井多尺度特征工程的储层参数三目标回归预测模型。

## 2. 方法简介

**任务**：输入 14 条测井/井身曲线（DEPTH, DEVI, AZIM, GR, SP, AC, DEN, CNL, PE, RXO, RT, CAL, BIT, CASE），对每个深度点回归预测孔隙度 POR（%）、渗透率 PERM（mD）、含水饱和度 SW（%）。

**特征工程**（全部"逐井无状态"构建，同一函数用于训练井与测试井，不引入跨井统计量，避免泄漏）：

- 原始曲线 13 条 + 井内相对深度；
- 电阻率对数变换（RXO、RT 跨数量级）及深浅电阻率对数差（泥浆侵入程度）；
- 物理启发特征：密度孔隙度、声波孔隙度（Wyllie 时间平均式）、中子孔隙度、声密组合孔隙度；
- 方位角 sin/cos 圆变量编码；
- 双尺度井内滚动均值/标准差（±0.5 m、±2 m，按深度断层分段，不跨段计算）+ 前后一阶差分；
- 井内稳健归一化（中位数/四分位距）与井内分位数排名。

**数据清洗**：哨兵值（-99999 等）置为 NaN；目标值按物理范围过滤（范围外脏值不参与训练）；缺失曲线自动补 NaN 列，由插补管线兜底。

**模型与训练策略**：

- POR / PERM / SW 三个独立的 LightGBM（GBDT）回归器；
- **PERM 在 log10 空间训练与预测**，与赛题评分的 log10 比值指标直接对齐；
- 按井分组的 5 折 GroupKFold 交叉验证（折内早停）→ 全量重训（迭代轮数取各目标 CV 最优轮数的中位数，固定轮数保证确定性）；
- LightGBM 开启 `deterministic=True`，配合固定随机种子（`config.yaml: model.seed`）。

**预测逻辑**：加载 `models/` 权重 → 逐井构建同一套特征 → 三目标预测 → PERM 由 log10 空间还原 → 物理范围裁剪（POR∈[0,40]%，SW∈[0,100]%，PERM∈[1e-4,5000] mD）→ 输出 result.json。

## 3. 运行环境

- Python 3.10+（开发验证环境：Python 3.13.12，Windows x64）
- CPU 即可，无需 GPU；内存建议 ≥ 8 GB
- 全部依赖及对应版本见 `requirements.txt`（已锁定版本），安装：

```bash
pip install -r requirements.txt
```

## 4. 运行命令

### 4.1 推理（复现预测结果）

```bash
python predict.py --data_dir ./data --output ./result.json
```

参数说明：

- `--data_dir`：测试数据目录或文件路径，目录结构与赛题下发的原始数据完全一致（若指向含 `train/` 与 `test/` 的根目录，会自动定位其中的测试井子目录）；
- `--output`：预测结果 result.json 保存路径；
- 可选：`--config`（配置文件，默认 `config.yaml`）、`--models_dir`（权重目录，默认 `models/`）。

运行过程一次性完成数据读取、预处理、模型加载、推理预测与结果文件生成，无需人工干预。

### 4.2 训练（生成模型权重）

```bash
python train.py --data_dir ./data/train
```

- 可选：`--n_folds`、`--num_boost_round`、`--models_dir`、`--cv_report`；
- 冒烟/流程验证：`python train.py --fast --max_wells 8`（2 折 + 60 轮，几分钟内完成）；
- 产物：`models/model.pkl`（三目标模型）、`models/scaler.pkl`（插补+标准化管线）、`cv_report.json`（本地交叉验证的赛题指标：AccPOR / AccPERM / AccSW / Total）。

## 5. 文件说明

```
submission_code/
├── README.md            # 本文件：算法说明、运行方式、依赖环境
├── predict.py           # 主预测脚本（评测复现入口）
├── train.py             # 训练脚本（生成模型权重与交叉验证报告）
├── requirements.txt     # Python 依赖包列表（锁定版本）
├── config.yaml          # 模型参数、特征参数、路径、结果格式配置
├── src/
│   ├── model.py         # 算法模块：LightGBM 三目标模型封装与持久化
│   ├── feature.py       # 特征处理模块：特征工程全流程
│   └── utils.py         # 工具：井文件读取、哨兵清洗、赛题评分指标
├── models/              # 训练好的模型文件或权重（train.py 生成）
│   ├── model.pkl        # 三个目标的 LightGBM 模型（训练后生成）
│   └── scaler.pkl       # 特征插补 + 标准化管线（训练后生成）
└── examples/
    ├── result_example.json          # 官方结果格式样例（logId/depth）
    └── result_example_webpage.json  # 网页提交规范样例（igoId/DEPTH）
```

## 6. 复现说明

- 代码内部固定随机种子（`config.yaml → model.seed: 2026`），LightGBM 参数含 `deterministic=True`、`force_col_wise=True`；
- `num_threads` 默认 4，复现时请保持与提交环境一致（线程数不同可能带来浮点级微小差异）；
- 相同环境 + 相同数据 + 相同命令 → 预测结果一致（允许浮点精度导致的微小差异）；
- 训练与推理完全通过命令行完成，无需修改代码或交互式操作。

## 7. 结果 JSON 格式说明（重要）

提交以**网页提交规范**为准，`config.yaml → result.format` 已设为 `webpage`（`igoId` / `DEPTH` 口径）。数据包内《测试数据返回结果格式.json》的 `logId`/`depth` 口径也已实现，如需切换改回 `official` 即可。

| 口径 | 顶层字段 | 井次键 | 深度键 |
|---|---|---|---|
| `webpage`（默认，跟随网页提交规范） | resultData | `igoId` | `DEPTH` |
| `official`（数据包官方文件示例） | modelId / modelName / version / resultData | `logId` | `depth` |

`igoId` / `logId` 取井文件名（不含 .txt 扩展名）。默认输出格式为 `webpage`，无需额外配置。

## 8. 训练与提交流程

1. 安装依赖：`pip install -r requirements.txt`；
2. 数据放置：仓库自带 `data/train`（80 口井）与 `data/test`（10 口井）目录结构；
3. 训练：`python train.py`（默认 5 折 CV + 全量重训，CPU 约几十分钟）；
4. 查看本地指标：`cv_report.json` 中的 `cv_total_score`（综合百分制，可对照晋级线自评）；
5. 预测：`python predict.py --data_dir ./data --output ./result.json`，将 result.json 打包为 `result.zip` 上传；
6. 代码包提交：将整个目录打包为 `submission_code.zip`，**务必包含训练生成的 `models/model.pkl` 与 `models/scaler.pkl`**（.gitignore 忽略权重文件，打包前请确认两者已在目录中）。
