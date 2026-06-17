# PCVRHyFormer —— 后点转化率预测推荐模型

> 基于 Transformer + RoPE 的混合推荐模型，用于 Post-Click Conversion Rate (PCVR) 预测。  
> 支持 Namespace Tokenizer（RankMixer / GroupNS），面向大规模稀疏特征场景。

---

## 目录

- [项目概述](#项目概述)
- [模型架构](#模型架构)
- [文件结构](#文件结构)
- [环境依赖](#环境依赖)
- [数据格式](#数据格式)
- [快速开始](#快速开始)
- [训练配置说明](#训练配置说明)
- [性能优化特性](#性能优化特性)
- [常见问题](#常见问题)

---

## 项目概述

PCVRHyFormer 是一个面向**推荐系统后点转化率预测**的深度学习模型，核心特点：

- **Hybrid Transformer  backbone**：融合用户序列与物品特征，支持多域序列建模
- **RoPE（旋转位置编码）**：为序列位置提供旋转嵌入，提升长序列建模能力
- **Namespace Tokenizer**：支持 `RankMixer` 与 `GroupNS` 两种特征分组策略，将高维稀疏特征压缩为固定数量的 NS Token
- **Parquet 原生数据管道**：直接读取 Parquet 格式训练数据，支持大规模数据流式加载

本代码为**自包含基线版本**，可直接用于 KDD Cup TAAC 2026 等推荐竞赛的训练与评估。

---

## 模型架构

```
输入层
 ├── user_int_feats  ──→ Embedding ──┐
 ├── item_int_feats ──→ Embedding ──┤
 ├── user_dense      ──→ Projection ─┤
 └── seq_data (多域) ──→ Embedding + RoPE ──┤
                                          │
                                          ▼
                              NS Tokenizer (RankMixer / Group)
                                  将特征压缩为 T 个 NS Token
                                          │
                                          ▼
                              Hybrid Transformer Encoder
                                    (多层 Self-Attention)
                                          │
                                          ▼
                                      预测头
                                          │
                                          ▼
                                    PCVR 预测值 ∈ [0, 1]
```

### 核心模块（`model.py`）

| 模块 | 说明 |
|------|------|
| `RotaryEmbedding` | RoPE 预计算与缓存，支持 `torch.compile` |
| `PCVRHyFormer` | 主模型，接收 `ModelInput` NamedTuple |
| `ModelInput` | 模型输入结构，包含用户特征、物品特征、序列数据、序列长度、时间分桶 |
| NS Tokenizer | `RankMixerNSTokenizer` / `GroupNSTokenizer` 二选一 |

### 超参数约束

`d_model` 必须能被 `T` 整除，其中：

```
T = num_queries × num_sequences + num_ns
```

- `num_sequences` = 4（seq_a / seq_b / seq_c / seq_d）
- `num_ns` = 用户 NS 组数 + 1（user_dense Token）+ 物品 NS 组数
- 例：`num_queries=2`，`num_ns=12` → `T=20`，`d_model` 需为 20 的倍数（如 60、80）

---

## 文件结构

```
推荐系统/
├── dataset.py        # Parquet 数据加载，FeatureSchema，IterableDataset
├── model.py          # PCVRHyFormer 模型定义（RoPE + Transformer）
├── trainer.py        # 训练器（训练循环、评估、EarlyStopping）
├── train.py          # 训练入口（CLI 参数解析、数据准备、启动训练）
├── utils.py          # 工具函数（set_seed、EarlyStopping、logger）
├── ns_groups.json    # NS 分组配置示例（GroupNS 模式用）
└── run.sh           # 启动脚本（两种配置示例）
```

| 文件 | 行数 | 核心职责 |
|------|------|----------|
| `model.py` | ~1500 | 模型架构，所有 NN 模块 |
| `trainer.py` | ~600 | 训练/验证循环，指标计算 |
| `train.py` | ~500 | 参数解析，特征构建，主流程 |
| `dataset.py` | ~800 | Parquet 读取，FeatureSchema，批处理 |
| `utils.py` | ~300 | 通用工具 |

---

## 环境依赖

```bash
torch >= 2.0          # 支持 torch.compile
pyarrow >= 12.0        # Parquet 读取
numpy
tqdm                    # 进度条
```

### 安装

```bash
pip install torch pyarrow numpy tqdm
```

> **注意**：`torch.compile` 需要 PyTorch 2.0+，可显著加速训练。若环境不支持，代码会自动回退。

---

## 数据格式

训练数据为 **Parquet 格式**，需配套 `schema.json` 描述特征维度。

### 目录结构

```
data_dir/
├── train_0.parquet
├── train_1.parquet
├── ...
└── schema.json
```

### `schema.json` 格式

```json
{
  "user_int_feats_dim": 128,
  "item_int_feats_dim": 64,
  "user_dense_feats_dim": 16,
  "item_dense_feats_dim": 8,
  "seq_features": {
    "seq_a": {"max_len": 50, "feat_dim": 32},
    "seq_b": {"max_len": 50, "feat_dim": 32},
    "seq_c": {"max_len": 20, "feat_dim": 32},
    "seq_d": {"max_len": 20, "feat_dim": 32}
  },
  "num_time_buckets": 128
}
```

### Parquet 列说明

| 列名 | 类型 | 说明 |
|------|------|------|
| `user_int_feats` | int64[n] | 用户离散特征 |
| `item_int_feats` | int64[n] | 物品离散特征 |
| `user_dense_feats` | float32[n] | 用户连续特征 |
| `item_dense_feats` | float32[n] | 物品连续特征 |
| `seq_a` | int64[n, L] | 域 a 行为序列 |
| `seq_a_len` | int32 | 序列 a 有效长度 |
| `seq_a_time` | int32[L] | 序列 a 时间分桶 |
| `label` | float32 | 转化率标签（0/1） |

---

## 快速开始

### 方式一：用 `run.sh`（推荐）

```bash
# 默认配置：RankMixer NS Tokenizer，d_model=64
bash run.sh

# 追加参数（覆盖默认值）
bash run.sh --num_epochs 20 --batch_size 512 --lr 3e-4
```

### 方式二：直接调用 `train.py`

```bash
python train.py \
    --data_dir /path/to/data \
    --ckpt_dir ./ckpt \
    --log_dir ./logs \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --batch_size 256 \
    --lr 1e-4 \
    --num_epochs 10 \
    --device cuda
```

### 环境变量（优先级高于 CLI）

```bash
export TRAIN_DATA_PATH=/path/to/data
export TRAIN_CKPT_PATH=./ckpt
export TRAIN_LOG_PATH=./logs
python train.py
```

---

## 训练配置说明

### NS Tokenizer 两种模式

#### 1. RankMixer 模式（默认，`run.sh` 主配置）

每个 Namespace 固定映射为 `K` 个可学习 Token，简单高效：

```bash
python train.py \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2
```

#### 2. GroupNS 模式（需 `ns_groups.json`）

按特征语义分组，每组生成一个 NS Token，需保证 `d_model % T == 0`：

```bash
python train.py \
    --ns_tokenizer_type group \
    --ns_groups_json ./ns_groups.json \
    --num_queries 1
```

`ns_groups.json` 中定义了用户侧 7 个组 + 物品侧 4 个组，可根据实际特征 schema 调整。

---

## 性能优化特性

`dataset.py` 中包含多项工程优化，适合大规模数据训练：

| 优化项 | 说明 |
|--------|------|
| 预分配 numpy buffer | 消除 `np.zeros` + `np.stack` 开销 |
| 融合 padding 循环 | 序列域一次性写入 3D buffer |
| 预计算列索引 | 避免逐行字符串查找 |
| `file_system` 共享策略 | 解决多 worker 下 `/dev/shm` 耗尽问题 |
| Parquet Row Group 切片 | 支持 `train_ratio` / `valid_ratio` 按比例切分训练/验证集 |
|  shuffle buffer | 以 batch 为单位 shuffle，降低内存占用 |

---

## 常见问题

### Q: `d_model` 设多少合适？

根据 `T = num_queries × 4 + num_ns` 计算，取 `d_model` 为 `T` 的倍数。  
例：`num_queries=2`，`num_ns=12` → `T=20` → `d_model=60/80/100...`

### Q: 如何调整 NS 分组？

修改 `ns_groups.json` 中的 `user_ns_groups` / `item_ns_groups`，将语义相近的 `fid` 归为一组。

### Q: 数据加载慢怎么办？

- 增大 `--num_workers`（推荐 8~16）
- 降低 `--buffer_batches`（减少 shuffle 内存）
- 使用 SSD 存储 Parquet 文件

### Q: CUDA OOM？

- 降低 `--batch_size`
- 降低 `d_model` 或 `num_queries`
- 缩短 `max_seq_len`（修改 `schema.json`）

---

## 引用 / 相关竞赛

- **KDD Cup 2026 TAAC Track**：本代码面向该竞赛的 PCVR 预测任务
- **RankMixer**：Namespace Tokenizer 参考 RankMixer 论文设计
- **RoPE**：Rotary Position Embedding（Su et al., 2021）

---

## License

MIT License
