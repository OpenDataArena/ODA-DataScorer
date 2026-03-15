# 基于模型的数据评估框架

<p align="center">
  <a href="./README.md">English</a> | 简体中文
</p>

本框架提供了一套完整的数据质量评估系统，利用深度学习模型对数据集进行多维度评估。

## ✨ 核心特性

- **🚀 自动数据并行**: 根据全局 GPU 数量和每个评分器的 GPU 需求自动计算数据并行度
- **🎯 智能 GPU 分配**: 支持多 GPU 任务，自动为每个并行任务分配专用 GPU 资源
- **📊 双模式评分**: 区分 **Pointwise**（逐样本评分）和 **Setwise**（全集评分）两种评分模式
- **🔧 配置驱动**: 通过 YAML 配置文件轻松管理模型、评估指标和运行参数
- **💾 结构化输出**: 自动合并并分类保存评分结果，支持中间结果查看
- **🔄 并行执行**: 多个评分器按顺序执行，每个评分器内部使用数据并行加速
- **⏸️ 断点续打**: 支持从中断处恢复评估任务，避免重复计算

## 📦 支持的评分器

本框架集成了近 40 种基于模型的评分器，涵盖质量、复杂度、梯度分析等多个维度：

### 🎯 质量类

评估数据的质量、准确性、可读性等维度：

- **SkyworkLlamaScorer**: Skywork Llama 奖励模型评分
- **SkyworkQwenScorer**: Skywork Qwen 奖励模型评分
- **AtheneScorer**: Athene 奖励模型评分  
- **RMDeBERTaScorer**: DeBERTa 奖励模型评分
- **Gpt2HarmlessScorer**: GPT-2 无害性奖励模型
- **Gpt2HelpfulScorer**: GPT-2 有用性奖励模型
- **InfOrmScorer**: INF-ORM 奖励模型评分
- **DeitaQScorer**: Deita 质量评分器
- **DebertaScorer**: DeBERTa 质量分类器
- **FinewebEduScorer**: FineWeb 教育质量分类器
- **TextbookScorer**: 教科书质量分类器
- **QuRateScorer**: QuRater 多维度质量评分
- **CleanlinessScorer**: 数据清洁度评分
- **ProfessionalismScorer**: 专业性评分
- **ReadabilityScorer**: 可读性评分
- **ReasoningScorer**: 推理能力评分
- **UniEvalD2tScorer**: Data-to-Text 质量评估
- **UniEvalDialogScorer**: 对话质量评估
- **UniEvalFactScorer**: 事实准确性评估
- **UniEvalSumScorer**: 摘要质量评估

### 🧠 复杂度类

评估数据的难度、复杂度、困惑度等维度：

- **DeitaCScorer**: Deita 复杂度评分
- **ComplexityScorer**: 使用本地 LLM（非 API 形式）作为 Judge，对数据指令进行复杂度打分
- **IFDScorer**: 指令遵循难度评分
- **ThinkingProbScorer**: 思考概率评分
- **PPLScorer**: 困惑度评分
- **NormLossScorer**: 归一化损失评分
- **UPDScorer**: 不确定性与预测性难度评分

### 🔧 其他类

包括梯度分析、数据选择、特定任务等：

- **GraNdScorer**: 梯度范数差异评分
- **NuclearNormScorer**: 核范数评分
- **EffectiveRankScorer**: 有效秩评分
- **EmbedSVDEntropyScorer**: 嵌入 SVD 熵评分
- **Task2VecScorer**: Task2Vec 嵌入评分
- **MIWVScorer**: 最大权重变化值评分
- **SelectitTokenScorer**: SelectIT Token 级别评分
- **SelectitSentenceScorer**: SelectIT 句子级别评分
- **SelectitModelScorer**: SelectIT 模型集成评分
- **HESScorer**: 高熵样本评分
- **AskLlmScorer**: 基于 LLM 的质量询问
- **FailRateScorer**: 失败率评估
- **InstagScorer**: 指令标签分类

## 🚀 快速开始

### 1. 配置 YAML 文件

在 `configs/` 目录下创建或修改配置文件，例如 `configs/my_scorer.yaml`:

```yaml
# 数据路径配置
input_path: /path/to/your/data.jsonl
output_path: results/my_experiment

# 全局 GPU 配置
num_gpu: 8                    # 可用的 GPU 总数
num_gpu_per_job: 1            # 每个任务默认使用的 GPU 数量（可被评分器覆盖）

# 评分器配置
scorers:
  # 示例 1: 质量评分器（使用 1 个 GPU）
  - name: DeitaQScorer
    model: /path/to/deita-quality-scorer
    max_length: 2048
    batch_size: 8
    num_gpu_per_job: 1
  
  # 示例 2: 多 GPU 任务（使用 4 个 GPU）
  - name: InfOrmScorer
    model: /path/to/INF-ORM-Llama3.1-70B
    batch_size: 8
    max_length: 2048
    num_gpu_per_job: 4
  
  # 示例 3: 多个评分器
  - name: PPLScorer
    model: /path/to/language-model
    batch_size: 16
    max_length: 2048
    num_gpu_per_job: 1
```

**配置说明**:
- **`input_path`**: 输入数据文件路径（JSONL 格式）
- **`output_path`**: 输出结果目录
- **`num_gpu`**: 全局可用 GPU 总数（必需）
- **`num_gpu_per_job`**: 全局默认的每任务 GPU 数量（可选，默认 1）
- **`resume`**: 是否启用断点续打（可选，默认 false），详见下方「断点续打」说明
- **`scorers`**: 评分器列表，每个评分器可指定自己的 `num_gpu_per_job` 覆盖全局设置

### 2. 准备数据

确保输入数据为 JSONL 格式，每行一个 JSON 对象：

```json
{"instruction": "What is machine learning?", "output": "Machine learning is...",...}
{"instruction": "Explain neural networks", "output": "Neural networks are...",...}
```

**字段要求**:
- `instruction`: 问题或指令（必需）
- `output`: 回答或输出（对于 QA 类评分器必需）
- `input`: 额外的输入字段（可选）
- `id`: 数据唯一标识（**使用断点续打功能时必需**，用于区分和恢复每条数据）
- 其他字段: 某些评分器可能还要求其它特定字段，具体请参考对应评分器的说明或配置文档

### 3. 运行评估

```bash
python main.py --config configs/my_scorer.yaml
```

**参数说明**:
- `--config`: YAML 配置文件路径

### 4. 断点续打

当评估任务因中断（如 OOM、机器故障等）未完成时，可使用断点续打功能从中断处继续，避免重复计算已完成的部分。

**使用方法**：在配置 YAML 中增加 `resume: true`：

```yaml
input_path: /path/to/your/data.jsonl
output_path: results/my_experiment
num_gpu: 8
resume: true                    # 启用断点续打

scorers:
  - name: DeitaQScorer
    model: /path/to/deita-quality-scorer
    batch_size: 8
```

**重要说明**：
- 使用断点续打时，**输入数据中的每条记录必须包含 `id` 字段**，用于唯一标识数据并正确恢复进度
- 框架会检查 `output_path` 下已有的中间结果，跳过已完成的评分器或已完成的数据分片，仅对未完成部分继续计算

## 🔧 数据并行机制

### 自动并行度计算

框架会自动计算每个评分器的数据并行度：

```
data_parallel = num_gpu ÷ num_gpu_per_job
```

**示例**:
- 全局有 8 个 GPU
- 评分器 A 需要 1 个 GPU → data_parallel = 8（数据分 8 份并行处理）
- 评分器 B 需要 4 个 GPU → data_parallel = 2（数据分 2 份并行处理）
- 评分器 C 需要 2 个 GPU → data_parallel = 4（数据分 4 份并行处理）

### GPU 分配策略

1. **数据分割**: 将数据集分成 `data_parallel` 份
2. **进程启动**: 为每份数据启动一个独立进程
3. **GPU 分配**: 每个进程通过 `CUDA_VISIBLE_DEVICES` 看到分配给它的 GPU
4. **并行执行**: 所有进程同时运行，互不干扰
5. **结果合并**: 所有进程完成后自动合并结果

**示例: 8 GPU, 评分器需要 2 GPU per job**
```
Job 0: GPU [0, 1] → 处理数据分片 0
Job 1: GPU [2, 3] → 处理数据分片 1
Job 2: GPU [4, 5] → 处理数据分片 2
Job 3: GPU [6, 7] → 处理数据分片 3
```

## 📤 输出结果

运行完成后，在 `output_path` 目录下会生成以下文件：

### Pointwise 评分结果 (`pointwise_scores.jsonl`)

逐样本的评分结果，每行对应输入数据的一条记录：

```json
{
  "id": 0,
  "scores": {
    "DeitaQScorer": {
      "score": 4.5
    },
    "PPLScorer": {
      "score": 12.34
    },
    "UPDScorer": {
      "score": 0.79
    }
  }
}
```

### Setwise 评分结果 (`setwise_scores.jsonl`)

对整个数据集的评分结果（如果有评分器返回整体评分）：

```json
{
  "Task2VecScorer": {
    "score": 0.024327838269528,
    "num_samples": 32,
    "num_anomalous": 0,
    "num_truncated": 31,
    "truncation_rate": 0.96875,
    "last_layer_only": true,
    "embedding_dim": 768
  }
}
```

### 中间结果 (`master_temp/`)

```
master_temp/
├── processed_data.jsonl              # 预处理后的数据（添加了 id）
├── scorer_DeitaQScorer/              # 每个评分器的临时目录
│   ├── job_0/                        # 每个并行任务的结果
│   │   └── DeitaQScorer.jsonl
│   ├── job_1/
│   │   └── DeitaQScorer.jsonl
│   └── DeitaQScorer_merged.jsonl    # 合并后的结果
└── scorer_PPLScorer/
    └── PPLScorer.jsonl
```

### 后处理（可选）

- `pointwise_scores.jsonl` 已包含 `id` 和 `scores`，可直接用于下游分析。
- 如需将评分合并回原始数据文件（保留 `instruction`、`output` 等字段），可编写脚本按 `id` 匹配合并。
- 若同时使用 [llm_as_judge](../llm_as_judge/README_zh-CN.md) 模块，可参考其 `tools/process_scores.py` 的实现思路进行分数合并。

## 📝 评分器配置详解

### 通用参数

所有评分器都支持以下参数：

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `name` | string | 评分器名称（必需） | - |
| `num_gpu_per_job` | int | 此评分器需要的 GPU 数量 | 1 |

### 基于模型的评分器参数

大多数基于深度学习模型的评分器支持：

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `model` | string | 模型路径或 HuggingFace 模型名称 | - |
| `batch_size` | int | 批处理大小 | 8 |
| `max_length` | int | 最大序列长度 | 2048 |
| `num_gpu_per_job` | int | 此评分器需要的 GPU 数量 | 1 |

### 特定评分器参数

详细的评分器配置请参考：
- **配置示例**: `configs/MultiScorer.yaml`（包含所有基于模型的评分器的完整配置）
- **在线文档**: [Wiki 页面](https://opendataarena-tool.readthedocs.io/en/latest/model-based-evaluation/)

## 🎯 使用场景示例

### 场景 1: 快速质量评估

评估数据集的基础质量指标：

```yaml
num_gpu: 4
scorers:
  - name: DeitaQScorer    # 质量
  - name: DeitaCScorer    # 复杂度  
  - name: PPLScorer       # 困惑度
```

### 场景 2: 全面数据分析

使用多个维度的评分器进行深度分析：

```yaml
num_gpu: 8
scorers:
  - name: SkyworkLlamaScorer       # 质量
  - name: IFDScorer                # 难度
  - name: PPLScorer                # 困惑度
  - name: UPDScorer                # 不确定性难度
```

### 场景 3: 大模型评估

使用大规模模型进行评估：

```yaml
num_gpu: 16
scorers:
  - name: InfOrmScorer
    model: /path/to/INF-ORM-Llama3.1-70B
    num_gpu_per_job: 8              # 使用 8 GPU → data_parallel=2
    batch_size: 4
```

### 场景 4: 数据选择优化

用于数据选择和过滤：

```yaml
num_gpu: 8
scorers:
  - name: HESScorer                # 高熵样本
  - name: SelectitTokenScorer      # SelectIT Token 评分
  - name: SelectitSentenceScorer   # SelectIT 句子评分
  - name: GraNdScorer              # 梯度范数差异
```

## ⚙️ 高级功能

### 自定义评分器

1. 继承 `BaseScorer` 类：

```python
from scorers.base_scorer import BaseScorer
from typing import Dict, List, Any

class MyCustomScorer(BaseScorer):
    def _validate_config(self):
        """验证配置参数"""
        required = ["model", "batch_size"]
        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing required config: {key}")
    
    def _setup(self):
        """初始化模型和资源"""
        self.model = load_model(self.config["model"])
    
    def score_item(self, data_item: Dict) -> Dict:
        """评分单个样本"""
        score = self.model.score(data_item["instruction"])
        return {"custom_score": score}
    
    def evaluate(self, dataset_path: str) -> List[Dict]:
        """评估整个数据集"""
        results = []
        with open(dataset_path, "r") as f:
            for line in f:
                item = json.loads(line)
                score = self.score_item(item)
                score["id"] = item["id"]
                results.append(score)
        return results
```

2. 在 `scorers/scores_info.json` 中注册：

```json
{
  "name": "MyCustomScorer",
  "module": "scorers.MyCustomScorer"
}
```

3. 在配置文件中使用：

```yaml
scorers:
  - name: MyCustomScorer
    model: /path/to/model
    batch_size: 8
```


## 📚 参考资料

- **配置示例**: `configs/MultiScorer.yaml` - 包含所有基于模型的评分器的完整配置
- **在线文档**: [https://opendataarena-tool.readthedocs.io](https://opendataarena-tool.readthedocs.io)
- **评分器详解**: 访问 Wiki 页面了解每个评分器的详细说明和论文引用


## 🤝 贡献

欢迎提交 Issue 和 Pull Request！
