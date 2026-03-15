# OpenDataArena 数据评分工具

<p align="center">
  <a href="./README.md">English</a> | 简体中文
</p>

## Introduction

[OpenDataArena-Tool](https://github.com/OpenDataArena/OpenDataArena-Tool) 中的数据评分工具通过一系列自动化、多方面的评分和处理方法，为 [OpenDataArena](https://opendataarena.github.io/) 提供了多维度的评估。

## Wiki 文档

更多关于数据评分的详细信息，请参阅 [OpenDataArena-Tool 数据评分文档](https://opendataarena-tool.readthedocs.io/en/latest/)。

## 核心模块

本项目集成了各种先进的数据处理和评分技术，主要包括以下三个核心模块：

* 📊 **基于模型的评分器**: 利用模型的内部信号评估数据。本框架集成了 40 种基于模型的评分器，涵盖质量、复杂度、梯度分析等多个维度：
  * **质量类**: SkyworkLlamaScorer, SkyworkQwenScorer, AtheneScorer, RMDeBERTaScorer, Gpt2HarmlessScorer, Gpt2HelpfulScorer, InfOrmScorer, DeitaQScorer, DebertaScorer, FinewebEduScorer, TextbookScorer, QuRateScorer, CleanlinessScorer, ProfessionalismScorer, ReadabilityScorer, ReasoningScorer, UniEvalD2tScorer, UniEvalDialogScorer, UniEvalFactScorer, UniEvalSumScorer
  * **复杂度类**: DeitaCScorer, IFDScorer, ThinkingProbScorer, PPLScorer, NormLossScorer, UPDScorer, ComplexityScorer
  * **其他类**: GraNdScorer, NuclearNormScorer, EffectiveRankScorer, Task2VecScorer, MIWVScorer, SelectitTokenScorer, SelectitSentenceScorer, SelectitModelScorer, HESScorer, EmbedSVDEntropyScorer, AskLlmScorer, FailRateScorer, InstagScorer

* ⚖️ **LLM-as-a-Judge 评分器**: 利用强大的 LLM 作为 "法官"，通过模拟人类的判断来评分数据。  
  在此框架中，常用的维度有 Q、A 和 QA：
  * **Q**：表示对“问题/指令”（Question/Instruction）本身进行评价。
  * **A**：表示对“回答/生成内容”（Answer）本身进行评价。
  * **QA**：表示评价“问答对”（Question-Answer Pair）的整体质量（如答案与问题的相关性）。
  
  当前内置指标包括：
  * Difficulty（Q）：问题的难度
  * Relevance（QA）：回答与问题的相关性
  * Clarity（Q & QA）：表述清晰度
  * Coherence（Q & QA）：内容连贯性
  * Completeness（Q & QA）：信息完整性
  * Complexity（Q & QA）：复杂程度
  * Correctness（Q & QA）：内容正确性
  * Meaningfulness（Q & QA）：意义/价值

* 🧠 **启发式评分器**: 使用启发式方法评分数据。本框架集成了 25 种启发式评分器，涵盖多样性、统计特征、内容检测等多个维度：
  * **多样性类**: VendiScorer, KNNScorer, ApsScorer, ApjsScorer, RadiusScorer, ClusterInertiaScorer, PartitionEntropyScorer, NovelSumScorer, FacilityLocationScorer, UniqueNgramScorer, UniqueNtokenScorer, MtldScorer, VocdDScorer, TokenEntropyScorer, GramEntropyScorer, HddScorer
  * **统计特征类**: TokenLengthScorer, StrLengthScorer, LogicalWordCountScorer, CompressRatioScorer, TreeInstructScorer, LogDetDistanceScorer
  * **内容检测类**: ThinkOrNotScorer, PureThinkScorer, TsPythonScorer

## 安装

```bash
conda create -n oda python=3.10 -y
conda activate oda
git clone https://github.com/OpenDataArena/OpenDataArena-Tool.git
cd OpenDataArena-Tool/data_scorer
pip install -r requirements.txt
pip install flash_attn==2.7.4.post1 --no-build-isolation
# if you want to calculate fail rate, run the following command, which will install the lighteval package
cd model_based/scorers/fail_rate
pip install -e .[dev]
```

## 如何使用

首先，确保您的输入数据符合预期格式。

### 数据格式

您的原始输入数据应主要包含两个键：`instruction` 和 `output`，**每行必须是一个有效的 JSON 对象**。这意味着您的文件应为 **JSONL 格式**。

**示例:** (您也可以参考 `data_process/example_input.jsonl`)

```jsonl
{"instruction": "What is the capital of France?", "output": "Paris"}
{"instruction": "Explain the concept of quantum entanglement.", "output": "Quantum entanglement is a phenomenon where two or more particles become linked in such a way that they share the same fate, regardless of the distance between them. Measuring the state of one entangled particle instantaneously influences the state of the other(s)."}
{"instruction": "List three benefits of regular exercise.", "output": "Regular exercise improves cardiovascular health, boosts mood and reduces stress, and strengthens muscles and bones."}
```

**重要提示:**

* 如果您的原始数据包含 `input` 键（在 Alpaca 格式中很常见），您必须将 `input` 值与 `instruction` 值连接起来，使用 `\n` 作为分隔符。
* 部分 scorer 可能还需要额外的字段或特殊格式要求。请务必查阅对应 scorer 的 Wiki 或 README，获取所需字段/格式的具体说明。

### 运行数据评分脚本

本项目采用模块化结构，每个核心模块作为独立的子目录。有关运行特定评分器的详细说明，**请参考相应子目录中的 `README.md` 文件。**

