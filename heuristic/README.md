# Heuristic-based Data Evaluation Framework

<p align="center">
  English | <a href="./README_zh-CN.md">简体中文</a>
</p>

This framework provides a complete data quality evaluation system that performs multi-dimensional assessment of datasets using statistical methods and heuristic algorithms.

## ✨ Core Features

- **🚀 CPU Multi-process Parallelism**: Efficient CPU parallel computing using `ProcessPoolExecutor`
- **📊 Dual Scoring Modes**: Distinguishes between **Pointwise** (per-sample scoring) and **Setwise** (whole-dataset scoring) modes
- **🔧 Configuration-Driven**: Easily manage evaluation metrics and runtime parameters through YAML configuration files
- **💾 Structured Output**: Unified result storage without data sharding for easy viewing and processing
- **🔄 Sequential Execution**: Multiple scorers execute sequentially, with each scorer using multi-process parallelism for acceleration
- **⚡ Efficient Computation**: Based on heuristic algorithms, no GPU required, high computational efficiency

## 📦 Supported Scorers

This framework integrates **25 heuristic scorers** covering diversity, statistical features, content detection, and more:

### 📈 Diversity

Evaluate dataset diversity, coverage, uniqueness, and related dimensions:

- **VendiScorer**: Vendi Score diversity metric
- **KNNScorer**: K-Nearest Neighbors diversity scoring
- **ApsScorer**: Average Pairwise Similarity
- **ApjsScorer**: Average Jaccard Similarity
- **RadiusScorer**: Data radius scoring
- **ClusterInertiaScorer**: Cluster inertia scoring
- **PartitionEntropyScorer**: Partition entropy scoring
- **NovelSumScorer**: Novelty and representativeness scoring
- **FacilityLocationScorer**: Facility location function scoring
- **UniqueNgramScorer**: N-gram uniqueness scoring
- **UniqueNtokenScorer**: N-token uniqueness scoring
- **MtldScorer**: Lexical diversity metric
- **VocdDScorer**: Vocabulary density D-value
- **TokenEntropyScorer**: Token entropy scoring
- **GramEntropyScorer**: N-gram entropy scoring
- **HddScorer**: HD-D diversity scoring

### 📊 Statistical Features

Evaluate basic statistical features of data:

- **TokenLengthScorer**: Token length statistics
- **StrLengthScorer**: String length statistics
- **TreeInstructScorer**: Syntax tree statistics
- **LogDetDistanceScorer**: Log-determinant distance scoring
- **LogicalWordCountScorer**: Logical word count
- **CompressRatioScorer**: Compression ratio scoring

### 🔍 Content Detection

Detect specific content features in data:

- **ThinkOrNotScorer**: Thinking content detection
- **PureThinkScorer**: Pure thinking content detection
- **TsPythonScorer**: Python code detection

## 🚀 Quick Start

### 1. Configure YAML File

Create or modify a configuration file in the `configs/` directory, e.g., `configs/my_scorer.yaml`:

```yaml
# Data path configuration
input_path: /path/to/your/data.jsonl
output_path: results/my_experiment

# Global GPU configuration
num_gpu: 0
num_gpu_per_job: 0

# Scorer configuration
scorers:
  # Example 1: Token length statistics
  - name: TokenLengthScorer
    encoder: o200k_base
    fields: ["instruction", "input", "output"]
    max_workers: 128
  
  # Example 2: Diversity evaluation
  - name: VendiScorer
    embedding_path: ../data_process/mock_embedding_128x512.npy
    similarity_metric: euclidean
    max_workers: 128
  
  # Example 3: KNN diversity
  - name: KNNScorer
    k: 10
    distance_metric: cosine
    max_workers: 128
    embedding_path: ../data_process/mock_embedding_128x512.npy
```

**Configuration Details**:
- **`input_path`**: Input data file path (JSONL format)
- **`output_path`**: Output results directory
- **`num_gpu`**: Total number of globally available GPUs (optional, default 0, heuristic scoring typically doesn't require GPU)
- **`num_gpu_per_job`**: Global default GPUs per task (optional, default 0, heuristic scoring typically doesn't require GPU)
- **`scorers`**: List of scorers, each can specify `max_workers` to control parallelism
- **`resume`**: Whether to enable resume from checkpoint (optional, default false). When enabled, the program skips already completed samples and continues from where it left off. **Note: When using this feature, each record in the input data must contain an `id` field for unique identification**

### 2. Prepare Data

Ensure input data is in JSONL format, with one JSON object per line:

```json
{"instruction": "What is machine learning?", "output": "Machine learning is...",...}
{"instruction": "Explain neural networks", "output": "Neural networks are...",...}
```

**Field Requirements**:
- `instruction`: Question or instruction (required)
- `output`: Answer or output (required for QA-type scorers)
- `input`: Additional input field (optional)
- `id`: Unique identifier for each record (**required when using resume feature**, used to identify and skip already completed samples)
- Other fields: Some scorers may require additional specific fields; please refer to the corresponding scorer documentation

### 3. Run Evaluation

```bash
python main.py --config configs/my_scorer.yaml
```

**Parameter Description**:
- `--config`: Path to YAML configuration file

### 4. Resume from Checkpoint

When evaluating large-scale datasets, if the program is interrupted midway, you can use the **resume from checkpoint** feature to continue from where it left off, avoiding redundant computation.

**How to use**: Add `resume: true` to your configuration YAML file:

```yaml
input_path: /path/to/your/data.jsonl
output_path: results/my_experiment
resume: true   # Enable resume from checkpoint

scorers:
  - name: TokenLengthScorer
    encoder: o200k_base
    fields: ["instruction", "input", "output"]
```

**Important notes**:
- When using resume, **each record in the input data must contain an `id` field** to uniquely identify each sample, so the program can correctly skip already completed samples
- If the `id` field is missing from the data, the program cannot correctly identify scored samples, which may lead to redundant computation or incorrect results

**Data format example** (with id required):

```json
{"id": "sample_001", "instruction": "What is ML?", "output": "Machine learning is..."}
{"id": "sample_002", "instruction": "Explain NN", "output": "Neural networks are..."}
```

## 🔧 CPU Parallelism Mechanism

### Parallel Processing Description

Heuristic scorers use **CPU parallel processing**, implementing multi-process parallelism through `ProcessPoolExecutor`

### Parallel Execution Flow

1. **Initialization**: Create a process pool of size `max_workers`
2. **Task Assignment**: Dynamically assign data items to idle worker processes
3. **Parallel Computation**: Multiple processes handle different data items simultaneously
4. **Result Collection**: Collect computation results from all processes and save them uniformly

**Note**: Unlike model-based scorers, heuristic scorers do not shard data into different job directories.

## 📤 Output Results

After completion, the following files will be generated in the `output_path` directory:

### Pointwise Scoring Results (`pointwise_scores.jsonl`)

Per-sample scoring results, each line corresponds to one record in the input data:

```json
{
  "id": 0,
  "scores": {
    "TokenLengthScorer": {
      "score": 120
    },
    "StrLengthScorer": {
      "score": 518
    },
    "ThinkOrNotScorer": {
      "score": 1
    }
  }
}
```

### Setwise Scoring Results (`setwise_scores.jsonl`)

Scoring results for the entire dataset:


```json
{
  "ApjsScorer": {
    "score": 0.16716303774426786,
    "num_samples": 30,
    "num_pairs": 435,
    "total_possible_pairs": 435,
    "is_sampled": false,
    "tokenization_method": "gram",
    "n": 1,
    "similarity_method": "direct",
    "max_workers": 128
  },
  "ApsScorer": {
    "score": 0.45395749064657004,
    "num_samples": 30,
    "num_pairs": 435,
    "total_possible_pairs": 435,
    "is_sampled": false,
    "similarity_metric": "euclidean",
    "max_workers": 128
  }
}
```

### Intermediate Results (`master_temp/`)

```
master_temp/
├── processed_data.jsonl              # Preprocessed data (with id added)
├── scorer_TokenLengthScorer/         # Temporary directory for each scorer
│   └── TokenLengthScorer.jsonl       # Scoring results
└── scorer_VendiScorer/
    └── VendiScorer.jsonl              # Scoring results
```


## 📝 Scorer Configuration Details

### Common Parameters

All scorers support the following parameters:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `name` | string | Scorer name (required) | - |
| `max_workers` | int | Maximum number of parallel worker processes | 128 |
| `num_gpu` | int | Total number of GPUs required (heuristic scorers typically set to 0) | 0 |
| `num_gpu_per_job` | int | Number of GPUs required per data shard task (heuristic scorers typically set to 0) | 0 |



### Specific Scorer Parameters

For detailed scorer configurations, please refer to:
- **Configuration Examples**: Configuration files in the `configs/` directory
- **Online Documentation**: [Wiki Page](https://opendataarena-tool.readthedocs.io/en/latest/heuristic-evaluation/)

## 🎯 Usage Scenario Examples

### Scenario 1: Basic Statistical Analysis

Evaluate basic statistical features of a dataset:

```yaml
scorers:
  - name: TokenLengthScorer        # Token length statistics
    encoder: o200k_base
    fields: ["instruction", "output"]
  - name: StrLengthScorer          # String length statistics
  - name: TreeInstructScorer       # Syntax tree statistics
```

### Scenario 2: Diversity Evaluation

Evaluate data diversity using scorers from multiple dimensions:

```yaml
scorers:
  - name: VendiScorer              # Vendi diversity
  - name: KNNScorer                # KNN diversity
    k: 10
  - name: ApsScorer                # Average pairwise similarity
  - name: UniqueNgramScorer        # N-gram uniqueness
```

### Scenario 3: Lexical Diversity Analysis

Evaluate lexical richness and diversity of data:

```yaml
scorers:
  - name: MtldScorer               # Lexical diversity metric
  - name: VocdDScorer              # Vocabulary density D-value
  - name: TokenEntropyScorer       # Token entropy scoring
  - name: GramEntropyScorer        # N-gram entropy scoring
  - name: HddScorer                # HD-D diversity scoring
```

### Scenario 4: Content Detection

Detect specific content features in data:

```yaml
scorers:
  - name: ThinkOrNotScorer         # Thinking content detection
  - name: PureThinkScorer          # Pure thinking content detection
  - name: TsPythonScorer           # Python code detection
```

### Scenario 5: Comprehensive Data Analysis

Comprehensive in-depth analysis using multiple scorers:

```yaml
scorers:
  - name: VendiScorer              # Diversity
  - name: KNNScorer                # KNN diversity
  - name: TokenLengthScorer        # Length statistics
  - name: MtldScorer               # Lexical diversity
  - name: ThinkOrNotScorer         # Content detection
```

## ⚙️ Advanced Features

### Custom Scorers

1. Inherit from the `BaseScorer` class:

```python
from scorers.base_scorer import BaseScorer
from typing import Dict, List, Any
import json

class MyCustomScorer(BaseScorer):
    def _validate_config(self):
        """Validate configuration parameters"""
        required = ["max_workers"]
        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing required config: {key}")
    
    def _setup(self):
        """Initialize resources"""
        self.max_workers = self.config.get("max_workers", 128)
    
    def score_item(self, data_item: Dict) -> Dict:
        """Score a single sample"""
        # Implement custom heuristic scoring logic
        text = data_item["instruction"]
        score = len(text.split())  # Example: count words
        return {"custom_score": score}
    
    def evaluate(self, dataset_path: str) -> List[Dict]:
        """Evaluate the entire dataset"""
        results = []
        with open(dataset_path, "r") as f:
            for line in f:
                item = json.loads(line)
                score = self.score_item(item)
                score["id"] = item["id"]
                results.append(score)
        return results
```

2. Register in `scorers/scores_info.json`:

```json
{
  "name": "MyCustomScorer",
  "module": "scorers.MyCustomScorer"
}
```

3. Use in configuration file:

```yaml
scorers:
  - name: MyCustomScorer
    max_workers: 128
```


## 📚 References

- **Configuration Examples**: `configs/` directory - Contains configuration examples for various heuristic scorers
- **Online Documentation**: [https://opendataarena-tool.readthedocs.io](https://opendataarena-tool.readthedocs.io)
- **Scorer Details**: Visit the Wiki page to learn detailed descriptions of each scorer


## 🤝 Contributing

Issues and Pull Requests are welcome!
