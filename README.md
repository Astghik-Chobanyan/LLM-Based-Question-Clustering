# LLM-Based Question Clustering

A pipeline for clustering customer support questions using Large Language Models (LLMs). This system extracts intents from questions and groups them into meaningful clusters using LLM techniques.

## 🎯 Overview

This project implements a two-stage clustering approach:
1. **Intent Extraction**: Extract the primary intent from each customer support question
2. **Intent Clustering**: Group similar intents into broader categories using LLM-based semantic understanding

### Key Features

- ✅ **LLM-Powered Intent Extraction**: Uses Ollama (Qwen models) for accurate intent identification
- ✅ **Iterative Clustering**: Intelligent batch-based clustering with cluster evolution
- ✅ **Comprehensive Evaluation**: Multiple metrics and visualizations for quality assessment

## 📁 Project Structure

```
LLM-Based-Question-Clustering/
├── data/
│   ├── raw/                                # Original dataset
│   │   └── Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv
│   └── processed/                          # Generated data files
│       ├── questiosn_with_extracted_intents.json
│       ├── normalized_grouped_intents.json
│       ├── llm_clustering_output.json
│       └── llm_clustering_output_with_gt.json
│
├── pipeline/                               # Core pipeline modules
│   ├── data_loader.py                      # Data loading and validation
│   ├── intent_extractor.py                 # LLM-based intent extraction
│   ├── llm_clusterer.py                    # LLM-based clustering
│   ├── analyze_intent_groups.py            # Intent grouping analysis
│   ├── evaluate_clusters.py                # Clustering evaluation metrics
│   └── dataset_analysis.py                 # Dataset exploration
│
├── prompts/                                # Prompt templates
│   ├── prompt_templates.py                 # Template loader (Singleton)
│   └── templates/                          # Prompt files
│       ├── intent_extractor.system         # System prompt for intent extraction
│       ├── intent_extractor.user           # User prompt template
│       ├── initial_clusters.system         # Initial clustering system prompt
│       ├── initial_clusters.user           # Initial clustering user prompt
│       ├── batch_clustering.system         # Batch clustering system prompt
│       └── batch_clustering.user           # Batch clustering user prompt
│
├── utils/                                  # Utility modules
│   └── singleton_meta.py                   # Singleton metaclass
│
├── results/                                # Evaluation results
│   └── eval_results.json                   # Clustering quality metrics
│
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12
- Ollama installed and running (for Qwen models)
- OpenAI API key (for GPT models)

### Installation

1. **Clone the repository**
```bash
cd LLM-Based-Question-Clustering
```

2. **Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up Ollama (if using local models)**
```bash
# Install Ollama: https://ollama.ai
ollama pull qwen3:8b
```

5. **Configure environment (if using OpenAI)**
Create a `.env` file:
```bash
OPENAI_API_KEY=your_api_key_here
```

## 📊 Usage

### 1. Data Loading

```python
from pipeline.data_loader import QuestionDataLoader

# Load dataset
loader = QuestionDataLoader("data/raw/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")

# Get all data
questions = loader.get_all_data()
```

### 2. Intent Extraction

Extract intents from questions using LLM:

```python
from pipeline.intent_extractor import IntentExtractor
from pipeline.data_loader import QuestionDataLoader

# Load data
loader = QuestionDataLoader("data/raw/your_dataset.csv")
questions = loader.get_all_data()

# Initialize extractor (uses Ollama by default)
extractor = IntentExtractor(model_name="qwen3:8b")

# Extract intents with checkpointing
results = extractor.extract_intents_batch(
    questions,
    checkpoint_file="data/processed/intent_checkpoint.json",
    checkpoint_interval=100
)

# Save results
extractor.save_results(results, "data/processed/extracted_intents.json")
```

**Key Features:**
- Automatic checkpointing every N questions
- Resume from checkpoint if interrupted
- Progress bar with ETA
- Batch processing for efficiency

### 3. Analyze Intent Groups

Group questions by extracted intent and analyze distribution:

```bash
python pipeline/analyze_intent_groups.py \
    data/processed/extracted_intents.json \
    --save data/processed/intent_groups.json \
    --top 50
```

**Command-line options:**
- `--top N`: Show only top N groups
- `--min-count N`: Filter groups with <N questions
- `--save FILE`: Save analysis to JSON

### 4. LLM-Based Clustering

Cluster intents into broader categories:

```python
from pipeline.llm_clusterer import LLMClusterer

# Initialize clusterer
clusterer = LLMClusterer()

# Load grouped intents
intent_groups = clusterer.load_analysis(
    "data/processed/intent_groups.json",
    min_questions=50
)

# Cluster intents
clusterer.cluster_intents(intent_groups, batch_size=100)

# Save results
clusterer.save_detailed_results(
    intent_groups,
    "data/processed/clustered_output.json"
)
```

**Clustering Process:**
1. **Initial Clustering**: First batch creates initial clusters
2. **Iterative Assignment**: Subsequent batches assign to existing clusters or create new ones
3. **Detailed Output**: Includes all questions with metadata

### 5. Evaluate Clustering Quality

Comprehensive evaluation with metrics and visualizations:

```bash
python pipeline/evaluate_clusters.py \
    data/processed/clustered_output.json \
    --save results/evaluation.json
```

**Generates:**
- **Purity scores**: Percentage of dominant category per cluster
- **Entropy measures**: Disorder in category distribution
- **Distribution analysis**: Category breakdown per cluster
- **Visual plots**: 6 comprehensive visualizations

**Plots Created:**
1. `purity_distribution.png` - Histogram and box plot
2. `top_20_by_purity.png` - Best clusters by purity
3. `top_20_by_size.png` - Largest clusters
4. `purity_vs_size.png` - Scatter plot analysis
5. `category_distribution_top_clusters.png` - Detailed breakdowns
6. `evaluation_dashboard.png` - Summary dashboard

## 📈 Evaluation Metrics

### Cluster Purity

**What it measures**: Percentage of dominant category in each cluster

**Formula**: `max(category_count) / total_questions_in_cluster × 100`

**Interpretation**:
- **High (≥80%)**: Excellent clustering
- **Moderate (60-80%)**: Good clustering
- **Low (<60%)**: Mixed clusters, needs improvement

### Category Distribution

Shows how questions from different categories are distributed across clusters.

### Entropy

Measures disorder in category distribution within clusters. Lower is better.

## 🎓 Methodology

### Two-Stage Approach

#### Stage 1: Intent Extraction
- **Input**: Raw customer support questions
- **Process**: LLM analyzes each question to identify primary intent
- **Output**: Concise intent labels (2-4 words)
- **Example**: "How do I cancel my order?" → "cancel order"

#### Stage 2: Intent Clustering
- **Input**: List of extracted intents with frequency
- **Process**: LLM groups similar intents into semantic clusters
- **Output**: Hierarchical clusters with category labels
- **Example**: ["cancel order", "modify order", "track order"] → "order management"

### Prompt Engineering

The system uses a template-based prompt system:

1. **Intent Extraction Prompts**: Focus on identifying user's goal
2. **Initial Clustering Prompts**: Create first set of semantic clusters
3. **Batch Clustering Prompts**: Assign new intents to existing clusters

All prompts are stored as separate files in `prompts/templates/` for easy modification and version control.

