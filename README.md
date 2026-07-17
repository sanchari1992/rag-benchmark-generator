# RAG Benchmark Generation Framework

A modular framework for **generating and evaluating benchmarks for Retrieval-Augmented Generation (RAG) systems**.

This project is designed to support multiple RAG repositories through a unified adapter interface and enables automated, LLM-driven benchmark creation, execution, and evaluation.

---

## 🔍 Motivation

Evaluating RAG systems is often difficult because:

- Benchmarks are tied to specific datasets or systems
- Existing evaluation sets are large and expensive to run
- There is limited support for *custom corpora-specific evaluation*

This framework addresses these issues by:

- Generating **RAG-specific synthetic QA benchmarks**
- Supporting **pluggable RAG systems via adapters**
- Enabling **standardized evaluation across systems**
- Reducing evaluation cost through controlled benchmark generation

---

## ⚙️ Key Features

- 🔌 **Adapter-based architecture** for multiple RAG systems
- 📊 **Corpus inspection and statistics extraction**
- 🔎 **Standardized retrieval + generation interface**
- 🤖 **LLM-based synthetic QA generation**
- 🧠 Strategy-driven benchmark construction
- 📦 Support for multiple repositories in a single evaluation pipeline
- 📈 Automated evaluation and metric reporting

---

## 🧱 Supported RAG Systems

Currently supported:

- [RAGify](https://github.com/kanad13/RAGify)
- [Neural Networks RAG](https://github.com/rDrayBen/Neural_networks_RAG)

---

## 🏗️ Architecture Overview

The framework follows a modular pipeline:

1. **Adapter Layer**
   - Wraps each RAG repository
   - Standardizes retrieval + corpus access

2. **Benchmark Generator**
   - Samples corpus content
   - Uses LLMs to generate QA pairs
   - Applies strategy-based prompting

3. **Evaluation Runner**
   - Runs each benchmark question through the selected RAG
   - Collects generated answers

4. **Evaluator**
   - Compares predictions vs ground truth
   - Computes accuracy metrics and reports

---

## 🚀 Sample Commands

### 1. Generate Answers using a RAG system

```bash
python generate_answers.py \
.\exports\ragas\ragify\run_20260617_142317\benchmark.json \
--rag ragify \
--output .\evaluation_results\ragas_ragify_results.json
```

### 2. Run Evaluation Metrics

```bash
python .\evaluators\evaluator.py \
.\exports\ragas\ragify\run_20260617_142317\benchmark.json \
.\evaluation_results\ragas_ragify_results.json \
--output .\evaluation_results\ragas_ragify_evalmetrics.json
```

## Adding a New RAG System

To integrate a new RAG repository:

### Step 1: Create an Adapter

Add a new adapter in:

adapters/

Implement the base interface defined in:

BaseAdapter

### Step 2: Register in Factory

Update:

factory.py

to include your new adapter.

### Step 3: Enable Benchmark Execution

Ensure your adapter is compatible with:

run_benchmark.py
generate_answers.py
evaluator.py

## Output Structure

Typical outputs include:

Generated benchmark dataset (benchmark.json)
RAG-generated answers (results.json)
Evaluation metrics (evalmetrics.json)

## Future Work

Planned extensions include:

Benchmark compression strategies
reasoning_graph_compressor.py

python reasoning_graph_compressor.py \
    --input abc.json \
    --output xyz.json \
    --clusters 20

Multi-strategy QA generation (beyond RAGAS-style prompting)
RAG robustness evaluation (hallucination + faithfulness metrics)
Cross-corpus generalization testing
Cost-aware benchmark optimization

## Notes

This framework assumes that:

Each RAG system can expose a retrieval + generation interface
Corpus access is available via the adapter layer
LLM API (e.g., DeepSeek / OpenAI-compatible) is configured externally

## License
