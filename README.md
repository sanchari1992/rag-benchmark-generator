# RAG Benchmark Generation Framework

This project provides a **modular framework for generating evaluation benchmarks for Retrieval-Augmented Generation (RAG) systems**.

It supports multiple RAG repositories via a unified adapter interface and enables automated dataset generation using LLM-based strategies.

---

## 🔧 Features

- Plug-and-play RAG adapters
- Corpus statistics extraction
- Retrieval interface standardization
- LLM-based synthetic QA generation
- Strategy-based benchmark generation (RAGAS baseline included)
- Multi-repository evaluation support

---

## 📁 Architecture
adapters/ # RAG system wrappers
benchmark_generation/ # dataset generation framework
repositories/ # external RAG systems
exports/ # generated datasets
configs/ # metadata configs


---

## 📁 RAGS used

- [RAGIFY](https://github.com/kanad13/RAGify)
- [NEURAL](https://github.com/rDrayBen/Neural_networks_RAG)

## Sample Commands

### Evaluation Runs
python generate_answers.py .\exports\ragas\ragify\run_20260617_142317\benchmark.json --rag ragify --output .\evaluation_results\ragas_ragify_results.json  


python .\evaluators\evaluator.py .\exports\ragas\ragify\run_20260617_142317\benchmark.json .\evaluation_results\ragas_ragify_results.json --output .\evaluation_results\ragas_ragify_evalmetrics