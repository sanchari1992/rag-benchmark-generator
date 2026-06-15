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