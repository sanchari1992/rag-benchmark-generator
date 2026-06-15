from pathlib import Path
import pandas as pd
import json
import yaml
import sys

from adapters.base_adapter import BaseAdapter


class NeuralNetworksRAGAdapter(BaseAdapter):

    def __init__(self, repo_path):

        self.repo_path = Path(repo_path)

        self.csv_path = (
            self.repo_path /
            "data" /
            "chunked_data_corpus.csv"
        )

        self.dataset = pd.read_csv(self.csv_path)

        # lazy-loaded retriever cache (IMPORTANT)
        self._retriever = None

    # ----------------------------------
    # REQUIRED: inspect()
    # ----------------------------------

    def inspect(self):

        stats = self.corpus_stats()

        return {
            "name": "Neural_Networks_RAG",
            "repo_path": str(self.repo_path),
            "retrieval_type": "BM25 + Dense + CrossEncoder",
            "dense_model": "all-MiniLM-L6-v2",
            "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "llm": "groq/llama3-8b-8192",
            "corpus_file": str(self.csv_path),
            **stats,
            "benchmark_size_5pct": max(
                10,
                int(stats["num_chunks"] * 0.05)
            )
        }

    # ----------------------------------
    # Corpus statistics
    # ----------------------------------

    def corpus_stats(self):

        num_chunks = len(self.dataset)

        total_characters = int(
            self.dataset["raw_text"]
            .astype(str)
            .str.len()
            .sum()
        )

        avg_chunk_length = (
            total_characters // num_chunks
            if num_chunks else 0
        )

        return {
            "num_chunks": num_chunks,
            "total_characters": total_characters,
            "avg_chunk_length": avg_chunk_length
        }

    # ----------------------------------
    # REQUIRED: export_chunks()
    # ----------------------------------

    def export_chunks(self):

        return [
            {
                "chunk_id": int(idx),
                "text": str(row["raw_text"]),
                "chapter_name": str(row["chapter_name"])
            }
            for idx, row in self.dataset.iterrows()
        ]

    # ----------------------------------
    # Save chunks
    # ----------------------------------

    def save_chunks(self, output_file):

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                self.export_chunks(),
                f,
                indent=2,
                ensure_ascii=False
            )

    # ----------------------------------
    # Config generation (wrapper over inspect)
    # ----------------------------------

    def build_config(self):
        return self.inspect()

    def save_config(self, output_yaml):

        config = self.build_config()

        output_path = Path(output_yaml)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, sort_keys=False)

        return config

    # ----------------------------------
    # Lazy retriever initialization
    # ----------------------------------

    def initialize_retriever(self):

        if self._retriever is not None:
            return self._retriever

        sys.path.append(str(self.repo_path))

        from chatbot.retriever import HybridRetrieverReranker

        self._retriever = HybridRetrieverReranker(self.dataset)

        return self._retriever

    # ----------------------------------
    # REQUIRED: retrieve()
    # ----------------------------------

    def retrieve(self, query, k=2):

        retriever = self.initialize_retriever()

        docs = retriever.hybrid_retrieve(
            query,
            top_n_rerank=k
        )

        return docs