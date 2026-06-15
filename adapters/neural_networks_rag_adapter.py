from pathlib import Path
import pandas as pd
import json
import yaml
import sys


class NeuralNetworksRAGAdapter:

    def __init__(self, repo_path):

        self.repo_path = Path(repo_path)

        self.csv_path = (
            self.repo_path /
            "data" /
            "chunked_data_corpus.csv"
        )

        self.dataset = pd.read_csv(
            self.csv_path
        )

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

        avg_chunk_length = int(
            total_characters / num_chunks
        )

        return {
            "num_chunks": num_chunks,
            "total_characters": total_characters,
            "avg_chunk_length": avg_chunk_length
        }

    # ----------------------------------
    # Export corpus
    # ----------------------------------

    def export_chunks(self):

        chunks = []

        for idx, row in self.dataset.iterrows():

            chunks.append({
                "chunk_id": int(idx),
                "text": str(row["raw_text"]),
                "chapter_name":
                    str(row["chapter_name"])
            })

        return chunks

    def save_chunks(self, output_file):

        chunks = self.export_chunks()

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                chunks,
                f,
                indent=2,
                ensure_ascii=False
            )

    # ----------------------------------
    # Config generation
    # ----------------------------------

    def build_config(self):

        stats = self.corpus_stats()

        config = {

            "name":
                "Neural_Networks_RAG",

            "repo_path":
                str(self.repo_path),

            "retrieval_type":
                "BM25 + Dense + CrossEncoder",

            "dense_model":
                "all-MiniLM-L6-v2",

            "cross_encoder":
                "cross-encoder/ms-marco-MiniLM-L-12-v2",

            "llm":
                "groq/llama3-8b-8192",

            "corpus_file":
                str(self.csv_path),

            "num_chunks":
                stats["num_chunks"],

            "total_characters":
                stats["total_characters"],

            "avg_chunk_length":
                stats["avg_chunk_length"],

            "benchmark_size_5pct":
                int(stats["num_chunks"] * 0.05)
        }

        return config

    def save_config(self, output_yaml):

        config = self.build_config()

        with open(
            output_yaml,
            "w"
        ) as f:

            yaml.dump(
                config,
                f,
                sort_keys=False
            )

        return config

    # ----------------------------------
    # Retrieval
    # ----------------------------------

    def initialize_retriever(self):

        sys.path.append(
            str(self.repo_path)
        )

        from chatbot.retriever import (
            HybridRetrieverReranker
        )

        return HybridRetrieverReranker(
            self.dataset
        )

    def retrieve(
            self,
            query,
            k=2):

        retriever = (
            self.initialize_retriever()
        )

        docs = retriever.hybrid_retrieve(
            query,
            top_n_rerank=k
        )

        return docs