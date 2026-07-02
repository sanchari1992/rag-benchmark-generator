from pathlib import Path
import json
from typing import Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from adapters.base_adapter import BaseAdapter


class COVIDQAAdapter(BaseAdapter):

    def __init__(self, repo_path):

        self.repo_path = Path(repo_path)

        self.dataset_path = (
            self.repo_path /
            "data" /
            "question-answering" /
            "COVID-QA.json"
        )

        self.embedding_model = (
            SentenceTransformer(
                "all-MiniLM-L6-v2"
            )
        )

        self.chunks = (
            self._load_dataset()
        )

        self.index = (
            self._build_faiss_index()
        )

    def _load_dataset(self):

        with open(
            self.dataset_path,
            "r",
            encoding="utf-8"
        ) as f:

            dataset = json.load(f)

        chunks = []

        chunk_id = 0

        for article_id, article in enumerate(dataset["data"]):

            title = article.get(
                "title",
                f"Article {article_id}"
            )

            for paragraph in article["paragraphs"]:

                context = paragraph["context"].strip()

                chunks.append({

                    "chunk_id": chunk_id,

                    "text": context,

                    "source_document": title,

                    "chapter_name": title

                })

                chunk_id += 1

        return chunks

    def _build_faiss_index(self):

        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        embeddings = (
            self.embedding_model.encode(
                texts,
                convert_to_numpy=True
            )
        )

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(
            dimension
        )

        index.add(
            embeddings.astype(
                "float32"
            )
        )

        return index

    def inspect(self):

        total_chars = sum(
            len(c["text"])
            for c in self.chunks
        )

        return {

            "name": "COVID-QA",

            "retrieval_type":
                "Dense FAISS Retrieval",

            "embedding_model":
                "all-mpnet-base-v2",

            "num_chunks":
                len(self.chunks),

            "total_characters":
                total_chars,

            "avg_chunk_length":
                int(
                    total_chars /
                    len(self.chunks)
                )
        }

    def export_chunks(self):

        return self.chunks

    def save_chunks(
        self,
        chunks_path
    ):

        chunks_path = Path(chunks_path)

        chunks_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            chunks_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.chunks,
                f,
                indent=2,
                ensure_ascii=False
            )

        return str(chunks_path)

    def retrieve(
        self,
        query,
        k=5
    ):

        query_embedding = (
            self.embedding_model.encode(
                [query]
            )
        )

        D, I = self.index.search(

            query_embedding.astype(
                "float32"
            ),

            k
        )

        results = []

        for idx in I[0]:

            results.append(
                self.chunks[idx]
            )

        return results

    def save_config(
        self,
        config_path=None
    ):

        config = {

            "adapter":
                "COVID-QA",

            "repo_path":
                str(self.repo_path),

            "dataset":
                str(self.dataset_path),

            "embedding_model":
                "all-mpnet-base-v2",

            "num_chunks":
                len(self.chunks)
        }

        if config_path:

            config_path = Path(
                config_path
            )

            config_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with open(
                config_path,
                "w"
            ) as f:

                json.dump(
                    config,
                    f,
                    indent=2
                )

        return config

    def build_config(
        self
    ) -> Dict[str, Any]:

        return {

            "name":
                "COVID-QA",

            "retrieval_type":
                "Dense FAISS Retrieval",

            "embedding_model":
                "all-mpnet-base-v2",

            "num_chunks":
                len(self.chunks),

            "chunk_size":
                "paragraph",

            "chunk_overlap":
                0,

            "dataset":
                str(self.dataset_path)
        }