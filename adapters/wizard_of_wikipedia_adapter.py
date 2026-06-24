from pathlib import Path
from typing import Dict, Any, List
import json

import faiss
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from adapters.base_adapter import BaseAdapter


class WizardOfWikipediaAdapter(BaseAdapter):
    """
    Adapter for the Wizard of Wikipedia dataset.

    The corpus consists of the Wikipedia knowledge passages supplied
    during each dialogue turn.
    """

    def __init__(self, repo_path):

        self.repo_path = Path(repo_path)

        self.dataset_path = (
            self.repo_path
            / "datasets"
            / "wizard_of_wikipedia_train"
        )

        self.embedding_model_name = "all-mpnet-base-v2"

        self.embedding_model = SentenceTransformer(
            self.embedding_model_name
        )

        self.dataset = load_from_disk(str(self.dataset_path))

        self.chunks = self._load_and_chunk_documents()

        self.index = self._build_faiss_index()

    # --------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------

    def _load_and_chunk_documents(self):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = []
        seen = set()
        chunk_id = 0

        for example_id, example in enumerate(self.dataset):

            topic = (
                example["topics"][0]
                if len(example["topics"]) > 0
                else None
            )

            for turn_id, knowledge_list in enumerate(example["knowledge"]):

                for passage in knowledge_list:

                    if "__knowledge__" in passage:
                        _, passage = passage.split(
                            "__knowledge__",
                            1
                        )

                    passage = passage.strip()

                    if (
                        not passage
                        or passage == "no_passages_used"
                    ):
                        continue

                    # remove duplicate passages
                    if passage in seen:
                        continue

                    seen.add(passage)

                    doc_chunks = splitter.split_text(passage)

                    for chunk in doc_chunks:

                        chunks.append({
                            "chunk_id": chunk_id,
                            "text": chunk,
                            "metadata": {
                                "example_id": example_id,
                                "turn_id": turn_id,
                                "topic": topic,
                                "source_dataset": "wizard_of_wikipedia"
                            }
                        })

                        chunk_id += 1

        return chunks

    # --------------------------------------------------
    # BUILD FAISS INDEX
    # --------------------------------------------------

    def _build_faiss_index(self):

        texts = [
            c["text"]
            for c in self.chunks
        ]

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)

        index.add(
            embeddings.astype("float32")
        )

        return index

    # --------------------------------------------------
    # REQUIRED
    # --------------------------------------------------

    def inspect(self):

        total_chars = sum(
            len(c["text"])
            for c in self.chunks
        )

        return {

            "name": "WizardOfWikipedia",

            "repo_path": str(self.repo_path),

            "dataset_path": str(self.dataset_path),

            "retrieval_type": "Dense FAISS",

            "embedding_model": self.embedding_model_name,

            "num_examples": len(self.dataset),

            "num_chunks": len(self.chunks),

            "total_characters": total_chars,

            "avg_chunk_length": (
                total_chars // len(self.chunks)
                if self.chunks else 0
            ),

            "chunk_size": 1000,

            "chunk_overlap": 200

        }

    # --------------------------------------------------
    # EXPORT CHUNKS
    # --------------------------------------------------

    def export_chunks(self) -> List[Dict[str, Any]]:

        return self.chunks

    # --------------------------------------------------
    # SAVE CHUNKS
    # --------------------------------------------------

    def save_chunks(self, output_file):

        output_file = Path(output_file)

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.chunks,
                f,
                indent=2,
                ensure_ascii=False
            )

        return str(output_file)

    # --------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------

    def retrieve(
        self,
        query,
        k=5
    ):

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )

        distances, indices = self.index.search(
            query_embedding.astype("float32"),
            k
        )

        results = []

        for score, idx in zip(
            distances[0],
            indices[0]
        ):

            if idx >= len(self.chunks):
                continue

            result = dict(self.chunks[idx])

            result["score"] = float(score)

            results.append(result)

        return results

    # --------------------------------------------------
    # CONFIG
    # --------------------------------------------------

    def build_config(self):

        return self.inspect()

    # --------------------------------------------------
    # SAVE CONFIG
    # --------------------------------------------------

    def save_config(self, output_path):

        config = self.build_config()

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                config,
                f,
                indent=2,
                ensure_ascii=False
            )

        return config