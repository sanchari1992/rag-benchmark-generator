from pathlib import Path
from typing import Dict, Any
import json

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from adapters.base_adapter import BaseAdapter


class WikiRagasAdapter(BaseAdapter):

    def __init__(self, repo_path):

        self.repo_path = Path(repo_path)

        self.docs_path = self.repo_path / "documents"

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.chunks = self._load_and_chunk_documents()

        self.index = self._build_faiss_index()

        self._doc_map = self._build_doc_map()


    # ----------------------------------
    # LOAD + CHUNK
    # ----------------------------------

    def _load_and_chunk_documents(self):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = []
        chunk_id = 0

        for file_path in self.docs_path.glob("*.txt"):

            text = file_path.read_text(encoding="utf-8")

            doc_chunks = splitter.split_text(text)

            for chunk in doc_chunks:

                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "source_document": file_path.name
                })

                chunk_id += 1

        return chunks


    # ----------------------------------
    # FAISS INDEX
    # ----------------------------------

    def _build_faiss_index(self):

        texts = [c["text"] for c in self.chunks]

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)

        index.add(embeddings.astype("float32"))

        return index


    # ----------------------------------
    # MAP (optional metadata helper)
    # ----------------------------------

    def _build_doc_map(self):

        return {
            i: self.chunks[i]
            for i in range(len(self.chunks))
        }


    # ----------------------------------
    # REQUIRED: inspect()
    # ----------------------------------

    def inspect(self):

        total_chars = sum(len(c["text"]) for c in self.chunks)

        return {
            "name": "WikiRagas",
            "repo_path": str(self.repo_path),
            "retrieval_type": "FAISS Dense Retrieval",
            "embedding_model": "all-mpnet-base-v2",
            "num_documents": len(list(self.docs_path.glob("*.txt"))),
            "num_chunks": len(self.chunks),
            "total_characters": total_chars,
            "avg_chunk_length": (
                total_chars // len(self.chunks)
                if self.chunks else 0
            ),
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "corpus_folder": str(self.docs_path)
        }


    # ----------------------------------
    # REQUIRED: export_chunks()
    # ----------------------------------

    def export_chunks(self):

        return self.chunks


    # ----------------------------------
    # SAVE CHUNKS
    # ----------------------------------

    def save_chunks(self, output_file):

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)

        return str(output_file)


    # ----------------------------------
    # RETRIEVAL
    # ----------------------------------

    def retrieve(self, query, k=5):

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )

        D, I = self.index.search(
            query_embedding.astype("float32"),
            k
        )

        results = []

        for idx in I[0]:
            if idx < len(self.chunks):
                results.append(self.chunks[idx])

        return results


    # ----------------------------------
    # CONFIG
    # ----------------------------------

    def build_config(self):

        return self.inspect()


    def save_config(self, output_path):

        config = self.build_config()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        return config