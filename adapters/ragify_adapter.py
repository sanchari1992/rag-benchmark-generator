from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

import faiss
import numpy as np
import pandas as pd
from typing import Dict, Any

from sentence_transformers import SentenceTransformer

from adapters.base_adapter import BaseAdapter

class RAGifyAdapter(BaseAdapter):

    def __init__(self, repo_path):

        self.repo_path = Path(repo_path)

        self.pdf_folder = (
            self.repo_path /
            "input_files"
        )

        self.embedding_model = (
            SentenceTransformer(
                "all-MiniLM-L6-v2"
            )
        )

        self.chunks = (
            self._load_and_chunk_pdfs()
        )

        self.index = (
            self._build_faiss_index()
        )

    def extract_text(self, pdf_path):

        reader = PdfReader(pdf_path)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text

    def _load_and_chunk_pdfs(self):

        splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
        )

        chunks = []

        chunk_id = 0

        for pdf in self.pdf_folder.glob("*.pdf"):

            text = self.extract_text(pdf)

            doc_chunks = splitter.split_text(text)

            for chunk in doc_chunks:

                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "source_document": pdf.name
                })

                chunk_id += 1

        return chunks

    def _build_faiss_index(self):

        texts = [
            c["text"]
            for c in self.chunks
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
            embeddings.astype("float32")
        )

        return index

    def inspect(self):

        total_chars = sum(
            len(c["text"])
            for c in self.chunks
        )

        return {

            "name": "RAGify",

            "retrieval_type":
                "FAISS Dense Retrieval",

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
    
    def save_chunks(self, chunks_path):
        import json
        from pathlib import Path

        chunks_path = Path(chunks_path)
        chunks_path.parent.mkdir(parents=True, exist_ok=True)

        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)

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

    def save_config(self, config_path=None):
        config = {
            "adapter": "RAGify",
            "repo_path": str(self.repo_path),
            "pdf_folder": str(self.pdf_folder),
            "embedding_model": "all-mpnet-base-v2",
            "num_chunks": len(self.chunks)
        }

        if config_path:
            import json
            config_path = Path(config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

        return config

    def build_config(self) -> Dict[str, Any]:
        return {
            "name": "RAGify",
            "retrieval_type": "FAISS Dense Retrieval",
            "embedding_model": "all-mpnet-base-v2",
            "num_chunks": len(self.chunks),
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "pdf_folder": str(self.pdf_folder)
        }