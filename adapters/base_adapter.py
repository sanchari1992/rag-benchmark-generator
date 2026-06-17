from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseAdapter(ABC):
    """
    Abstract interface for all RAG adapters in the benchmark framework.

    Every RAG repo must be wrapped with this interface so that:
    - benchmarking is repo-agnostic
    - dataset generation is uniform
    - retrieval testing is standardized
    """

    # --------------------------------------------------
    # REQUIRED: Load + Inspect
    # --------------------------------------------------

    @abstractmethod
    def inspect(self) -> Dict[str, Any]:
        """
        Returns metadata about the RAG system.

        Expected fields (recommended):
        - num_chunks
        - total_characters
        - avg_chunk_length
        - embedding_model (if available)
        - retrieval_type
        """
        pass

    # --------------------------------------------------
    # REQUIRED: Export corpus in unified format
    # --------------------------------------------------

    @abstractmethod
    def export_chunks(self) -> List[Dict[str, Any]]:
        """
        Must return unified chunk format:

        [
            {
                "chunk_id": int or str,
                "text": str,
                "metadata": dict (optional)
            }
        ]
        """
        pass

    # --------------------------------------------------
    # REQUIRED: Retrieval interface
    # --------------------------------------------------

    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Standard retrieval function across all RAG systems.

        Must return:
        [
            {
                "chunk_id": ...,
                "text": ...,
                "score": optional float
            }
        ]
        """
        pass

    # --------------------------------------------------
    # OPTIONAL: Direct QA (if RAG supports it)
    # --------------------------------------------------

    def query(self, question: str) -> Optional[str]:
        """
        Optional end-to-end QA call (retrieval + LLM).

        Not required for benchmark generation,
        but useful for evaluation later.
        """
        return None

    # --------------------------------------------------
    # OPTIONAL: Corpus stats shortcut
    # --------------------------------------------------

    def corpus_stats(self) -> Dict[str, Any]:
        """
        Optional helper.
        Can be overridden if system has native stats.
        """
        chunks = self.export_chunks()

        total_chars = sum(
            len(c.get("text", ""))
            for c in chunks
        )

        return {
            "num_chunks": len(chunks),
            "total_characters": total_chars,
            "avg_chunk_length": (
                total_chars // len(chunks)
                if chunks else 0
            )
        }

    # --------------------------------------------------
    # OPTIONAL: Benchmark size helper
    # --------------------------------------------------

    def benchmark_size(self, percent: float = 0.05) -> int:
        """
        Default heuristic:
        benchmark size = % of corpus
        """
        stats = self.corpus_stats()

        return max(
            10,
            int(stats["num_chunks"] * percent)
        )

    @abstractmethod
    def build_config(self) -> Dict[str, Any]:
        """
        Must return reproducibility metadata for benchmarking.

        Required fields:
        - name
        - retrieval_type
        - embedding_model (if applicable)
        - num_chunks (optional but recommended)
        """
        pass