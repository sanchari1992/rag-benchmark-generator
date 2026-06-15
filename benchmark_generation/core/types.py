from dataclasses import dataclass


@dataclass
class BenchmarkExample:
    chunk_id: int
    question: str
    answer: str
    source_text: str