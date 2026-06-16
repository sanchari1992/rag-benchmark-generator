from dataclasses import dataclass


@dataclass
class BenchmarkExample:

    # unique example id
    id: int

    # provenance
    repo_name: str
    strategy: str
    generator_model: str

    # source info
    source_chunk: int
    context_length: int
    chapter_name: str

    # QA pair
    question: str
    answer: str

    # original source text
    source_text: str
    compression_stage: str