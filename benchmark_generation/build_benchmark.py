import json
from tqdm import tqdm

from benchmark_generation.core.sampler import sample_chunks
from benchmark_generation.core.types import BenchmarkExample


def build_benchmark(
    adapter,
    llm,
    model,
    strategy,
    sample_size=50,
    output_path="benchmark.json"
):

    chunks = adapter.export_chunks()
    sampled = sample_chunks(chunks, k=sample_size)
    repo_info = adapter.build_config()

    dataset = []

    for i, chunk in tqdm(enumerate(sampled), total=len(sampled)):

        try:

            question = strategy.generate_question(
                llm,
                model,
                chunk
            )

            answer = strategy.generate_answer(
                llm,
                model,
                question,
                chunk
            )

            example = BenchmarkExample(

                id=i,

                repo_name=repo_info["name"],

                strategy=strategy.name,

                generator_model=model,

                source_chunk=chunk["chunk_id"],

                context_length=len(chunk["text"]),

                chapter_name=chunk.get(
                    "chapter_name",
                    ""
                ),

                question=question,

                answer=answer,

                source_text=chunk["text"],

                compression_stage="none"
            )

            record = example.__dict__.copy()

            # Optional strategy-specific metadata
            if hasattr(strategy, "last_metadata"):
                record.update(strategy.last_metadata)

            dataset.append(record)

        except Exception as e:

            print(f"Error at {i}: {e}")

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dataset,
            f,
            indent=2,
            ensure_ascii=False
        )

    return dataset