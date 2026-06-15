import json
from tqdm import tqdm

from core.sampler import sample_chunks
from core.types import BenchmarkExample


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

    dataset = []

    for i, chunk in tqdm(enumerate(sampled), total=len(sampled)):

        try:
            question = strategy.generate_question(
                llm, model, chunk
            )

            answer = strategy.generate_answer(
                llm, model, question, chunk
            )

            example = BenchmarkExample(
                chunk_id=chunk["chunk_id"],
                question=question,
                answer=answer,
                source_text=chunk["text"]
            )

            dataset.append(example.__dict__)

        except Exception as e:
            print(f"Error at {i}: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    return dataset