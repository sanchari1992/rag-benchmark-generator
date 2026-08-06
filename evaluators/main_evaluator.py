import argparse
import json
import os

from tqdm import tqdm

from llm_judge import DeepSeekJudge
from metric_registry import MetricRegistry


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_data(benchmark, results):
    result_map = {r["id"]: r for r in results}

    merged = []

    for b in benchmark:

        r = result_map.get(b["id"])

        if r is None:
            continue

        merged.append({

            "id": b["id"],

            "repo_name": b.get("repo_name"),
            "strategy": b.get("strategy"),
            "generator_model": r.get("generator_model"),

            "question": b["question"],

            # Ground truth
            "reference_answer": b["answer"],

            # RAG output
            "generated_answer": r["response"],

            # Retrieval information
            "retrieved_context": b["source_text"],

            # Optional metadata
            "source_chunk": b.get("source_chunk"),
            "context_length": b.get("context_length"),
            "compression_stage": b.get("compression_stage")
        })

    return merged


def evaluate(
    benchmark_path,
    results_path,
    metric_names,
    output_path
):
    benchmark = load_json(benchmark_path)
    results = load_json(results_path)

    merged = merge_data(benchmark, results)

    judge = DeepSeekJudge()

    registry = MetricRegistry(judge)

    evaluations = []

    for item in tqdm(
        merged,
        desc="Evaluating",
        unit="question"
    ): 

        row = dict(item)

        for metric in metric_names:

            metric_fn = registry.get(metric)

            score = metric_fn(item)

            row["evaluations"] = {}

            for metric in metric_names:

                metric_fn = registry.get(metric)

                row["evaluations"][metric] = metric_fn(item)

        evaluations.append(row)
        summary = {}

        for metric in metric_names:

            scores = []

            for row in evaluations:

                value = row["evaluations"][metric]

                if isinstance(value, dict):

                    s = value.get("score")

                    if s is not None:
                        scores.append(s)

            if scores:

                summary[metric] = {
                    "mean": round(sum(scores)/len(scores), 3),
                    "min": round(min(scores), 3),
                    "max": round(max(scores), 3),
                    "count": len(scores)
                }

    with open(output_path, "w", encoding="utf-8") as f:
        output = {

            "summary": summary,

            "results": evaluations

        }

        json.dump(output, f, indent=2)

    print(f"\nSaved {len(evaluations)} evaluations")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("benchmark")

    parser.add_argument("results")

    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "ragas_faithfulness",
            "ragas_answer_relevancy",
            "ragas_context_relevance"
        ]
    )                   

    parser.add_argument(
        "--output",
        default="evaluation.json"
    )

    args = parser.parse_args()

    evaluate(
        args.benchmark,
        args.results,
        args.metrics,
        args.output
    )