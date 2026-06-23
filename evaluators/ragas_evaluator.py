import json
import os

from tqdm import tqdm

from ragas_faithfulness_judge import FaithfulnessJudge
from ragas_metrics import compute_ragas_metrics


# =========================================================
# LOAD JSON
# =========================================================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# MERGE BENCHMARK + RESULTS
# =========================================================
def merge_data(benchmark, results):
    """
    Merge benchmark and generated answers using the sample id.
    """
    result_map = {r["id"]: r for r in results}

    merged = []

    for b in benchmark:
        r = result_map.get(b["id"])

        if r is None:
            continue

        merged.append({
            **b,
            **r
        })

    return merged


# =========================================================
# EVALUATION
# =========================================================
def evaluate(
    benchmark_path,
    results_path,
    output_dir="outputs"
):
    os.makedirs(output_dir, exist_ok=True)

    benchmark = load_json(benchmark_path)
    results = load_json(results_path)

    merged = merge_data(benchmark, results)

    judge = FaithfulnessJudge()

    detailed_results = []

    print("\nEvaluating Faithfulness...\n")

    for item in tqdm(merged, desc="Faithfulness"):

        # ----------------------------------------
        # Get retrieved context
        # ----------------------------------------
        retrieved_context = item.get("retrieved_context", [])

        if isinstance(retrieved_context, list):
            context = "\n\n".join(retrieved_context)
        else:
            context = str(retrieved_context)

        # ----------------------------------------
        # Generated answer
        # ----------------------------------------
        answer = item.get("response", "")

        # ----------------------------------------
        # Evaluate
        # ----------------------------------------
        judgment = judge.judge(
            question=item["question"],
            context=context,
            answer=answer
        )

        detailed_results.append({
            "id": item["id"],
            "repo_name": item.get("repo_name"),
            "strategy": item.get("strategy"),
            "generator_model": item.get("generator_model"),

            "question": item["question"],
            "answer": answer,

            "retrieved_context": retrieved_context,

            "faithfulness": judgment.get(
                "faithfulness",
                0.0
            ),

            "statements": judgment.get(
                "statements",
                []
            )
        })

    # ----------------------------------------
    # Compute overall metrics
    # ----------------------------------------
    metrics = compute_ragas_metrics(detailed_results)

    # ----------------------------------------
    # Save detailed results
    # ----------------------------------------
    detailed_path = os.path.join(
        output_dir,
        "faithfulness_results.json"
    )

    with open(
        detailed_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            detailed_results,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ----------------------------------------
    # Save report
    # ----------------------------------------
    report_path = os.path.join(
        output_dir,
        "faithfulness_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\nDONE\n")
    print(json.dumps(metrics["summary"], indent=2))


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate RAG Faithfulness (RAGAS)"
    )

    parser.add_argument(
        "benchmark_path",
        help="Benchmark JSON"
    )

    parser.add_argument(
        "results_path",
        help="Generated answers JSON"
    )

    parser.add_argument(
        "--output",
        default="outputs",
        help="Output directory"
    )

    args = parser.parse_args()

    evaluate(
        args.benchmark_path,
        args.results_path,
        args.output
    )