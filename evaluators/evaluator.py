import json
import os
from tqdm import tqdm

from llm_judge import DeepSeekJudge
from metrics import compute_metrics


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_data(benchmark, results):
    """
    Join benchmark + results on id or question
    """
    result_map = {r["id"]: r for r in results}

    merged = []
    for b in benchmark:
        r = result_map.get(b["id"], None)
        if not r:
            continue

        merged.append({
            **b,
            **r
        })

    return merged


def evaluate(benchmark_path, results_path, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    benchmark = load_json(benchmark_path)
    results = load_json(results_path)

    merged = merge_data(benchmark, results)

    judge = DeepSeekJudge()

    detailed_results = []

    for item in tqdm(merged, desc="Evaluating"):
        question = item["question"]
        expected = item["answer"]
        predicted = item["response"]

        judgment = judge.judge(question, expected, predicted)

        detailed_results.append({
            "id": item["id"],
            "repo_name": item.get("repo_name"),
            "strategy": item.get("strategy"),
            "generator_model": item.get("generator_model"),
            "question": question,
            "expected": expected,
            "predicted": predicted,
            "verdict": judgment.get("verdict"),
            "score": judgment.get("score", 0),
            "reason": judgment.get("reason", "")
        })

    metrics = compute_metrics(detailed_results)

    # save outputs
    with open(os.path.join(output_dir, "detailed_results.json"), "w") as f:
        json.dump(detailed_results, f, indent=2)

    with open(os.path.join(output_dir, "evaluation_report.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nDONE")
    print(json.dumps(metrics["summary"], indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("benchmark_path")
    parser.add_argument("results_path")
    parser.add_argument("--output", default="outputs")

    args = parser.parse_args()

    evaluate(args.benchmark_path, args.results_path, args.output)