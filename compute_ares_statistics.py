import json
import os


def compute_stats(data):

    total = len(data)

    if total == 0:
        return {}

    stats = {
        "total_samples": total,
        "context_relevance": 0,
        "answer_relevance": 0,
        "faithfulness": 0
    }

    for ex in data:

        labels = ex.get("positive_labels", {})

        stats["context_relevance"] += int(labels.get("context_relevance", False))
        stats["answer_relevance"] += int(labels.get("answer_relevance", False))
        stats["faithfulness"] += int(labels.get("faithfulness", False))

    # convert to percentages
    for k in ["context_relevance", "answer_relevance", "faithfulness"]:
        stats[k] = {
            "count": stats[k],
            "percentage": round(stats[k] / total * 100, 4)
        }

    return stats


def save_stats(stats, output_path):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":

    input_path = "exports/ares/wizard/run_20260623_210133/labeled_run2.json"
    output_path = "exports/ares/wizard/run_20260623_210133/metrics2.json"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = compute_stats(data)

    save_stats(stats, output_path)

    print("\n✔ ARES metrics saved to:", output_path)
    print(json.dumps(stats, indent=2))