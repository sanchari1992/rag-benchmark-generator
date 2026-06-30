"""
rank_by_metric.py

Utility for extracting the highest and lowest scoring records
for a chosen metric.

Example
-------
python rank_by_metric.py benchmark.json question_word_count 20
"""

import json
import os
import sys


def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_records(records, path):

    with open(path, "w", encoding="utf-8") as f:

        json.dump(records, f, indent=2, ensure_ascii=False)


def rank_records(records, metric, n):

    valid = [r for r in records if metric in r]

    if not valid:
        raise ValueError(f"Metric '{metric}' not found.")

    highest = sorted(
        valid,
        key=lambda r: r[metric],
        reverse=True
    )[:n]

    lowest = sorted(
        valid,
        key=lambda r: r[metric]
    )[:n]

    return highest, lowest


def main():

    if len(sys.argv) != 4:

        print(
            "Usage:\n"
            "python rank_by_metric.py input.json metric_name n"
        )
        return

    input_file = sys.argv[1]
    metric = sys.argv[2]
    n = int(sys.argv[3])

    records = load_records(input_file)

    highest, lowest = rank_records(records, metric, n)

    output_dir = os.path.join("rank_results", metric)
    os.makedirs(output_dir, exist_ok=True)

    save_records(
        highest,
        os.path.join(output_dir, f"highest_{n}.json")
    )

    save_records(
        lowest,
        os.path.join(output_dir, f"lowest_{n}.json")
    )

    print(f"Metric: {metric}")
    print(f"Highest saved to {output_dir}/highest_{n}.json")
    print(f"Lowest saved to {output_dir}/lowest_{n}.json")


if __name__ == "__main__":
    main()