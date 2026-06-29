"""
metric_runner.py

Usage:
python metric_runner.py \
    --input input.json \
    --output outputs/output.json
"""

import argparse
import json
import os
import time
import numbers

from benchmark_generation.metrics import lexical_metrics
from benchmark_generation.metrics import readability_metrics
from benchmark_generation.metrics import syntax_metrics
from benchmark_generation.metrics import graph_metrics


############################################################
# Metric Modules
############################################################

METRIC_MODULES = [
    lexical_metrics,
    readability_metrics,
    syntax_metrics,
    graph_metrics,
]


############################################################
# JSON Helpers
############################################################

def load_json(filename):

    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, filename):

    directory = os.path.dirname(filename)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


############################################################
# Metric Runner
############################################################

def run_metrics(records):

    for module in METRIC_MODULES:

        print("=" * 60)
        print(f"Running {module.__name__}")

        start = time.time()

        records = module.compute(records)

        end = time.time()

        print(f"Finished in {end-start:.2f} sec")

    return records


############################################################
# Summary Statistics
############################################################

def compute_summary(records):
    """
    Computes the average of every numeric field.

    Non-numeric fields are ignored.
    """

    summary = {}

    if len(records) == 0:
        return summary

    numeric_fields = {}

    for record in records:

        for key, value in record.items():

            if isinstance(value, numbers.Number):

                numeric_fields.setdefault(key, []).append(value)

    for key, values in numeric_fields.items():

        summary[key] = {
            "average": round(sum(values) / len(values), 4),
            "minimum": round(min(values), 4),
            "maximum": round(max(values), 4)
        }

    summary["total_records"] = len(records)

    return summary


############################################################
# Main
############################################################

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input benchmark JSON"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON with metrics"
    )

    args = parser.parse_args()

    print(f"\nLoading {args.input}")

    records = load_json(args.input)

    print(f"{len(records)} records loaded.\n")

    records = run_metrics(records)

    save_json(records, args.output)

    ########################################################
    # Save summary statistics
    ########################################################

    summary = compute_summary(records)

    output_dir = os.path.dirname(args.output)

    summary_file = os.path.join(
        output_dir,
        "summary_metrics.json"
    )

    save_json(summary, summary_file)

    print("\nDone.")
    print(f"Metrics written to : {args.output}")
    print(f"Summary written to : {summary_file}")


if __name__ == "__main__":
    main()