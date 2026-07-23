import json
import os


def merge_json_files(input_files, output_file):
    """
    Merge multiple JSON files containing lists of benchmark items into one JSON file.

    Args:
        input_files (list): List of JSON file paths to merge.
        output_file (str): Path for the merged JSON output.
    """

    merged_data = []

    for file_path in input_files:
        print(f"Loading: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            if not isinstance(data, list):
                raise ValueError(
                    f"{file_path} does not contain a JSON list."
                )

            print(f"  Loaded {len(data)} items")
            merged_data.extend(data)

    print(f"\nTotal merged items: {len(merged_data)}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            merged_data,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"Saved merged dataset to: {output_file}")


if __name__ == "__main__":

    # Add your JSON files here
    input_files = [
        # "./exports/ragas/neural/run_20260721_080651/benchmark.json",
        # "./exports/ares/neural/run_20260721_085342/benchmark.json",
        # "./exports/datamorgana/neural/run_20260721_090911/benchmark.json",
        # "./exports/deepeval/neural/run_20260721_092307/benchmark.json"

        # "./exports/ragas/ragify/run_20260721_092751/benchmark.json",
        # "./exports/ares/ragify/run_20260721_092936/benchmark.json",
        # "./exports/datamorgana/ragify/run_20260721_093102/benchmark.json",
        # "./exports/deepeval/ragify/run_20260721_093255/benchmark.json"

        # "./exports/ragas/wizard/run_20260721_101509/benchmark.json",
        # "./exports/ares/wizard/run_20260721_112920/benchmark.json",
        # "./exports/datamorgana/wizard/run_20260721_124759/benchmark.json",
        # "./exports/deepeval/wizard/run_20260721_140125/benchmark.json"

        # "./exports/ragas/wiki/run_20260721_142743/benchmark.json",
        # "./exports/ares/wiki/run_20260721_151026/benchmark.json",
        # "./exports/datamorgana/wiki/run_20260721_181309/benchmark.json",
        # "./exports/deepeval/wiki/run_20260721_194626/benchmark.json"

        "./exports/ragas/covid/run_20260721_195834/benchmark.json",
        "./exports/ares/covid/run_20260721_200554/benchmark.json",
        "./exports/datamorgana/covid/run_20260721_201250/benchmark.json",
        # "./exports/deepeval/covid/run_20260707_140000/benchmark.json"
    ]

    # Output merged JSON
    output_file = "./union_benchmarks/ratio_100/covid_benchmark.json"

    merge_json_files(
        input_files,
        output_file
    )