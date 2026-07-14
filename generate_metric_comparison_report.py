"""
Compare the average values of metrics between multiple human-generated
benchmark JSON files and multiple existing benchmark JSON files.

Each JSON file is expected to have the format:

{
    "metric_name": {
        "average": ...,
        "minimum": ...,
        "maximum": ...
    },
    ...
}

Output:
    - CSV report sorted by |Z-score|
"""

import json
import numpy as np
import pandas as pd

##########################################################
#                     USER INPUTS
##########################################################

# Human-generated benchmark files
HUMAN_FILES = {
    "Human1": "metrics_results\human_generated_examples\neural\summary_metrics.json",
    "Human2": "metrics_results\human_generated_examples\ragify\summary_metrics.json",
    "Human3": "metrics_results\human_generated_examples\wizard\summary_metrics.json",
    "Human4": "metrics_results\human_generated_examples\wiki\summary_metrics.json",
    "Human5": "metrics_results\human_generated_examples\covid\summary_metrics.json"
}

# Existing benchmark files
OTHER_FILES = {
    "RAGAS": "ragas.json",
    "ARES": "ares.json",
    "DataMorgana": "datamorgana.json",
    "DeepEval": "deepeval.json",
    "Neural": "neural.json",
}

# Output CSV
OUTPUT_CSV = "metric_comparison_report.csv"

##########################################################


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


##########################################################
# Load files
##########################################################

human_jsons = {name: load_json(path) for name, path in HUMAN_FILES.items()}
other_jsons = {name: load_json(path) for name, path in OTHER_FILES.items()}

##########################################################
# Get all metrics that appear in at least one file
##########################################################

all_metrics = set()

for d in list(human_jsons.values()) + list(other_jsons.values()):
    all_metrics.update(d.keys())

##########################################################
# Main comparison
##########################################################

rows = []

for metric in sorted(all_metrics):

    ####################################################
    # Skip invalid entries
    ####################################################

    valid = False

    for d in list(human_jsons.values()) + list(other_jsons.values()):
        if (
            metric in d
            and isinstance(d[metric], dict)
            and "average" in d[metric]
        ):
            valid = True
            break

    if not valid:
        continue

    row = {"Metric": metric}

    ####################################################
    # Human values
    ####################################################

    human_values = []

    for name, hjson in human_jsons.items():

        value = hjson.get(metric, {}).get("average", np.nan)

        row[name] = value

        if not np.isnan(value):
            human_values.append(value)

    if len(human_values) == 0:
        continue

    human_mean = np.mean(human_values)
    human_min = np.min(human_values)
    human_max = np.max(human_values)

    ####################################################
    # Existing benchmark values
    ####################################################

    other_values = []

    for name, ojson in other_jsons.items():

        value = ojson.get(metric, {}).get("average", np.nan)

        row[name] = value

        if not np.isnan(value):
            other_values.append(value)

    if len(other_values) == 0:
        continue

    other_mean = np.mean(other_values)
    other_min = np.min(other_values)
    other_max = np.max(other_values)

    ####################################################
    # Statistics
    ####################################################

    other_std = np.std(other_values)

    if other_mean == 0:
        diff_percent = np.nan
    else:
        diff_percent = (
            (human_mean - other_mean)
            / other_mean
            * 100
        )

    if other_std == 0:
        zscore = np.nan
    else:
        zscore = (
            (human_mean - other_mean)
            / other_std
        )

    ####################################################
    # Observation
    ####################################################

    if np.isnan(zscore):
        observation = "No variation"

    elif zscore >= 3:
        observation = "Significantly Higher"

    elif zscore >= 2:
        observation = "Higher"

    elif zscore >= 1:
        observation = "Slightly Higher"

    elif zscore <= -3:
        observation = "Significantly Lower"

    elif zscore <= -2:
        observation = "Lower"

    elif zscore <= -1:
        observation = "Slightly Lower"

    else:
        observation = "Comparable"

    ####################################################
    # Store results
    ####################################################

    row["Human Mean"] = round(human_mean, 4)
    row["Human Min"] = round(human_min, 4)
    row["Human Max"] = round(human_max, 4)

    row["Others Mean"] = round(other_mean, 4)
    row["Others Min"] = round(other_min, 4)
    row["Others Max"] = round(other_max, 4)

    row["Difference (%)"] = (
        round(diff_percent, 2)
        if not np.isnan(diff_percent)
        else np.nan
    )

    row["Z-score"] = (
        round(zscore, 3)
        if not np.isnan(zscore)
        else np.nan
    )

    row["Observation"] = observation

    rows.append(row)

##########################################################
# Create DataFrame
##########################################################

df = pd.DataFrame(rows)

##########################################################
# Rank metrics by absolute Z-score
##########################################################

df["Rank"] = (
    df["Z-score"]
    .abs()
    .rank(method="dense", ascending=False)
)

df = df.sort_values(
    by="Rank"
)

##########################################################
# Save
##########################################################

df.to_csv(OUTPUT_CSV, index=False)

##########################################################
# Print summary
##########################################################

print("\n==============================")
print("Top distinguishing metrics")
print("==============================\n")

cols = [
    "Rank",
    "Metric",
    "Human Mean",
    "Others Mean",
    "Difference (%)",
    "Z-score",
    "Observation",
]

print(df[cols].head(20).to_string(index=False))

print(f"\nFull report saved to:\n{OUTPUT_CSV}")