# Usage:
# python generate_metrics_graph.py
#
# Before running, edit the following variables:
#   INPUT_FILE   = "path/to/metrics.json"
#   OUTPUT_FOLDER = "path/to/output"
#   OUTPUT_NAME   = "metric_statistics.png"
#
# Example:
#   INPUT_FILE = "results/metric_statistics.json"
#   OUTPUT_FOLDER = "figures"
#   OUTPUT_NAME = "metrics_summary.png"

import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# User Settings
# ==========================
INPUT_FILE = "metrics_results/ragas/wiki/run_20260622_112350/summary_metrics.json"
OUTPUT_FOLDER = "metrics_results/ragas/wiki/run_20260622_112350"
OUTPUT_NAME = "metric_statistics.png"
# ==========================

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load JSON
with open(INPUT_FILE, "r") as f:
    data = json.load(f)

metrics = []
mins = []
avgs = []
maxs = []

# Extract metrics
for metric, values in data.items():
    if isinstance(values, dict) and all(k in values for k in ("minimum", "average", "maximum")):
        metrics.append(metric)
        mins.append(values["minimum"])
        avgs.append(values["average"])
        maxs.append(values["maximum"])

# Plot
y = np.arange(len(metrics))
bar_height = 0.25

fig_height = max(6, len(metrics) * 0.35)

plt.figure(figsize=(12, fig_height))

plt.barh(y - bar_height, mins, height=bar_height, label="Minimum")
plt.barh(y, avgs, height=bar_height, label="Average")
plt.barh(y + bar_height, maxs, height=bar_height, label="Maximum")

plt.yticks(y, metrics)
plt.xlabel("Value")
plt.title("Metric Statistics")
plt.legend()

plt.tight_layout()

# Save figure
output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_NAME)
plt.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()

print(f"Saved graph to: {output_path}")