import os
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import json

load_dotenv()

# ----------------------------
# Adapter
# ----------------------------
from adapters.neural_networks_rag_adapter import NeuralNetworksRAGAdapter

# ----------------------------
# LLM
# ----------------------------
from benchmark_generation.llm.client import LLMClient

# ----------------------------
# Benchmark pipeline
# ----------------------------
from benchmark_generation.build_benchmark import build_benchmark

# ----------------------------
# Strategy
# ----------------------------
from benchmark_generation.strategies.ragas.strategy import RAGASStrategy

# ----------------------------
# Size module
# ----------------------------
from benchmark_generation.core.sizing import DatasetSizingConfig, compute_benchmark_size


# =========================================================
# STEP 1: Load RAG Adapter
# =========================================================

print("\n[1/4] Loading RAG adapter...")

MODEL_NAME = "deepseek-chat"

# RAG_NAME = "wizard"
# RAG_NAME = "wiki"
# RAG_NAME = "neural"
# RAG_NAME = "ragify"
RAG_NAME = "covid"

from adapters.factory import load_adapter

adapter = load_adapter(RAG_NAME)

stats = adapter.inspect()

print("✔ Adapter loaded")
print(f"   - Chunks: {stats['num_chunks']}")


# =========================================================
# STEP 2: Compute benchmark size
# =========================================================

SAMPLE_SIZE = compute_benchmark_size(
    num_chunks=stats["num_chunks"],
    config=DatasetSizingConfig(
        # ratio=0.05,
        ratio=1,
        min_size=20,
        max_size=500
    )
)

print(f"Benchmark size (5% rule): {SAMPLE_SIZE}")


# =========================================================
# STEP 3: Initialize LLM
# =========================================================

print("\n[2/4] Initializing LLM...")

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("DEEPSEEK_API_KEY not found")

llm = LLMClient(api_key=api_key)

print("✔ LLM ready")


# =========================================================
# STEP 4: Strategy
# =========================================================

print("\n[3/4] Loading benchmark strategy...")

from benchmark_generation.strategy_factory import load_strategy

# STRATEGY = "ragas"
# STRATEGY = "ares"
STRATEGY = "datamorgana"

strategy = load_strategy(STRATEGY)

print(f"✔ {strategy.name} strategy active")


# =========================================================
# STEP 5: EXPERIMENT SETUP (NEW)
# =========================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

RUN_ID = f"run_{timestamp}"

BASE_DIR = Path("exports") / STRATEGY / RAG_NAME / RUN_ID
BASE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = BASE_DIR / "benchmark.json"
CONFIG_PATH = BASE_DIR / "config.json"
STATS_PATH = BASE_DIR / "stats.json"


# Save run metadata BEFORE generation
run_metadata = {
    "rag": RAG_NAME,
    "strategy": STRATEGY,
    "model": MODEL_NAME,
    "sample_size": SAMPLE_SIZE,
    "num_chunks": stats["num_chunks"],
    "timestamp": timestamp
}

with open(CONFIG_PATH, "w") as f:
    json.dump(run_metadata, f, indent=2)


# =========================================================
# STEP 6: Build Benchmark
# =========================================================

print("\n[4/4] Generating benchmark dataset...")

dataset = build_benchmark(
    adapter=adapter,
    llm=llm,
    model=MODEL_NAME,
    strategy=strategy,
    sample_size=SAMPLE_SIZE,
    output_path=str(OUTPUT_PATH)
)

# Save dataset stats
dataset_stats = {
    "dataset_size": len(dataset),
    "corpus_size": stats["num_chunks"]
}

with open(STATS_PATH, "w") as f:
    json.dump(dataset_stats, f, indent=2)


# =========================================================
# FINAL REPORT
# =========================================================

print("\n✔ DONE")
print(f"Dataset size: {len(dataset)}")
print(f"Corpus size: {stats['num_chunks']}")
print(f"Run saved in: {BASE_DIR}")