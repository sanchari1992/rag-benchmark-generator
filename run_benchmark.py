import os
from dotenv import load_dotenv

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
from benchmark_generation.strategies.ragas import RAGASStrategy

# ----------------------------
# Size module
# ----------------------------
from benchmark_generation.core.sizing import DatasetSizingConfig, compute_benchmark_size


# =========================================================
# STEP 1: Load RAG Adapter (MUST COME FIRST)
# =========================================================

print("\n[1/4] Loading RAG adapter...")

REPO_PATH = "repositories/Neural_networks_RAG"
OUTPUT_PATH = "exports/ragas_benchmark.json"
MODEL_NAME = "deepseek-chat"

adapter = NeuralNetworksRAGAdapter(REPO_PATH)

stats = adapter.inspect()   # ✅ NOW IT EXISTS

print("✔ Adapter loaded")
print(f"   - Chunks: {stats['num_chunks']}")


# =========================================================
# STEP 2: Compute benchmark size (AFTER stats exist)
# =========================================================

SAMPLE_SIZE = compute_benchmark_size(
    num_chunks=stats["num_chunks"],
    config=DatasetSizingConfig(
        ratio=0.05,
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

strategy = RAGASStrategy()

print("✔ RAGAS strategy active")


# =========================================================
# STEP 5: Build Benchmark
# =========================================================

print("\n[4/4] Generating benchmark dataset...")

dataset = build_benchmark(
    adapter=adapter,
    llm=llm,
    model=MODEL_NAME,
    strategy=strategy,
    sample_size=SAMPLE_SIZE,
    output_path=OUTPUT_PATH
)

print("\n✔ DONE")
print(f"Dataset size: {len(dataset)}")
print(f"Corpus size: {stats['num_chunks']}")
print(f"Saved to: {OUTPUT_PATH}")