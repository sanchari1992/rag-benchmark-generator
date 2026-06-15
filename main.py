import os

from adapters.neural_networks_rag_adapter import NeuralNetworksRAGAdapter

from benchmark_generation.llm.client import LLMClient
from benchmark_generation.build_benchmark import build_benchmark
from benchmark_generation.strategies.ragas import RAGASStrategy


# -------------------------
# Adapter
# -------------------------
adapter = NeuralNetworksRAGAdapter(
    "repositories/Neural_networks_RAG"
)

# -------------------------
# LLM
# -------------------------
llm = LLMClient(
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# -------------------------
# Strategy (PLUG-IN)
# -------------------------
strategy = RAGASStrategy()

# -------------------------
# Build benchmark
# -------------------------
dataset = build_benchmark(
    adapter=adapter,
    llm=llm,
    model="deepseek-chat",
    strategy=strategy,
    sample_size=50,
    output_path="exports/ragas_benchmark.json"
)

print("DONE:", len(dataset))