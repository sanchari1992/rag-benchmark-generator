from adapters.neural_networks_rag_adapter import (
    NeuralNetworksRAGAdapter
)

adapter = NeuralNetworksRAGAdapter(
    "repositories/Neural_networks_RAG"
)

# ----------------------------
# Create config yaml
# ----------------------------

config = adapter.save_config(
    "configs/neural_networks_rag.yaml"
)

print("\nCONFIG\n")
print(config)

# ----------------------------
# Export corpus
# ----------------------------

adapter.save_chunks(
    "exports/neural_chunks.json"
)

print(
    "\nChunks exported successfully."
)

# ----------------------------
# Test retrieval
# ----------------------------

# results = adapter.retrieve(
#     "Why was Edmond Dantes imprisoned?"
# )

# print("\nRETRIEVAL RESULTS\n")

# for i, doc in enumerate(results):

#     print("=" * 80)

#     print(
#         f"RESULT {i+1}"
#     )

#     print(
#         doc["raw_text"][:1000]
#     )