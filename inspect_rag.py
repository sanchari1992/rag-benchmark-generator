from pathlib import Path

from adapters.factory import (
    load_adapter
)

# =====================================================
# SELECT RAG
# =====================================================

RAG_NAME = "ragify"
# RAG_NAME = "neural"

# =====================================================
# LOAD ADAPTER
# =====================================================

print("\nLoading adapter...\n")

adapter = load_adapter(
    RAG_NAME
)

print(
    f"Adapter loaded: {RAG_NAME}"
)

# =====================================================
# OUTPUT PATHS
# =====================================================

config_path = (
    f"configs/{RAG_NAME}.yaml"
)

chunks_path = (
    f"exports/{RAG_NAME}_chunks.json"
)

Path("configs").mkdir(
    exist_ok=True
)

Path("exports").mkdir(
    exist_ok=True
)

# =====================================================
# BUILD CONFIG
# =====================================================

print("\nBuilding config...\n")

config = adapter.save_config(
    config_path
)

print("CONFIG\n")

for k, v in config.items():

    print(
        f"{k}: {v}"
    )

# =====================================================
# EXPORT CHUNKS
# =====================================================

print("\nExporting chunks...\n")

adapter.save_chunks(
    chunks_path
)

print(
    f"Chunks exported to {chunks_path}"
)

# =====================================================
# RETRIEVAL TEST
# =====================================================

print("\nTesting retrieval...\n")

TEST_QUERY = (
    "What are the company policies?"
    if RAG_NAME == "ragify"
    else
    "Why was Edmond Dantes imprisoned?"
)

results = adapter.retrieve(
    TEST_QUERY,
    k=3
)

print(
    f"\nQuery: {TEST_QUERY}\n"
)

print(
    f"Retrieved {len(results)} chunks\n"
)

for i, doc in enumerate(results):

    print("=" * 80)

    print(
        f"RESULT {i+1}"
    )

    print("-" * 80)

    # RAGify format
    if "source_document" in doc:

        print(
            f"Source: {doc['source_document']}"
        )

        print()

        print(
            doc["text"][:1000]
        )

    # Neural Networks RAG format
    elif "raw_text" in doc:

        print(
            doc["raw_text"][:1000]
        )

    # Generic fallback
    elif "text" in doc:

        print(
            doc["text"][:1000]
        )

    print("\n")