from adapters.neural_networks_rag_adapter import (
    NeuralNetworksRAGAdapter
)

from adapters.ragify_adapter import (
    RAGifyAdapter
)


def load_adapter(name):

    if name.lower() == "neural":

        return NeuralNetworksRAGAdapter(
            "repositories/Neural_networks_RAG"
        )

    elif name.lower() == "ragify":

        return RAGifyAdapter(
            "repositories/RAGify"
        )

    raise ValueError(
        f"Unknown adapter: {name}"
    )