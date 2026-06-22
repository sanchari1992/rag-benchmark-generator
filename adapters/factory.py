from adapters.neural_networks_rag_adapter import (
    NeuralNetworksRAGAdapter
)

from adapters.ragify_adapter import (
    RAGifyAdapter
)

from adapters.wiki_ragas_adapter import (
    WikiRagasAdapter
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
    
    elif name.lower() == "wiki":

        return WikiRagasAdapter(
            "repositories/wiki_ragas"
        )

    raise ValueError(
        f"Unknown adapter: {name}"
    )