from benchmark_generation.strategies.ragas.strategy import (
    RAGASStrategy
)

from benchmark_generation.strategies.ares.strategy import (
    ARESStrategy
)

from benchmark_generation.strategies.datamorgana.strategy import (
    DataMorganaStrategy
)

from benchmark_generation.strategies.deepeval.strategy import (
    DeepEvalStrategy
)


def load_strategy(name):

    if name.lower() == "ragas":
        return RAGASStrategy()

    if name.lower() == "ares":
        return ARESStrategy()
    
    if name.lower() == "datamorgana":
        return DataMorganaStrategy()

    if name.lower() == "deepeval":
        return DeepEvalStrategy()

    raise ValueError(
        f"Unknown strategy: {name}"
    )