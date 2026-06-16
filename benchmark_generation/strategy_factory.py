from benchmark_generation.strategies.ragas.strategy import (
    RAGASStrategy
)

from benchmark_generation.strategies.ares.strategy import (
    ARESStrategy
)


def load_strategy(name):

    if name.lower() == "ragas":
        return RAGASStrategy()

    if name.lower() == "ares":
        return ARESStrategy()

    raise ValueError(
        f"Unknown strategy: {name}"
    )