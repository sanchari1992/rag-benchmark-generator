from dataclasses import dataclass


@dataclass
class DatasetSizingConfig:
    """
    Defines how benchmark size scales with corpus size.
    """

    ratio: float = 0.05          # default 5%
    min_size: int = 500          # lower bound
    max_size: int = 10000         # upper bound


def compute_benchmark_size(num_chunks: int, config: DatasetSizingConfig):
    """
    Computes benchmark size from corpus size.

    This is your key research variable:
        |D| = f(|C|)

    where:
        D = benchmark dataset
        C = corpus (RAG chunks)
    """

    raw_size = int(num_chunks * config.ratio)

    return max(
        config.min_size,
        min(raw_size, config.max_size)
    )


def scaling_profile(num_chunks: int):
    """
    Returns multiple benchmark sizes for ablation studies.
    Useful for paper experiments.
    """

    return {
        "1%": compute_benchmark_size(num_chunks, DatasetSizingConfig(0.01)),
        "5%": compute_benchmark_size(num_chunks, DatasetSizingConfig(0.05)),
        "10%": compute_benchmark_size(num_chunks, DatasetSizingConfig(0.10)),
    }