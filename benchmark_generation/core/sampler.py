import random


def sample_chunks(chunks, k=50, seed=42):

    random.seed(seed)

    if len(chunks) <= k:
        return chunks

    return random.sample(chunks, k)