
"""
DualSpace Representative Selection (DSRS)

Research benchmark compression algorithm.

This program compresses a benchmark while preserving

    1. Semantic diversity
    2. Structural diversity

Algorithm:

    Semantic Embedding
            +
    Structural Features
            ↓
    Similarity Matrices
            ↓
    Greedy Facility Location
            ↓
    Representative Benchmark

Example

pyth_compressor.py \
    --input wiki.json \
    --output wiki_50.json \
    --report report.json \
    --ratio 0.5 \
    --semantic-weight 0.7

Author:
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from features import (
    compute_semantic_embeddings,
    compute_structural_features,
)

@dataclass
class BenchmarkItem:

    id: int
    repo_name: str
    strategy: str
    generator_model: str
    source_chunk: int
    context_length: int
    chapter_name: str
    question: str
    answer: str
    source_text: str
    compression_stage: str

    @staticmethod
    def from_json(obj):

        return BenchmarkItem(
            id=obj["id"],
            repo_name=obj["repo_name"],
            strategy=obj["strategy"],
            generator_model=obj["generator_model"],
            source_chunk=obj["source_chunk"],
            context_length=obj["context_length"],
            chapter_name=obj["chapter_name"],
            question=obj["question"],
            answer=obj["answer"],
            source_text=obj.get("source_text", ""),
            compression_stage=obj.get(
                "compression_stage",
                "none",
            ),
        )

    def to_json(self):

        return {
            "id": self.id,
            "repo_name": self.repo_name,
            "strategy": self.strategy,
            "generator_model": self.generator_model,
            "source_chunk": self.source_chunk,
            "context_length": self.context_length,
            "chapter_name": self.chapter_name,
            "question": self.question,
            "answer": self.answer,
            "source_text": self.source_text,
            "compression_stage": self.compression_stage,
        }

def load_benchmark(filename):

    with open(filename, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return [
        BenchmarkItem.from_json(x)
        for x in raw
    ]

def save_benchmark(items, filename):

    with open(filename, "w", encoding="utf-8") as f:

        json.dump(
            [x.to_json() for x in items],
            f,
            indent=2,
            ensure_ascii=False,
        )


def save_report(report, filename):

    with open(filename, "w", encoding="utf-8") as f:

        json.dump(
            report,
            f,
            indent=2,
        )

def normalize_similarity(M):
    """
    Normalize cosine similarity
    into [0,1].
    """

    M = np.asarray(M)

    return (M + 1.0) / 2.0


def build_similarity_matrix(features):

    sim = cosine_similarity(features)

    return normalize_similarity(sim)

def average_coverage(similarity, selected):

    if len(selected) == 0:
        return 0.0

    coverage = np.max(
        similarity[:, selected],
        axis=1,
    )

    return float(np.mean(coverage))


###############################################################################
# Statistics
###############################################################################

def average_pairwise_similarity(similarity, selected):
    """
    Average pairwise similarity among selected questions.
    Lower means more diverse.
    """

    if len(selected) <= 1:
        return 1.0

    sub = similarity[np.ix_(selected, selected)]

    upper = sub[np.triu_indices(len(selected), k=1)]

    return float(np.mean(upper))


def compression_statistics(
    benchmark,
    selected,
    semantic_similarity,
    structural_similarity,
    semantic_cov,
    structural_cov,
    ratio,
    semantic_weight,
    runtime,
):
    """
    Create JSON report.
    """

    pair_semantic = average_pairwise_similarity(
        semantic_similarity,
        selected,
    )

    pair_structural = average_pairwise_similarity(
        structural_similarity,
        selected,
    )

    return {

        "algorithm": "DualSpace Representative Selection",

        "original_size": len(benchmark),

        "compressed_size": len(selected),

        "compression_ratio_requested": ratio,

        "compression_ratio_actual":
            len(selected) / len(benchmark),

        "semantic_weight": semantic_weight,

        "structural_weight":
            1.0 - semantic_weight,

        "semantic_coverage":
            round(semantic_cov, 4),

        "structural_coverage":
            round(structural_cov, 4),

        "average_semantic_similarity":
            round(pair_semantic, 4),

        "average_structural_similarity":
            round(pair_structural, 4),

        "runtime_seconds":
            round(runtime, 2),
    }



###############################################################################
# DualSpace Representative Selection (DSRS)
###############################################################################

def dualspace_selection(
    semantic_similarity,
    structural_similarity,
    ratio=0.5,
    semantic_weight=0.7,
):
    """
    DualSpace Representative Selection

    Parameters
    ----------
    semantic_similarity : ndarray (N,N)

    structural_similarity : ndarray (N,N)

    ratio : float
        Fraction of benchmark to keep.

    semantic_weight : float
        Lambda in paper.

    Returns
    -------
    selected_indices
    """

    n = semantic_similarity.shape[0]

    target = max(1, int(math.ceil(n * ratio)))

    print(f"Selecting {target} representative questions...")

    ###################################################################
    # coverage arrays
    ###################################################################

    semantic_coverage = np.zeros(n)

    structural_coverage = np.zeros(n)

    ###################################################################
    # selected indices
    ###################################################################

    selected = []

    selected_mask = np.zeros(n, dtype=bool)

    ###################################################################
    # greedy loop
    ###################################################################

    progress = tqdm(total=target)

    while len(selected) < target:

        best_gain = -1

        best_idx = None

        ###############################################################
        # Evaluate every candidate
        ###############################################################

        for j in range(n):

            if selected_mask[j]:
                continue

            #
            # semantic improvement
            #

            semantic_gain = np.maximum(
                0,
                semantic_similarity[:, j] - semantic_coverage,
            ).sum()

            #
            # structural improvement
            #

            structural_gain = np.maximum(
                0,
                structural_similarity[:, j] - structural_coverage,
            ).sum()

            #
            # normalize gains
            #

            semantic_gain /= n

            structural_gain /= n

            gain = (
                semantic_weight * semantic_gain
                +
                (1.0 - semantic_weight) * structural_gain
            )

            if gain > best_gain:

                best_gain = gain

                best_idx = j

        ###############################################################
        # Add best question
        ###############################################################

        selected.append(best_idx)

        selected_mask[best_idx] = True

        ###############################################################
        # Update coverage
        ###############################################################

        semantic_coverage = np.maximum(
            semantic_coverage,
            semantic_similarity[:, best_idx],
        )

        structural_coverage = np.maximum(
            structural_coverage,
            structural_similarity[:, best_idx],
        )

        progress.update(1)

    progress.close()

    ###################################################################
    # statistics
    ###################################################################

    semantic_cov = float(np.mean(semantic_coverage))

    structural_cov = float(np.mean(structural_coverage))

    return (
        selected,
        semantic_cov,
        structural_cov,
    )




def print_header():

    print("=" * 60)
    print("DualSpace Representative Selection")
    print("=" * 60)

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--report",
        required=True,
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=0.5,
        help="Fraction of benchmark to keep",
    )

    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.7,
        help="Weight of semantic similarity",
    )

    return parser.parse_args()



def main():

    args = parse_args()

    print_header()

    start = time.time()

    benchmark = load_benchmark(args.input)

    print(f"Loaded {len(benchmark)} questions")

    ############################################################################
    # Extract questions
    ############################################################################

    questions = [
        x.question
        for x in benchmark
    ]

    print()

    print("Computing semantic embeddings...")

    semantic_vectors = compute_semantic_embeddings(
        questions
    )

    print()

    print("Computing structural features...")

    structural_vectors = compute_structural_features(
        questions
    )

    ############################################################################
    # Similarity matrices
    ############################################################################

    print()

    print("Building semantic similarity matrix...")

    semantic_similarity = build_similarity_matrix(
        semantic_vectors
    )

    print("Building structural similarity matrix...")

    structural_similarity = build_similarity_matrix(
        structural_vectors
    )

    ############################################################################
    # Selection
    ############################################################################

    selected_indices, semantic_cov, structural_cov = \
        dualspace_selection(
            semantic_similarity,
            structural_similarity,
            ratio=args.ratio,
            semantic_weight=args.semantic_weight,
        )

    ############################################################################
    # Create reduced benchmark
    ############################################################################

    selected_indices = sorted(selected_indices)

    compressed = []

    for idx in selected_indices:

        item = benchmark[idx]

        item.compression_stage = "DSRS"

        compressed.append(item)

    ############################################################################
    # Save
    ############################################################################

    ############################################################################
    # Save benchmark
    ############################################################################

    print()

    print("Saving compressed benchmark...")

    save_benchmark(
        compressed,
        args.output,
    )

    ############################################################################
    # Report
    ############################################################################

    runtime = time.time() - start

    report = compression_statistics(
        benchmark,
        selected_indices,
        semantic_similarity,
        structural_similarity,
        semantic_cov,
        structural_cov,
        args.ratio,
        args.semantic_weight,
        runtime,
    )

    print("Writing report...")

    save_report(
        report,
        args.report,
    )

    ############################################################################
    # Summary
    ############################################################################

    print()

    print("=" * 60)

    print("Compression Complete")

    print("=" * 60)

    print(f"Original size      : {len(benchmark)}")

    print(f"Compressed size    : {len(compressed)}")

    print(f"Semantic coverage  : {semantic_cov:.4f}")

    print(f"Structural coverage: {structural_cov:.4f}")

    print(f"Runtime            : {runtime:.2f} sec")

    print("=" * 60)

    elapsed = time.time() - start

    print()
    print(f"Finished in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()



