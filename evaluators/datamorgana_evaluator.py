import json
import gzip
from pathlib import Path
from collections import Counter

import spacy
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------
# Models
# --------------------------------------------------------------------

nlp = spacy.load("en_core_web_sm")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# --------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------

def load_questions(json_file):

    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    return [
        item["question"]
        for item in data
        if "question" in item
    ]


def tokenize(text):

    return [
        token.text.lower()
        for token in nlp(text)
        if not token.is_space
    ]


# --------------------------------------------------------------------
# N-Gram Diversity
# --------------------------------------------------------------------

def ngrams(tokens, n):

    return [
        tuple(tokens[i:i+n])
        for i in range(len(tokens)-n+1)
    ]


def ngram_diversity(questions):

    total_score = 0.0

    for n in range(1, 5):

        unique = set()

        total = 0

        for q in questions:

            tokens = tokenize(q)

            ng = ngrams(tokens, n)

            unique.update(ng)

            total += len(ng)

        if total > 0:

            total_score += len(unique) / total

    return float(total_score)


# --------------------------------------------------------------------
# Self-Repetition Score
# --------------------------------------------------------------------

def self_repetition_score(questions, n=4):

    question_ngrams = []

    all_counter = Counter()

    for q in questions:

        tokens = tokenize(q)

        ng = set(
            ngrams(tokens, n)
        )

        question_ngrams.append(ng)

        all_counter.update(ng)

    repeated = 0

    for ng_set in question_ngrams:

        if any(
            all_counter[g] > 1
            for g in ng_set
        ):
            repeated += 1

    if len(questions) == 0:
        return 0.0

    return float(repeated / len(questions))


# --------------------------------------------------------------------
# Compression Ratio
# --------------------------------------------------------------------

def compression_ratio(text):

    raw = text.encode("utf-8")

    compressed = gzip.compress(raw)

    return float(len(raw) / len(compressed))


def word_cr(questions):

    return compression_ratio(
        "\n".join(questions)
    )


def pos_cr(questions):

    pos_sequences = []

    for q in questions:

        doc = nlp(q)

        pos_sequences.append(

            " ".join(
                token.pos_
                for token in doc
            )

        )

    return compression_ratio(
        "\n".join(pos_sequences)
    )


# --------------------------------------------------------------------
# Homogenization Score
# --------------------------------------------------------------------

def homogenization_score(questions):

    if len(questions) < 2:
        return 0.0

    embeddings = embedding_model.encode(
        questions,
        convert_to_numpy=True
    )

    similarity = cosine_similarity(
        embeddings
    )

    np.fill_diagonal(
        similarity,
        0
    )

    score = similarity.sum() / (
        len(questions) * (len(questions) - 1)
    )

    return float(score)


# --------------------------------------------------------------------
# Main Evaluation
# --------------------------------------------------------------------

def evaluate_datamorgana(
    benchmark_path,
    output_path
):

    questions = load_questions(
        benchmark_path
    )

    metrics = {

        "num_questions":
            int(len(questions)),

        "ngram_diversity":
            float(round(
                ngram_diversity(questions),
                4
            )),

        "self_repetition_score":
            float(round(
                self_repetition_score(questions),
                4
            )),

        "word_compression_ratio":
            float(round(
                word_cr(questions),
                4
            )),

        "pos_compression_ratio":
            float(round(
                pos_cr(questions),
                4
            )),

        "embedding_homogenization_score":
            float(round(
                homogenization_score(questions),
                4
            ))
    }

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
            ensure_ascii=False
        )

    return metrics


if __name__ == "__main__":

    metrics = evaluate_datamorgana(

        benchmark_path="exports/datamorgana/covid/run_20260626_120758/benchmark.json",

        output_path="exports/datamorgana/covid/run_20260626_120758/diversity_metrics.json"

    )

    print(
        json.dumps(
            metrics,
            indent=2
        )
    )