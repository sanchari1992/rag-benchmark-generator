"""
lexical_metrics.py

Lexical statistics for benchmark questions.

Metrics implemented
-------------------
- Question word count
- Question sentence count
- Question longest sentence word count
- Question hapax ratio
- Question lexical diversity (Type-Token Ratio)
- Question rare word ratio
- Question reasoning cue count
"""

import re
from collections import Counter


###########################################################
# Utility Functions
###########################################################

REASONING_CUES = {
    "why",
    "how",
    "explain",
    "compare",
    "contrast",
    "justify",
    "reason",
    "infer",
    "predict",
    "evaluate",
    "analyze",
    "analyse",
    "because",
    "effect",
    "cause",
    "consequence",
    "relationship",
    "impact",
}


def tokenize(text):
    """
    Simple word tokenizer.
    """
    if text is None:
        return []

    return re.findall(r"\b[\w'-]+\b", text.lower())


def split_sentences(text):
    """
    Lightweight sentence splitter.
    """
    if text is None:
        return []

    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


###########################################################
# Question Metrics
###########################################################

def question_word_count(record):
    return len(tokenize(record.get("question", "")))


def question_sentence_count(record):
    return len(split_sentences(record.get("question", "")))


def question_longest_sentence_word_count(record):
    sentences = split_sentences(record.get("question", ""))

    if not sentences:
        return 0

    return max(len(tokenize(sentence)) for sentence in sentences)


def question_hapax_ratio(record):
    """
    Ratio of words occurring exactly once to total words.
    """
    tokens = tokenize(record.get("question", ""))

    if not tokens:
        return 0.0

    counts = Counter(tokens)
    hapax = sum(1 for count in counts.values() if count == 1)

    return hapax / len(tokens)


def question_lexical_diversity(record):
    """
    Type-Token Ratio (TTR).

    Number of unique words divided by total words.
    """
    tokens = tokenize(record.get("question", ""))

    if not tokens:
        return 0.0

    return len(set(tokens)) / len(tokens)


def question_rare_word_ratio(record):
    """
    Ratio of long words (>=7 characters).

    A simple proxy for technical/domain-specific vocabulary.
    """
    tokens = tokenize(record.get("question", ""))

    if not tokens:
        return 0.0

    rare = sum(1 for token in tokens if len(token) >= 7)

    return rare / len(tokens)


def question_reasoning_cue_count(record):
    """
    Counts explicit reasoning cue words.

    These cues often indicate higher-order reasoning questions.
    """
    tokens = tokenize(record.get("question", ""))

    return sum(1 for token in tokens if token in REASONING_CUES)


###########################################################
# Main Compute Function
###########################################################

def compute(records):
    """
    Compute lexical metrics for every benchmark question.
    """

    for record in records:

        record["question_word_count"] = question_word_count(record)

        record["question_sentence_count"] = question_sentence_count(record)

        record["question_longest_sentence_word_count"] = (
            question_longest_sentence_word_count(record)
        )

        record["question_hapax_ratio"] = round(
            question_hapax_ratio(record), 3
        )

        record["question_lexical_diversity"] = round(
            question_lexical_diversity(record), 3
        )

        record["question_rare_word_ratio"] = round(
            question_rare_word_ratio(record), 3
        )

        record["question_reasoning_cue_count"] = (
            question_reasoning_cue_count(record)
        )

    return records