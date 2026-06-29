"""
lexical_metrics.py

Simple lexical statistics for benchmark questions and answers.

Metrics implemented
-------------------
Question Metrics
- Question word count
- Question character count
- Question average word length
- Question sentence count
- Question longest sentence word count
- Question hapax ratio

Answer Metrics
- Answer word count
- Answer character count
- Answer average word length
- Answer sentence count
- Answer longest sentence word count
- Answer hapax ratio
"""

import re
from collections import Counter


###########################################################
# Utility Functions
###########################################################

def tokenize(text):
    """
    Simple word tokenizer.
    """
    if text is None:
        return []

    return re.findall(r"\b[\w'-]+\b", text.lower())


def split_sentences(text):
    """
    Very lightweight sentence splitter.

    Splits text on '.', '!', and '?'.
    """
    if text is None:
        return []

    sentences = re.split(r"[.!?]+", text)

    return [sentence.strip() for sentence in sentences if sentence.strip()]


###########################################################
# Question Metrics
###########################################################

def question_word_count(record):
    return len(tokenize(record.get("question", "")))


def question_character_count(record):
    return len(record.get("question", ""))


def question_average_word_length(record):
    tokens = tokenize(record.get("question", ""))

    if not tokens:
        return 0.0

    return sum(len(token) for token in tokens) / len(tokens)


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


###########################################################
# Answer Metrics
###########################################################

def answer_word_count(record):
    return len(tokenize(record.get("answer", "")))


def answer_character_count(record):
    return len(record.get("answer", ""))


def answer_average_word_length(record):
    tokens = tokenize(record.get("answer", ""))

    if not tokens:
        return 0.0

    return sum(len(token) for token in tokens) / len(tokens)


def answer_sentence_count(record):
    return len(split_sentences(record.get("answer", "")))


def answer_longest_sentence_word_count(record):
    sentences = split_sentences(record.get("answer", ""))

    if not sentences:
        return 0

    return max(len(tokenize(sentence)) for sentence in sentences)


def answer_hapax_ratio(record):
    """
    Ratio of words occurring exactly once to total words.
    """
    tokens = tokenize(record.get("answer", ""))

    if not tokens:
        return 0.0

    counts = Counter(tokens)
    hapax = sum(1 for count in counts.values() if count == 1)

    return hapax / len(tokens)


###########################################################
# Main Compute Function
###########################################################

def compute(records):
    """
    Compute all lexical metrics for every benchmark record.
    """

    for record in records:

        # Question metrics
        record["question_word_count"] = question_word_count(record)
        record["question_character_count"] = question_character_count(record)
        record["question_average_word_length"] = round(
            question_average_word_length(record), 2
        )
        record["question_sentence_count"] = question_sentence_count(record)
        record["question_longest_sentence_word_count"] = (
            question_longest_sentence_word_count(record)
        )
        record["question_hapax_ratio"] = round(
            question_hapax_ratio(record), 3
        )

        # Answer metrics
        record["answer_word_count"] = answer_word_count(record)
        record["answer_character_count"] = answer_character_count(record)
        record["answer_average_word_length"] = round(
            answer_average_word_length(record), 2
        )
        record["answer_sentence_count"] = answer_sentence_count(record)
        record["answer_longest_sentence_word_count"] = (
            answer_longest_sentence_word_count(record)
        )
        record["answer_hapax_ratio"] = round(
            answer_hapax_ratio(record), 3
        )

    return records