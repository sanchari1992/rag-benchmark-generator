"""
lexical_metrics.py

Simple lexical statistics for benchmark questions and answers.
"""

import re


def tokenize(text):
    """
    Simple word tokenizer.
    """
    if text is None:
        return []

    return re.findall(r"\b[\w'-]+\b", text)


###########################################################
# Individual Metrics
###########################################################

def question_word_count(record):
    """
    Number of words in the question.
    """

    tokens = tokenize(record.get("question", ""))

    return len(tokens)


def answer_word_count(record):
    """
    Number of words in the answer.
    """

    tokens = tokenize(record.get("answer", ""))

    return len(tokens)


###########################################################
# Main Compute Function
###########################################################

def compute(records):

    for record in records:

        record["question_word_count"] = question_word_count(record)

        record["answer_word_count"] = answer_word_count(record)

    return records