"""
readability_metrics.py

Readability statistics for benchmark questions and answers.

This module computes standard readability scores used in NLP and
text evaluation research.

Metrics implemented
-------------------

Question Metrics
- Flesch Reading Ease
- Flesch-Kincaid Grade Level
- Gunning Fog Index
- Coleman-Liau Index
- SMOG Index
- Automated Readability Index (ARI)

Answer Metrics
- Flesch Reading Ease
- Flesch-Kincaid Grade Level
- Gunning Fog Index
- Coleman-Liau Index
- SMOG Index
- Automated Readability Index (ARI)

Note:
This module uses the `textstat` library for standardized readability formulas.
Install with:
    pip install textstat
"""

import textstat


###########################################################
# Utility
###########################################################

def safe_text(text):
    """
    Ensure text is always a valid string.
    """
    if text is None:
        return ""
    return str(text)


###########################################################
# Question Readability Metrics
###########################################################

def question_flesch_reading_ease(record):
    return textstat.flesch_reading_ease(
        safe_text(record.get("question", ""))
    )


def question_flesch_kincaid_grade(record):
    return textstat.flesch_kincaid_grade(
        safe_text(record.get("question", ""))
    )


def question_gunning_fog(record):
    return textstat.gunning_fog(
        safe_text(record.get("question", ""))
    )


def question_coleman_liau(record):
    return textstat.coleman_liau_index(
        safe_text(record.get("question", ""))
    )


def question_smog(record):
    return textstat.smog_index(
        safe_text(record.get("question", ""))
    )


def question_ari(record):
    return textstat.automated_readability_index(
        safe_text(record.get("question", ""))
    )


###########################################################
# Answer Readability Metrics
###########################################################

def answer_flesch_reading_ease(record):
    return textstat.flesch_reading_ease(
        safe_text(record.get("answer", ""))
    )


def answer_flesch_kincaid_grade(record):
    return textstat.flesch_kincaid_grade(
        safe_text(record.get("answer", ""))
    )


def answer_gunning_fog(record):
    return textstat.gunning_fog(
        safe_text(record.get("answer", ""))
    )


def answer_coleman_liau(record):
    return textstat.coleman_liau_index(
        safe_text(record.get("answer", ""))
    )


def answer_smog(record):
    return textstat.smog_index(
        safe_text(record.get("answer", ""))
    )


def answer_ari(record):
    return textstat.automated_readability_index(
        safe_text(record.get("answer", ""))
    )


###########################################################
# Main Compute Function
###########################################################

def compute(records):
    """
    Compute readability metrics for each QA record.
    """

    for record in records:

        # -------------------------
        # Question readability
        # -------------------------
        record["question_flesch_reading_ease"] = question_flesch_reading_ease(record)
        record["question_flesch_kincaid_grade"] = question_flesch_kincaid_grade(record)
        record["question_gunning_fog"] = question_gunning_fog(record)
        record["question_coleman_liau"] = question_coleman_liau(record)
        record["question_smog"] = question_smog(record)
        record["question_ari"] = question_ari(record)

        # -------------------------
        # Answer readability
        # -------------------------
        record["answer_flesch_reading_ease"] = answer_flesch_reading_ease(record)
        record["answer_flesch_kincaid_grade"] = answer_flesch_kincaid_grade(record)
        record["answer_gunning_fog"] = answer_gunning_fog(record)
        record["answer_coleman_liau"] = answer_coleman_liau(record)
        record["answer_smog"] = answer_smog(record)
        record["answer_ari"] = answer_ari(record)

    return records