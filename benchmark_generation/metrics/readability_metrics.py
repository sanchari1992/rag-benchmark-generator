"""
readability_metrics.py

Readability statistics for benchmark questions.

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
# Main Compute Function
###########################################################

def compute(records):
    """
    Compute readability metrics for benchmark questions.
    """

    for record in records:

        record["question_flesch_reading_ease"] = round(
            question_flesch_reading_ease(record), 2
        )

        record["question_flesch_kincaid_grade"] = round(
            question_flesch_kincaid_grade(record), 2
        )

        record["question_gunning_fog"] = round(
            question_gunning_fog(record), 2
        )

        record["question_coleman_liau"] = round(
            question_coleman_liau(record), 2
        )

        record["question_smog"] = round(
            question_smog(record), 2
        )

        record["question_ari"] = round(
            question_ari(record), 2
        )

    return records