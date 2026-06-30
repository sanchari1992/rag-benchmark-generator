"""
semantic_metrics.py

Semantic metrics for benchmark questions and answers.

These metrics capture semantic and information-oriented properties
of benchmark questions and answers.

Metrics implemented
-------------------

Question Metrics
- Named entity count
- Entity density
- WH-word category
- Question type
- Constraint count

Answer Metrics
- Named entity count
- Entity density

Dependencies:
    pip install spacy
    python -m spacy download en_core_web_sm
"""

import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


###########################################################
# Utility
###########################################################

def safe_doc(text):
    if text is None:
        text = ""
    return nlp(str(text))


###########################################################
# Named Entity Metrics
###########################################################

def named_entity_count(doc):
    """
    Number of named entities.
    """
    return len(doc.ents)


def entity_density(doc):
    """
    Named entities divided by total tokens.
    """
    tokens = [token for token in doc if not token.is_space]

    if not tokens:
        return 0.0

    return len(doc.ents) / len(tokens)


def question_named_entity_count(record):
    return named_entity_count(safe_doc(record.get("question", "")))


def answer_named_entity_count(record):
    return named_entity_count(safe_doc(record.get("answer", "")))


def question_entity_density(record):
    return entity_density(safe_doc(record.get("question", "")))


def answer_entity_density(record):
    return entity_density(safe_doc(record.get("answer", "")))


###########################################################
# WH-word Category
###########################################################

WH_WORDS = {
    "what",
    "who",
    "whom",
    "whose",
    "when",
    "where",
    "why",
    "which",
    "how"
}


def wh_word_category(doc):
    """
    Return the first WH-word if present.
    """
    for token in doc:
        word = token.text.lower()

        if word in WH_WORDS:
            return word

    return "none"


def question_wh_word_category(record):
    return wh_word_category(safe_doc(record.get("question", "")))


###########################################################
# Question Type
###########################################################

BOOLEAN_STARTERS = {
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "would",
    "will",
    "has",
    "have",
    "had"
}


def question_type(doc):
    """
    Simple rule-based question classification.
    """

    tokens = [token.text.lower() for token in doc]

    if not tokens:
        return "unknown"

    first = tokens[0]

    if first in BOOLEAN_STARTERS:
        return "boolean"

    if first == "what":
        return "what"

    if first == "who":
        return "who"

    if first == "where":
        return "where"

    if first == "when":
        return "when"

    if first == "why":
        return "why"

    if first == "how":
        return "how"

    if first == "which":
        return "which"

    return "other"


def question_question_type(record):
    return question_type(safe_doc(record.get("question", "")))


###########################################################
# Constraint Count
###########################################################

CONSTRAINT_WORDS = {
    "must",
    "only",
    "exactly",
    "least",
    "most",
    "without",
    "include",
    "excluding",
    "except",
    "using",
    "before",
    "after",
    "between",
    "under",
    "over",
    "less",
    "greater",
    "maximum",
    "minimum"
}


def constraint_count(doc):
    """
    Count explicit constraint words.
    """

    count = 0

    for token in doc:

        if token.text.lower() in CONSTRAINT_WORDS:
            count += 1

    return count


def question_constraint_count(record):
    return constraint_count(safe_doc(record.get("question", "")))


###########################################################
# Main Compute Function
###########################################################

def compute(records):
    """
    Compute semantic metrics for each QA pair.
    """

    for record in records:

        # -------------------------
        # Question metrics
        # -------------------------

        record["question_named_entity_count"] = (
            question_named_entity_count(record)
        )

        record["question_entity_density"] = round(
            question_entity_density(record), 3
        )

        record["question_wh_word_category"] = (
            question_wh_word_category(record)
        )

        record["question_question_type"] = (
            question_question_type(record)
        )

        record["question_constraint_count"] = (
            question_constraint_count(record)
        )

        # -------------------------
        # Answer metrics
        # -------------------------

        record["answer_named_entity_count"] = (
            answer_named_entity_count(record)
        )

        record["answer_entity_density"] = round(
            answer_entity_density(record), 3
        )

    return records