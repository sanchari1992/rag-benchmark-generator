"""
semantic_metrics.py

Semantic metrics for benchmark questions.

These metrics capture semantic properties that are useful for
analyzing benchmark coverage and identifying opportunities for
generating new benchmark questions.

Metrics implemented
-------------------

Entities
- Named entity count
- Entity type diversity
- Multi-entity relation count

Reasoning Semantics
- Constraint count
- Numeric reasoning count
- Temporal reasoning count
- Comparative reasoning count
- Causal reasoning count

Dependencies:
    pip install spacy
    python -m spacy download en_core_web_sm
"""

from collections import Counter
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


###########################################################
# Utility
###########################################################

NUMERIC_ENTITY_TYPES = {
    "DATE",
    "TIME",
    "MONEY",
    "PERCENT",
    "QUANTITY",
    "CARDINAL",
    "ORDINAL",
}

TEMPORAL_WORDS = {
    "before",
    "after",
    "during",
    "since",
    "until",
    "when",
    "while",
    "first",
    "last",
    "previous",
    "next",
    "earlier",
    "later",
}

COMPARATIVE_WORDS = {
    "compare",
    "comparison",
    "difference",
    "different",
    "better",
    "worse",
    "larger",
    "smaller",
    "greater",
    "less",
    "higher",
    "lower",
    "versus",
    "vs",
    "than",
}

CAUSAL_WORDS = {
    "because",
    "therefore",
    "thus",
    "hence",
    "cause",
    "causes",
    "caused",
    "effect",
    "effects",
    "result",
    "results",
    "impact",
    "impacts",
    "due",
    "leads",
    "lead",
    "consequence",
    "consequences",
}

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
    "minimum",
}


def safe_doc(text):
    if text is None:
        text = ""
    return nlp(str(text))


###########################################################
# Entity Metrics
###########################################################

def named_entity_count(doc):
    """
    Number of named entities.
    """
    return len(doc.ents)


def entity_type_diversity(doc):
    """
    Number of unique named entity types.
    """
    return len(set(ent.label_ for ent in doc.ents))


def multi_entity_relation_count(doc):
    """
    Approximate number of entity relations.

    Assumes questions with N entities have N-1 possible
    semantic relationships.
    """
    n = len(doc.ents)

    if n <= 1:
        return 0

    return n - 1


def question_named_entity_count(record):
    return named_entity_count(safe_doc(record.get("question", "")))


def question_entity_type_diversity(record):
    return entity_type_diversity(safe_doc(record.get("question", "")))


def question_multi_entity_relation_count(record):
    return multi_entity_relation_count(
        safe_doc(record.get("question", ""))
    )


###########################################################
# Constraint Metrics
###########################################################

def constraint_count(doc):
    return sum(
        1
        for token in doc
        if token.text.lower() in CONSTRAINT_WORDS
    )


def question_constraint_count(record):
    return constraint_count(safe_doc(record.get("question", "")))


###########################################################
# Numeric Reasoning
###########################################################

def numeric_reasoning_count(doc):
    """
    Counts numeric-related named entities.
    """
    return sum(
        1
        for ent in doc.ents
        if ent.label_ in NUMERIC_ENTITY_TYPES
    )


def question_numeric_reasoning_count(record):
    return numeric_reasoning_count(
        safe_doc(record.get("question", ""))
    )


###########################################################
# Temporal Reasoning
###########################################################

def temporal_reasoning_count(doc):
    return sum(
        1
        for token in doc
        if token.text.lower() in TEMPORAL_WORDS
    )


def question_temporal_reasoning_count(record):
    return temporal_reasoning_count(
        safe_doc(record.get("question", ""))
    )


###########################################################
# Comparative Reasoning
###########################################################

def comparative_reasoning_count(doc):
    return sum(
        1
        for token in doc
        if token.text.lower() in COMPARATIVE_WORDS
    )


def question_comparative_reasoning_count(record):
    return comparative_reasoning_count(
        safe_doc(record.get("question", ""))
    )


###########################################################
# Causal Reasoning
###########################################################

def causal_reasoning_count(doc):
    return sum(
        1
        for token in doc
        if token.text.lower() in CAUSAL_WORDS
    )


def question_causal_reasoning_count(record):
    return causal_reasoning_count(
        safe_doc(record.get("question", ""))
    )


###########################################################
# Main Compute Function
###########################################################

def compute(records):
    """
    Compute semantic metrics for benchmark questions.
    """

    for record in records:

        # ------------------------------------
        # Entity Metrics
        # ------------------------------------

        record["question_named_entity_count"] = (
            question_named_entity_count(record)
        )

        record["question_entity_type_diversity"] = (
            question_entity_type_diversity(record)
        )

        record["question_multi_entity_relation_count"] = (
            question_multi_entity_relation_count(record)
        )

        # ------------------------------------
        # Constraints
        # ------------------------------------

        record["question_constraint_count"] = (
            question_constraint_count(record)
        )

        # ------------------------------------
        # Reasoning Categories
        # ------------------------------------

        record["question_numeric_reasoning_count"] = (
            question_numeric_reasoning_count(record)
        )

        record["question_temporal_reasoning_count"] = (
            question_temporal_reasoning_count(record)
        )

        record["question_comparative_reasoning_count"] = (
            question_comparative_reasoning_count(record)
        )

        record["question_causal_reasoning_count"] = (
            question_causal_reasoning_count(record)
        )

    return records