"""
syntax_metrics.py

Syntactic metrics for benchmark questions.

These metrics focus on syntactic properties that are useful for
identifying missing benchmark coverage and generating new question
variants.

Metrics implemented
-------------------

Sentence Structure
- Average dependency distance
- Clause count
- Subordinate clause count
- Coordination count

Question Form
- Interrogative type
- Conditional count
- Negation count

Distribution
- POS entropy

Construction
- Passive voice ratio

Dependencies:
    pip install spacy
    python -m spacy download en_core_web_sm
"""

import math
from collections import Counter
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


###########################################################
# Utility
###########################################################

SUBORDINATE_MARKERS = {
    "because",
    "although",
    "though",
    "while",
    "whereas",
    "since",
    "if",
    "unless",
    "until",
    "before",
    "after",
    "once",
    "when",
    "whenever",
}

COORDINATION_MARKERS = {
    "and",
    "or",
    "but",
    "nor",
    "yet",
}

NEGATION_MARKERS = {
    "not",
    "no",
    "never",
    "neither",
    "without",
    "except",
    "unless",
}

CONDITIONAL_MARKERS = {
    "if",
    "assuming",
    "suppose",
    "provided",
    "providing",
    "unless",
    "otherwise",
    "case",
}


def safe_doc(text):
    if text is None:
        text = ""
    return nlp(str(text))


def entropy(counter, total):
    if total == 0:
        return 0.0

    return -sum(
        (count / total) * math.log2(count / total)
        for count in counter.values()
    )


###########################################################
# Dependency Distance
###########################################################

def dependency_distance(doc):
    """
    Average absolute distance between each token and its head.
    """
    distances = [
        abs(token.i - token.head.i)
        for token in doc
        if token.head != token
    ]

    if not distances:
        return 0.0

    return sum(distances) / len(distances)


def question_dependency_distance(record):
    return dependency_distance(safe_doc(record.get("question", "")))


###########################################################
# Clause Count
###########################################################

def clause_count(doc):
    """
    Approximate clause count using finite verbs.
    """
    return sum(
        1
        for token in doc
        if token.pos_ in {"VERB", "AUX"} and token.dep_ != "aux"
    )


def question_clause_count(record):
    return clause_count(safe_doc(record.get("question", "")))


###########################################################
# Subordinate Clause Count
###########################################################

def subordinate_clause_count(doc):
    """
    Counts subordinate clause markers.
    """
    return sum(
        1
        for token in doc
        if token.text.lower() in SUBORDINATE_MARKERS
    )


def question_subordinate_clause_count(record):
    return subordinate_clause_count(
        safe_doc(record.get("question", ""))
    )


###########################################################
# Coordination Count
###########################################################

def coordination_count(doc):
    """
    Counts coordinating conjunctions.
    """
    return sum(
        1
        for token in doc
        if token.text.lower() in COORDINATION_MARKERS
    )


def question_coordination_count(record):
    return coordination_count(
        safe_doc(record.get("question", ""))
    )


###########################################################
# POS Entropy
###########################################################

def pos_entropy(doc):
    tags = [token.pos_ for token in doc]

    counter = Counter(tags)

    return entropy(counter, len(tags))


def question_pos_entropy(record):
    return pos_entropy(
        safe_doc(record.get("question", ""))
    )


###########################################################
# Passive Voice
###########################################################

def passive_voice_ratio(doc):
    """
    Ratio of sentences written in passive voice.
    """
    passive = 0
    total = len(list(doc.sents))

    if total == 0:
        return 0.0

    for sent in doc.sents:

        if any(
            token.dep_ == "auxpass"
            or token.dep_ == "nsubjpass"
            for token in sent
        ):
            passive += 1

    return passive / total


def question_passive_voice(record):
    return passive_voice_ratio(
        safe_doc(record.get("question", ""))
    )


###########################################################
# Negation Count
###########################################################

def negation_count(doc):
    """
    Counts negation words.
    """
    return sum(
        1
        for token in doc
        if token.text.lower() in NEGATION_MARKERS
    )


def question_negation_count(record):
    return negation_count(
        safe_doc(record.get("question", ""))
    )


###########################################################
# Conditional Count
###########################################################

def conditional_count(doc):
    """
    Counts conditional markers.
    """
    return sum(
        1
        for token in doc
        if token.text.lower() in CONDITIONAL_MARKERS
    )


def question_conditional_count(record):
    return conditional_count(
        safe_doc(record.get("question", ""))
    )


###########################################################
# Interrogative Type
###########################################################

def interrogative_type(record):
    """
    Classifies question by its opening word.

    Returns one of:
    what, who, when, where, why, how,
    which, yes_no, other
    """
    text = record.get("question", "").strip().lower()

    if not text:
        return "other"

    first = text.split()[0]

    wh = {
        "what",
        "who",
        "when",
        "where",
        "why",
        "how",
        "which",
        "whose",
        "whom",
    }

    if first in wh:
        return first

    yes_no = {
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
        "had",
        "may",
        "might",
    }

    if first in yes_no:
        return "yes_no"

    return "other"


###########################################################
# Main Compute Function
###########################################################

def compute(records):
    """
    Compute syntactic metrics for benchmark questions.
    """

    for record in records:

        record["question_dependency_distance"] = round(
            question_dependency_distance(record),
            2,
        )

        record["question_clause_count"] = (
            question_clause_count(record)
        )

        record["question_subordinate_clause_count"] = (
            question_subordinate_clause_count(record)
        )

        record["question_coordination_count"] = (
            question_coordination_count(record)
        )

        record["question_pos_entropy"] = round(
            question_pos_entropy(record),
            3,
        )

        record["question_passive_voice_ratio"] = round(
            question_passive_voice(record),
            3,
        )

        record["question_negation_count"] = (
            question_negation_count(record)
        )

        record["question_conditional_count"] = (
            question_conditional_count(record)
        )

        record["question_interrogative_type"] = (
            interrogative_type(record)
        )

    return records