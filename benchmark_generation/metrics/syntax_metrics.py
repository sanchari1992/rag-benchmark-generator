"""
syntax_metrics.py

Syntactic complexity metrics for benchmark questions and answers.

These metrics capture structural and grammatical properties of text,
useful for evaluating reasoning complexity and linguistic variation.

Metrics implemented
-------------------

Tree / Structure
- Parse tree depth (constituency proxy via dependency depth)
- Dependency tree depth
- Average dependency distance

Clause / Phrase structure
- Clause count
- Average noun phrase length
- Average verb phrase length

Distributional syntax
- POS entropy

Voice / construction
- Passive voice detection ratio

Dependencies:
    pip install spacy
    python -m spacy download en_core_web_sm
"""

import math
from collections import Counter
import spacy

# Load spaCy model (fast, sufficient for research metrics)
nlp = spacy.load("en_core_web_sm")


###########################################################
# Utility
###########################################################

def safe_doc(text):
    if text is None:
        text = ""
    return nlp(str(text))


def entropy(counter, total):
    """
    Compute Shannon entropy over POS tags.
    """
    if total == 0:
        return 0.0

    return -sum(
        (count / total) * math.log2(count / total)
        for count in counter.values()
    )


###########################################################
# Tree Depth Metrics
###########################################################

def dependency_tree_depth(doc):
    """
    Max depth of dependency tree.
    """
    def depth(token):
        if not list(token.children):
            return 1
        return 1 + max(depth(child) for child in token.children)

    roots = [token for token in doc if token.head == token]

    if not roots:
        return 0

    return max(depth(root) for root in roots)


def parse_tree_depth(record, field):
    """
    Proxy parse tree depth using dependency tree depth.
    """
    doc = safe_doc(record.get(field, ""))
    return dependency_tree_depth(doc)


def question_parse_tree_depth(record):
    return parse_tree_depth(record, "question")


def answer_parse_tree_depth(record):
    return parse_tree_depth(record, "answer")


def question_dependency_tree_depth(record):
    return question_parse_tree_depth(record)


def answer_dependency_tree_depth(record):
    return answer_parse_tree_depth(record)


###########################################################
# Dependency Distance
###########################################################

def dependency_distance(doc):
    """
    Average absolute distance between token and its head.
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
    doc = safe_doc(record.get("question", ""))
    return dependency_distance(doc)


def answer_dependency_distance(record):
    doc = safe_doc(record.get("answer", ""))
    return dependency_distance(doc)


###########################################################
# Clause Count (heuristic)
###########################################################

def clause_count(doc):
    """
    Approximate clause count using finite verbs.
    """
    return sum(1 for token in doc if token.pos_ == "VERB")


def question_clause_count(record):
    doc = safe_doc(record.get("question", ""))
    return clause_count(doc)


def answer_clause_count(record):
    doc = safe_doc(record.get("answer", ""))
    return clause_count(doc)


###########################################################
# Phrase Lengths
###########################################################

def average_np_length(doc):
    nps = list(doc.noun_chunks)

    if not nps:
        return 0.0

    return sum(len(np) for np in nps) / len(nps)


def average_vp_length(doc):
    vps = [token for token in doc if token.pos_ == "VERB"]

    if not vps:
        return 0.0

    return sum(len(list(token.children)) + 1 for token in vps) / len(vps)


def question_average_np_length(record):
    return average_np_length(safe_doc(record.get("question", "")))


def answer_average_np_length(record):
    return average_np_length(safe_doc(record.get("answer", "")))


def question_average_vp_length(record):
    return average_vp_length(safe_doc(record.get("question", "")))


def answer_average_vp_length(record):
    return average_vp_length(safe_doc(record.get("answer", "")))


###########################################################
# POS Entropy
###########################################################

def pos_entropy(doc):
    tags = [token.pos_ for token in doc]

    counter = Counter(tags)
    total = len(tags)

    return entropy(counter, total)


def question_pos_entropy(record):
    return pos_entropy(safe_doc(record.get("question", "")))


def answer_pos_entropy(record):
    return pos_entropy(safe_doc(record.get("answer", "")))


###########################################################
# Passive Voice Detection
###########################################################

def passive_voice_ratio(doc):
    passive = 0
    total_sentences = len(list(doc.sents))

    for sent in doc.sents:
        for token in sent:
            if token.dep_ == "auxpass" or token.tag_ == "VBN":
                passive += 1
                break

    if total_sentences == 0:
        return 0.0

    return passive / total_sentences


def question_passive_voice(record):
    return passive_voice_ratio(safe_doc(record.get("question", "")))


def answer_passive_voice(record):
    return passive_voice_ratio(safe_doc(record.get("answer", "")))


###########################################################
# Main Compute Function
###########################################################

def compute(records):
    """
    Compute all syntactic metrics for QA pairs.
    """

    for record in records:

        # -------------------------
        # Parse / Tree structure
        # -------------------------
        record["question_parse_tree_depth"] = question_parse_tree_depth(record)
        record["answer_parse_tree_depth"] = answer_parse_tree_depth(record)

        record["question_dependency_tree_depth"] = question_dependency_tree_depth(record)
        record["answer_dependency_tree_depth"] = answer_dependency_tree_depth(record)

        # -------------------------
        # Dependency distance
        # -------------------------
        record["question_dependency_distance"] = question_dependency_distance(record)
        record["answer_dependency_distance"] = answer_dependency_distance(record)

        # -------------------------
        # Clause structure
        # -------------------------
        record["question_clause_count"] = question_clause_count(record)
        record["answer_clause_count"] = answer_clause_count(record)

        # -------------------------
        # Phrase structure
        # -------------------------
        record["question_average_np_length"] = question_average_np_length(record)
        record["answer_average_np_length"] = answer_average_np_length(record)

        record["question_average_vp_length"] = question_average_vp_length(record)
        record["answer_average_vp_length"] = answer_average_vp_length(record)

        # -------------------------
        # POS entropy
        # -------------------------
        record["question_pos_entropy"] = question_pos_entropy(record)
        record["answer_pos_entropy"] = answer_pos_entropy(record)

        # -------------------------
        # Passive voice
        # -------------------------
        record["question_passive_voice_ratio"] = question_passive_voice(record)
        record["answer_passive_voice_ratio"] = answer_passive_voice(record)

    return records