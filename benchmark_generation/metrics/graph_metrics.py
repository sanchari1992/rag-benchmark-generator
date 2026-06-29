"""
graph_metrics.py

Entity graph metrics.
"""

import networkx as nx
import spacy

nlp = spacy.load("en_core_web_sm")


###########################################################
# Utilities
###########################################################

def extract_entities(text):
    """
    Returns list of named entities.
    """

    if text is None:
        return []

    doc = nlp(text)

    return [ent.text.strip() for ent in doc.ents]


def build_entity_graph(entities):
    """
    Creates a simple co-occurrence graph.

    Every entity in the same text is connected.
    """

    G = nx.Graph()

    for entity in entities:
        G.add_node(entity)

    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            G.add_edge(entities[i], entities[j])

    return G


###########################################################
# Metrics
###########################################################

def entity_count(record):
    """
    Number of unique named entities
    in question + answer.
    """

    text = (
        record.get("question", "")
        + " "
        + record.get("answer", "")
    )

    entities = extract_entities(text)

    return len(set(entities))


def entity_density(record):
    """
    Entity density =
    unique entities / total words
    """

    text = (
        record.get("question", "")
        + " "
        + record.get("answer", "")
    )

    entities = extract_entities(text)

    words = text.split()

    if len(words) == 0:
        return 0.0

    return len(set(entities)) / len(words)


###########################################################
# Optional graph object
###########################################################

def entity_graph(record):
    """
    Build entity graph.

    Not written into JSON because NetworkX
    objects aren't serializable, but future
    metrics can use it.
    """

    text = (
        record.get("question", "")
        + " "
        + record.get("answer", "")
    )

    entities = extract_entities(text)

    return build_entity_graph(entities)


###########################################################
# Compute
###########################################################

def compute(records):

    for record in records:

        record["entity_count"] = entity_count(record)

        record["entity_density"] = round(
            entity_density(record),
            4,
        )

    return records