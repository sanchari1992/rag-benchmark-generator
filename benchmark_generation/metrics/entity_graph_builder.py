import networkx as nx
import spacy

nlp = spacy.load("en_core_web_sm")


def build_entity_graph(text):
    """
    Build an undirected entity co-occurrence graph.

    Nodes:
        Named entities.

    Edges:
        Two entities are connected if they appear
        in the same sentence.
    """

    doc = nlp(text)

    G = nx.Graph()

    # Add entity nodes
    for ent in doc.ents:
        G.add_node(ent.text)

    # Connect entities appearing in the same sentence
    for sent in doc.sents:

        entities = [
            ent.text
            for ent in doc.ents
            if ent.start >= sent.start and ent.end <= sent.end
        ]

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                G.add_edge(entities[i], entities[j])

    return G