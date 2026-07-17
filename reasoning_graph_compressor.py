import json
import argparse
import hashlib
import numpy as np

from tqdm import tqdm

import spacy
import networkx as nx

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


# -----------------------------
# Load NLP model
# -----------------------------

nlp = spacy.load("en_core_web_sm")


# -----------------------------
# Extract reasoning graph
# -----------------------------

def build_reasoning_graph(question, answer):

    text = question + " " + answer

    doc = nlp(text)

    G = nx.Graph()

    entities = []

    # Entity nodes
    for ent in doc.ents:
        node = ent.text.lower().strip()
        entities.append(node)

        G.add_node(
            node,
            type="entity",
            label=ent.label_
        )


    # Dependency based relations
    for token in doc:

        if token.dep_ in ["nsubj", "dobj", "pobj"]:

            head = token.head.text.lower()
            child = token.text.lower()

            G.add_node(head, type="concept")
            G.add_node(child, type="concept")

            G.add_edge(
                head,
                child,
                relation=token.dep_
            )


    # If no edges, create semantic fallback graph
    if G.number_of_edges() == 0:

        words = [
            t.lemma_.lower()
            for t in doc
            if not t.is_stop and t.is_alpha
        ]

        for i in range(len(words)-1):

            G.add_edge(
                words[i],
                words[i+1],
                relation="sequence"
            )


    return G



# -----------------------------
# Graph feature extraction
# -----------------------------

def graph_features(G):

    features = []

    # number of nodes
    features.append(G.number_of_nodes())

    # number of edges
    features.append(G.number_of_edges())

    # density
    features.append(nx.density(G))


    # average degree

    if G.number_of_nodes()>0:
        avg_degree = sum(
            dict(G.degree()).values()
        ) / G.number_of_nodes()
    else:
        avg_degree=0

    features.append(avg_degree)


    # clustering coefficient

    features.append(
        nx.average_clustering(G)
        if G.number_of_nodes()>1
        else 0
    )


    # graph hash
    wl_hash = nx.weisfeiler_lehman_graph_hash(G)

    # convert hash to numbers
    hash_vector = [
        int(wl_hash[i:i+4],16)
        for i in range(0,len(wl_hash),4)
    ]

    features.extend(hash_vector)


    return features



# -----------------------------
# Select representative question
# -----------------------------

def select_medoid(indices, embeddings):

    cluster_vectors = embeddings[indices]

    centroid = np.mean(
        cluster_vectors,
        axis=0
    )


    distances = np.linalg.norm(
        cluster_vectors-centroid,
        axis=1
    )


    return indices[
        np.argmin(distances)
    ]



# -----------------------------
# Main compressor
# -----------------------------

def compress_dataset(
        input_file,
        output_file,
        n_clusters
):

    with open(input_file,"r") as f:
        data=json.load(f)


    graphs=[]
    features=[]


    print("Building reasoning graphs...")


    for item in tqdm(data):

        G=build_reasoning_graph(
            item["question"],
            item["answer"]
        )

        graphs.append(G)

        features.append(
            graph_features(G)
        )


    X=np.array(features)

    X=np.nan_to_num(X)

    X=normalize(X)



    # -------------------------
    # Clustering
    # -------------------------

    print("Clustering graphs...")


    k=min(
        n_clusters,
        len(data)
    )


    model=KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )


    labels=model.fit_predict(X)



    # -------------------------
    # Create clusters
    # -------------------------

    clusters={}


    for idx,label in enumerate(labels):

        clusters.setdefault(
            int(label),
            []
        ).append(idx)



    # -------------------------
    # Pick representatives
    # -------------------------

    compressed=[]


    for cluster_id,indices in clusters.items():

        representative = select_medoid(
            indices,
            X
        )

        item=data[representative].copy()

        item["cluster_id"]=cluster_id

        item["cluster_size"]=len(indices)

        item["reasoning_graph_nodes"]=graphs[
            representative
        ].number_of_nodes()

        item["reasoning_graph_edges"]=graphs[
            representative
        ].number_of_edges()


        compressed.append(item)



    # -------------------------
    # Metrics
    # -------------------------

    original_size=len(data)

    compressed_size=len(compressed)


    compression_ratio=(
        1-
        compressed_size/original_size
    )*100



    # graph coverage

    original_graph_patterns=set()

    compressed_graph_patterns=set()


    for G in graphs:

        original_graph_patterns.add(
            nx.weisfeiler_lehman_graph_hash(G)
        )


    for item in compressed:

        G=build_reasoning_graph(
            item["question"],
            item["answer"]
        )

        compressed_graph_patterns.add(
            nx.weisfeiler_lehman_graph_hash(G)
        )


    coverage=(
        len(compressed_graph_patterns)
        /
        len(original_graph_patterns)
    )*100



    result={

        "metrics":{

            "original_questions":
                original_size,

            "compressed_questions":
                compressed_size,

            "compression_ratio_percent":
                round(compression_ratio,2),

            "number_of_clusters":
                len(clusters),

            "unique_original_reasoning_graphs":
                len(original_graph_patterns),

            "unique_compressed_reasoning_graphs":
                len(compressed_graph_patterns),

            "graph_coverage_percent":
                round(coverage,2)
        },


        "clusters":clusters,


        "compressed_dataset":
            compressed

    }


    with open(output_file,"w") as f:
        json.dump(
            result,
            f,
            indent=4
        )


    print("\nDone!")
    print(json.dumps(
        result["metrics"],
        indent=4
    ))



# -----------------------------
# CLI
# -----------------------------

if __name__=="__main__":

    parser=argparse.ArgumentParser(
        description=
        "Reasoning Graph Coverage Compression"
    )


    parser.add_argument(
        "--input",
        required=True
    )


    parser.add_argument(
        "--output",
        required=True
    )


    parser.add_argument(
        "--clusters",
        type=int,
        default=50
    )


    args=parser.parse_args()


    compress_dataset(
        args.input,
        args.output,
        args.clusters
    )