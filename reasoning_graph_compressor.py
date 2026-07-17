import json
import argparse
import hashlib

import numpy as np

from tqdm import tqdm

import spacy
import networkx as nx

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


# =====================================================
# Load NLP model
# =====================================================

nlp = spacy.load("en_core_web_sm")


# =====================================================
# Robust JSON loader
# =====================================================

def load_json(filename):

    encodings = [
        "utf-8",
        "utf-8-sig",
        "latin-1"
    ]

    for enc in encodings:

        try:
            with open(
                filename,
                "r",
                encoding=enc
            ) as f:
                return json.load(f)

        except UnicodeDecodeError:
            continue

    raise ValueError(
        "Could not decode JSON file"
    )



# =====================================================
# Reasoning Graph Construction
# =====================================================

def build_reasoning_graph(question, answer):

    text = question + " " + answer

    doc = nlp(text)

    G = nx.Graph()


    # -------------------------
    # Add entity nodes
    # -------------------------

    for ent in doc.ents:

        node = ent.text.lower().strip()

        G.add_node(
            node,
            node_type="entity",
            label=ent.label_
        )


    # -------------------------
    # Add dependency relations
    # -------------------------

    for token in doc:


        if token.dep_ in [
            "nsubj",
            "dobj",
            "pobj",
            "attr",
            "compound"
        ]:

            head = token.head.text.lower()

            child = token.text.lower()


            G.add_node(
                head,
                node_type="concept"
            )

            G.add_node(
                child,
                node_type="concept"
            )


            G.add_edge(
                head,
                child,
                relation=token.dep_
            )


    # -------------------------
    # Fallback graph
    # -------------------------

    if G.number_of_nodes()==0:

        words = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha
            and not token.is_stop
        ]


        for word in words:

            G.add_node(
                word,
                node_type="concept"
            )


        for i in range(len(words)-1):

            G.add_edge(
                words[i],
                words[i+1],
                relation="sequence"
            )


    return G



# =====================================================
# Fixed length graph fingerprint
# =====================================================

def graph_hash_embedding(G, size=16):

    graph_hash = nx.weisfeiler_lehman_graph_hash(G)


    digest = hashlib.sha256(
        graph_hash.encode()
    ).digest()


    vector = np.frombuffer(
        digest,
        dtype=np.uint8
    )


    return vector[:size] / 255.0



# =====================================================
# Graph Feature Extraction
# =====================================================

def graph_features(G):

    features=[]


    # basic size

    features.append(
        G.number_of_nodes()
    )


    features.append(
        G.number_of_edges()
    )


    # density

    features.append(
        nx.density(G)
    )


    # degree

    if G.number_of_nodes()>0:

        avg_degree = (
            sum(dict(G.degree()).values())
            /
            G.number_of_nodes()
        )

    else:

        avg_degree=0


    features.append(
        avg_degree
    )


    # clustering coefficient

    if G.number_of_nodes()>1:

        clustering = nx.average_clustering(G)

    else:

        clustering=0


    features.append(
        clustering
    )


    # connected components

    features.append(
        nx.number_connected_components(G)
    )


    # shortest path

    if (
        G.number_of_nodes()>1
        and nx.is_connected(G)
    ):

        avg_path = nx.average_shortest_path_length(G)

    else:

        avg_path=0


    features.append(
        avg_path
    )


    # graph hash embedding

    features.extend(
        graph_hash_embedding(G)
    )


    return np.array(
        features,
        dtype=float
    )



# =====================================================
# Select cluster representative
# =====================================================

def select_medoid(indices, embeddings):


    vectors = embeddings[
        indices
    ]


    centroid = np.mean(
        vectors,
        axis=0
    )


    distances = np.linalg.norm(
        vectors-centroid,
        axis=1
    )


    return indices[
        np.argmin(distances)
    ]



# =====================================================
# Compression
# =====================================================

def compress_dataset(
        input_file,
        output_file,
        n_clusters
):


    data = load_json(
        input_file
    )


    graphs=[]

    features=[]


    print(
        "Building reasoning graphs..."
    )


    for item in tqdm(data):


        G = build_reasoning_graph(
            item["question"],
            item["answer"]
        )


        graphs.append(G)


        features.append(
            graph_features(G)
        )



    X=np.vstack(features)


    X=np.nan_to_num(
        X
    )


    X=normalize(
        X
    )



    # -------------------------
    # clustering
    # -------------------------

    print(
        "Clustering reasoning graphs..."
    )


    k=min(
        n_clusters,
        len(data)
    )


    model=KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )


    labels=model.fit_predict(
        X
    )



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
    # Select representatives
    # -------------------------

    compressed=[]


    cluster_information={}


    for cluster_id,indices in clusters.items():


        representative = select_medoid(
            indices,
            X
        )


        item=data[
            representative
        ].copy()


        item["cluster_id"]=cluster_id

        item["cluster_size"]=len(indices)


        item["reasoning_graph_nodes"]=(
            graphs[representative]
            .number_of_nodes()
        )


        item["reasoning_graph_edges"]=(
            graphs[representative]
            .number_of_edges()
        )


        compressed.append(
            item
        )


        cluster_information[
            cluster_id
        ]={

            "size":len(indices),

            "members":indices,

            "representative":
                representative
        }



    # -------------------------
    # Metrics
    # -------------------------

    original_size=len(data)

    compressed_size=len(compressed)


    compression_ratio=(

        1 -
        compressed_size/original_size

    )*100



    original_patterns=set()

    compressed_patterns=set()



    for G in graphs:

        original_patterns.add(
            nx.weisfeiler_lehman_graph_hash(G)
        )


    for item in compressed:

        G=build_reasoning_graph(
            item["question"],
            item["answer"]
        )


        compressed_patterns.add(
            nx.weisfeiler_lehman_graph_hash(G)
        )



    coverage=(

        len(compressed_patterns)
        /
        len(original_patterns)

    )*100



    result={


        "metrics":{

            "original_questions":
                original_size,


            "compressed_questions":
                compressed_size,


            "compression_ratio_percent":
                round(
                    compression_ratio,
                    2
                ),


            "number_of_clusters":
                len(clusters),


            "unique_original_reasoning_graphs":
                len(original_patterns),


            "unique_compressed_reasoning_graphs":
                len(compressed_patterns),


            "graph_coverage_percent":
                round(
                    coverage,
                    2
                )
        },


        "clusters":
            cluster_information,


        "compressed_dataset":
            compressed

    }



    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=4,
            ensure_ascii=False
        )



    print("\nCompression Complete")

    print(
        json.dumps(
            result["metrics"],
            indent=4
        )
    )



# =====================================================
# CLI
# =====================================================

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