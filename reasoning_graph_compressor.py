import json
import argparse
import numpy as np

from tqdm import tqdm

import spacy
import networkx as nx

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# NLP MODEL
# ==========================================================

nlp = spacy.load("en_core_web_sm")



# ==========================================================
# JSON LOADER
# ==========================================================

def load_json(path):

    for enc in [
        "utf-8",
        "utf-8-sig",
        "latin-1"
    ]:

        try:

            with open(
                path,
                "r",
                encoding=enc
            ) as f:

                return json.load(f)

        except UnicodeDecodeError:
            pass

    raise Exception(
        "Could not decode JSON file"
    )



# ==========================================================
# BUILD REASONING GRAPH
# ==========================================================

def build_reasoning_graph(question, answer):

    text = question + " " + answer

    doc = nlp(text)


    G = nx.Graph()



    # -------------------------
    # Entity nodes
    # -------------------------

    for ent in doc.ents:

        node = ent.text.lower().strip()

        G.add_node(
            node,
            type="entity"
        )



    # -------------------------
    # Dependency edges
    # -------------------------

    for token in doc:


        if token.dep_ in [

            "nsubj",
            "dobj",
            "pobj",
            "attr",
            "compound",
            "prep"

        ]:


            head = token.head.text.lower()

            child = token.text.lower()


            G.add_node(
                head,
                type="concept"
            )


            G.add_node(
                child,
                type="concept"
            )


            G.add_edge(
                head,
                child
            )



    # -------------------------
    # Fallback graph
    # -------------------------

    if G.number_of_nodes() == 0:


        words = [

            token.lemma_.lower()

            for token in doc

            if token.is_alpha
            and not token.is_stop

        ]


        for w in words:

            G.add_node(w)



        for i in range(
            len(words)-1
        ):

            G.add_edge(
                words[i],
                words[i+1]
            )


    return G



# ==========================================================
# GRAPH FEATURE EXTRACTION
# ==========================================================

def graph_features(G):


    features=[]


    # Number of nodes

    n_nodes = G.number_of_nodes()

    features.append(
        n_nodes
    )


    # Number of edges

    n_edges = G.number_of_edges()

    features.append(
        n_edges
    )


    # Density

    features.append(
        nx.density(G)
    )



    # Average degree

    if n_nodes > 0:

        avg_degree = (

            sum(
                dict(G.degree()).values()
            )
            /
            n_nodes

        )

    else:

        avg_degree = 0


    features.append(
        avg_degree
    )



    # Clustering coefficient

    if n_nodes > 1:

        clustering = (
            nx.average_clustering(G)
        )

    else:

        clustering = 0


    features.append(
        clustering
    )



    # Connected components

    features.append(
        nx.number_connected_components(G)
    )



    # Path features

    if nx.is_connected(G) and n_nodes > 1:


        features.append(
            nx.average_shortest_path_length(G)
        )


        features.append(
            nx.diameter(G)
        )


    else:

        features.append(0)

        features.append(0)



    # Entity/concept ratio

    entity_count = sum(

        1

        for _,d in G.nodes(data=True)

        if d.get("type")=="entity"

    )


    features.append(
        entity_count
    )


    features.append(
        n_nodes - entity_count
    )


    return np.array(
        features,
        dtype=float
    )



# ==========================================================
# GRAPH PATTERN HASH
# ==========================================================

def graph_hash(G):

    return nx.weisfeiler_lehman_graph_hash(
        G
    )



# ==========================================================
# GREEDY CORESET SELECTION
# ==========================================================

def greedy_selection(
        embeddings,
        target_coverage
):


    n = len(embeddings)



    similarity = cosine_similarity(
        embeddings
    )



    selected=[]



    coverage_vector = np.zeros(n)



    achieved = 0



    # avoid extreme compression

    minimum_size = max(
        5,
        int(0.05*n)
    )



    print(
        "\nGreedy selection started"
    )



    while (

        achieved < target_coverage

        or

        len(selected) < minimum_size

    ):



        best_index = None

        best_score = -1



        for i in range(n):


            if i in selected:

                continue



            candidate = similarity[i]



            new_coverage = np.maximum(

                coverage_vector,

                candidate

            )



            coverage_gain = (

                np.mean(new_coverage)

                -

                np.mean(coverage_vector)

            )



            # diversity bonus

            if len(selected)>0:

                diversity = (

                    1 -

                    max(
                        similarity[i][j]
                        for j in selected
                    )

                )

            else:

                diversity = 1



            score = (

                coverage_gain

                +

                0.2*diversity

            )



            if score > best_score:

                best_score = score

                best_index = i



        selected.append(
            best_index
        )


        coverage_vector = np.maximum(

            coverage_vector,

            similarity[best_index]

        )


        achieved = np.mean(
            coverage_vector
        )


        # print(
        #     f"Selected={len(selected)} "
        #     f"Coverage={achieved:.4f}"
        # )



        if len(selected)==n:

            break



    return selected, achieved



# ==========================================================
# MAIN
# ==========================================================

def compress_dataset(
        input_file,
        output_file,
        report_file,
        target_coverage
):


    data = load_json(
        input_file
    )



    graphs=[]

    embeddings=[]



    print(
        "Building reasoning graphs..."
    )



    for item in tqdm(data):


        G = build_reasoning_graph(

            item["question"],

            item["answer"]

        )


        graphs.append(G)



        embeddings.append(

            graph_features(G)

        )



    embeddings = np.vstack(
        embeddings
    )


    embeddings = normalize(
        embeddings
    )



    selected, achieved = greedy_selection(

        embeddings,

        target_coverage

    )



    # ======================================================
    # Save clean benchmark
    # ======================================================

    compressed = [

        data[i]

        for i in selected

    ]



    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:


        json.dump(

            compressed,

            f,

            indent=4,

            ensure_ascii=False

        )



    # ======================================================
    # Metrics
    # ======================================================

    original_hashes=set()

    compressed_hashes=set()



    for G in graphs:

        original_hashes.add(
            graph_hash(G)
        )



    for i in selected:

        compressed_hashes.add(

            graph_hash(
                graphs[i]
            )

        )



    original=len(data)

    compressed=len(compressed)



    report={


        "algorithm":

        "Greedy Reasoning Graph Coreset Selection",



        "parameters":{

            "target_coverage":

            target_coverage

        },


        "metrics":{


            "original_questions":

            original,


            "compressed_questions":

            compressed,


            "compression_ratio_percent":

            round(

                (1-compressed/original)*100,

                2

            ),


            "retention_ratio_percent":

            round(

                compressed/original*100,

                2

            ),


            "achieved_similarity_coverage":

            round(

                achieved,

                4

            ),



            "unique_original_reasoning_graphs":

            len(original_hashes),



            "unique_compressed_reasoning_graphs":

            len(compressed_hashes),



            "exact_graph_pattern_coverage_percent":

            round(

                len(compressed_hashes)
                /
                len(original_hashes)
                *
                100,

                2

            )

        },


        "selected_indices":

        selected

    }



    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as f:


        json.dump(

            report,

            f,

            indent=4

        )



    print(
        "\nCompression Complete"
    )


    print(
        json.dumps(
            report["metrics"],
            indent=4
        )
    )



# ==========================================================
# CLI
# ==========================================================

if __name__=="__main__":


    parser=argparse.ArgumentParser()


    parser.add_argument(
        "--input",
        required=True
    )


    parser.add_argument(
        "--output",
        required=True
    )


    parser.add_argument(
        "--report",
        required=True
    )


    parser.add_argument(
        "--coverage",
        type=float,
        default=0.95
    )


    args=parser.parse_args()



    compress_dataset(

        args.input,

        args.output,

        args.report,

        args.coverage

    )