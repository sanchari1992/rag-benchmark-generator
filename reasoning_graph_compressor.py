import json
import argparse
import numpy as np

from tqdm import tqdm

import spacy
import networkx as nx

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# NLP
# =====================================================

nlp = spacy.load("en_core_web_sm")



# =====================================================
# Load JSON
# =====================================================

def load_json(filename):

    for enc in [
        "utf-8",
        "utf-8-sig",
        "latin-1"
    ]:

        try:

            with open(
                filename,
                "r",
                encoding=enc
            ) as f:

                return json.load(f)

        except UnicodeDecodeError:
            continue

    raise Exception(
        "Cannot decode JSON"
    )



# =====================================================
# Build reasoning graph
# =====================================================

def build_reasoning_graph(question, answer):

    doc = nlp(
        question + " " + answer
    )

    G = nx.Graph()


    # entities

    for ent in doc.ents:

        node = ent.text.lower()

        G.add_node(
            node,
            type="entity"
        )


    # dependency relations

    for token in doc:

        if token.dep_ in [
            "nsubj",
            "dobj",
            "pobj",
            "attr",
            "compound"
        ]:

            h = token.head.text.lower()

            c = token.text.lower()


            G.add_node(
                h
            )

            G.add_node(
                c
            )


            G.add_edge(
                h,
                c
            )



    # fallback

    if G.number_of_nodes()==0:

        words = [
            t.lemma_.lower()
            for t in doc
            if t.is_alpha
            and not t.is_stop
        ]


        for w in words:
            G.add_node(w)


        for i in range(len(words)-1):

            G.add_edge(
                words[i],
                words[i+1]
            )


    return G



# =====================================================
# Graph embedding
# =====================================================

def graph_embedding(G, dim=32):

    features=[]


    features.append(
        G.number_of_nodes()
    )

    features.append(
        G.number_of_edges()
    )


    features.append(
        nx.density(G)
    )


    if G.number_of_nodes()>0:

        features.append(
            np.mean(
                list(
                    dict(
                        G.degree()
                    ).values()
                )
            )
        )

    else:
        features.append(0)



    if G.number_of_nodes()>1:

        features.append(
            nx.average_clustering(G)
        )

    else:

        features.append(0)



    # WL hash converted to vector

    h = nx.weisfeiler_lehman_graph_hash(G)


    nums=[]

    for c in h:

        nums.append(
            ord(c)/255
        )


    nums = nums[:dim]


    while len(nums)<dim:

        nums.append(0)



    features.extend(nums)


    return np.array(
        features,
        dtype=float
    )



# =====================================================
# Greedy coreset
# =====================================================

def greedy_coreset(
        embeddings,
        target_coverage=0.95
):


    n=len(embeddings)


    similarity = cosine_similarity(
        embeddings
    )


    selected=[]


    current_coverage=np.zeros(
        n
    )


    covered=0



    print(
        "Selecting reasoning graph coreset..."
    )


    while covered < target_coverage:


        best_gain=-1

        best_index=None


        for i in range(n):

            if i in selected:
                continue


            new_coverage=np.maximum(
                current_coverage,
                similarity[i]
            )


            gain=np.mean(
                new_coverage
            )


            if gain > best_gain:

                best_gain=gain

                best_index=i



        selected.append(
            best_index
        )


        current_coverage=np.maximum(
            current_coverage,
            similarity[best_index]
        )


        covered=np.mean(
            current_coverage
        )


        print(
            f"Selected {len(selected)} "
            f"Coverage={covered:.4f}"
        )


        if len(selected)==n:
            break



    return selected, covered



# =====================================================
# Main
# =====================================================

def compress(
        input_file,
        output_file,
        coverage
):


    data=load_json(
        input_file
    )


    graphs=[]

    embeddings=[]


    print(
        "Creating reasoning graphs..."
    )


    for item in tqdm(data):

        G=build_reasoning_graph(
            item["question"],
            item["answer"]
        )


        graphs.append(G)


        embeddings.append(
            graph_embedding(G)
        )



    embeddings=np.vstack(
        embeddings
    )


    embeddings=normalize(
        embeddings
    )



    selected,final_coverage = greedy_coreset(
        embeddings,
        coverage
    )



    compressed=[]


    for idx in selected:

        item=data[idx].copy()

        item["coreset_selected"]=True

        item["original_index"]=idx


        compressed.append(
            item
        )



    result={

        "metrics":{

            "original_questions":
                len(data),

            "compressed_questions":
                len(compressed),

            "compression_ratio":
                round(
                    1-len(compressed)/len(data),
                    4
                ),

            "target_coverage":
                coverage,

            "achieved_coverage":
                round(
                    final_coverage,
                    4
                )

        },


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


    print("\nFinished")



# =====================================================
# CLI
# =====================================================

if __name__=="__main__":

    parser=argparse.ArgumentParser(
        description=
        "Greedy Reasoning Graph Coreset Selection"
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
        "--coverage",
        type=float,
        default=0.95
    )


    args=parser.parse_args()


    compress(
        args.input,
        args.output,
        args.coverage
    )