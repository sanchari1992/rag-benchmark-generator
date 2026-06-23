import json
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import jensenshannon
from collections import Counter
import re


# =========================
# LOAD YOUR DATASET
# =========================

def load_your_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# LOAD WIKIEVAL
# =========================

def load_wikieval():
    ds = load_dataset("vibrantlabsai/WikiEval")
    return ds["train"]


# =========================
# BASIC STATS
# =========================

def get_lengths(texts):
    return np.array([len(t.split()) for t in texts])


# =========================
# ENTITY HEURISTIC
# =========================

def entity_density(text):
    # simple proxy: capitalized words not at sentence start
    return len(re.findall(r"\b[A-Z][a-z]+\b", text)) / max(len(text.split()), 1)


# =========================
# EMBEDDING MODEL
# =========================

model = SentenceTransformer("all-mpnet-base-v2")


# =========================
# EMBEDDING SIMILARITY
# =========================

def embedding_similarity(list1, list2):

    emb1 = model.encode(list1, convert_to_numpy=True)
    emb2 = model.encode(list2, convert_to_numpy=True)

    # cosine similarity matrix
    sim_matrix = emb1 @ emb2.T

    # for each item in dataset A, find best match in B
    max_sim = np.max(sim_matrix, axis=1)

    return {
        "mean_similarity": float(np.mean(max_sim)),
        "std_similarity": float(np.std(max_sim))
    }


# =========================
# DISTRIBUTION SIMILARITY
# =========================

def distribution_similarity(a, b, bins=20):

    hist_a, _ = np.histogram(a, bins=bins, density=True)
    hist_b, _ = np.histogram(b, bins=bins, density=True)

    # Jensen-Shannon divergence
    js = jensenshannon(hist_a + 1e-8, hist_b + 1e-8)

    return float(js)


# =========================
# EXTRACT FIELDS
# =========================

def extract_wikieval(ds):

    questions = [x["question"] for x in ds]
    answers = [x["answer"] if x["answer"] else "" for x in ds]

    return {
        "questions": questions,
        "answers": answers
    }


def extract_yours(data):

    questions = [x["question"] for x in data]
    answers = [x["answer"] for x in data]

    contexts = [x["source_text"] for x in data]

    return {
        "questions": questions,
        "answers": answers,
        "contexts": contexts
    }


# =========================
# MAIN EVALUATION
# =========================

def evaluate(wikieval, yours):

    w = extract_wikieval(wikieval)
    y = extract_yours(yours)

    print("\n--- LENGTH STATS ---")

    w_q_len = get_lengths(w["questions"])
    y_q_len = get_lengths(y["questions"])

    print("Question length JS divergence:",
          distribution_similarity(w_q_len, y_q_len))

    print("\n--- ANSWER LENGTH ---")

    w_a_len = get_lengths(w["answers"])
    y_a_len = get_lengths(y["answers"])

    print("Answer length JS divergence:",
          distribution_similarity(w_a_len, y_a_len))

    print("\n--- CONTEXT LENGTH (your dataset only) ---")

    y_c_len = get_lengths(y["contexts"])

    print("Context avg:", np.mean(y_c_len))


    print("\n--- ENTITY DENSITY ---")

    w_ent = [entity_density(q) for q in w["questions"]]
    y_ent = [entity_density(q) for q in y["questions"]]

    print("Entity density JS divergence:",
          distribution_similarity(w_ent, y_ent))


    print("\n--- EMBEDDING SIMILARITY ---")

    emb_result = embedding_similarity(
        w["questions"][:200],
        y["questions"][:200]
    )

    print(emb_result)


    print("\n--- FINAL SCORE ---")

    final_score = (
        emb_result["mean_similarity"] * 0.5 +
        (1 - distribution_similarity(w_q_len, y_q_len)) * 0.2 +
        (1 - distribution_similarity(w_ent, y_ent)) * 0.2 +
        (1 - distribution_similarity(w_a_len, y_a_len)) * 0.1
    )

    print("Dataset similarity score:", final_score)


# =========================
# RUN
# =========================

if __name__ == "__main__":

    wikieval = load_wikieval()

    your_data = load_your_dataset(
        "../../exports/ragas/wiki/run_20260622_112350/benchmark.json"
    )

    evaluate(wikieval, your_data)