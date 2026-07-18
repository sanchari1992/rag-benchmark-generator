#!/usr/bin/env python3
"""
greedy_reasoning_graph_coreset_compressor.py

Provable coreset compression for RAG question benchmarks.

METHOD
------
1. Embed each question's text (TF-IDF by default; sentence-transformers if
   installed and requested) into a vector space.
2. Build a cosine-similarity "reasoning graph" over questions: a sparse
   k-NN graph for large datasets, a dense full graph for small ones.
3. Select a coreset S by greedily maximizing the facility-location coverage
   objective:

        F(S) = (1/N) * sum_i  max_{j in S} sim(i, j)

   F is monotone submodular, so greedy selection is guaranteed to reach at
   least (1 - 1/e) ~= 0.632 of the optimal coverage achievable by ANY
   subset of the same size (Nemhauser, Wolsey & Fisher, 1978). We use the
   standard *lazy greedy* / accelerated greedy acceleration (Minoux, 1978)
   so this stays fast for large N.
4. Stop once the target --coverage is reached (or a target --ratio size is
   reached, whichever the user asked for). This is the size-control knob:
   coverage=1.0 essentially forces near-full coverage (large coreset),
   coverage=0.5 stops much earlier (small coreset).
5. Every ORIGINAL question is assigned to its nearest selected representative
   (a Voronoi partition of the similarity graph). The size of that partition
   becomes the representative's `coreset_weight` -- this is what lets you
   do unbiased reweighted evaluation on the compressed set instead of just
   throwing away information about how "common" a question pattern was.

USAGE
-----
    python greedy_reasoning_graph_coreset.py \\
        --input abc.json \\
        --output compressed_benchmark.json \\
        --report compression_report.json \\
        --coverage 0.95

    # or force an exact size ratio instead of / in addition to coverage:
    python greedy_reasoning_graph_coreset.py \\
        --input abc.json --output out.json --report report.json \\
        --ratio 0.3
"""

import argparse
import json
import sys
import time
import heapq
import math
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------

def embed_questions(questions, method="tfidf", sbert_model="all-MiniLM-L6-v2"):
    """Return an (N, d) matrix of unit-norm embeddings, one row per question."""
    if method == "sbert":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print(
                "[warn] sentence-transformers not installed; "
                "falling back to --embedding tfidf",
                file=sys.stderr,
            )
            method = "tfidf"
        else:
            model = SentenceTransformer(sbert_model)
            vecs = model.encode(questions, show_progress_bar=False)
            vecs = normalize(np.asarray(vecs, dtype=np.float32))
            return vecs

    # TF-IDF fallback / default: robust, dependency-light, deterministic.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(questions)  # sparse, rows already scaled
    X = normalize(X)  # unit-norm rows -> dot product == cosine similarity
    return X


# --------------------------------------------------------------------------
# Similarity graph
# --------------------------------------------------------------------------

def build_similarity_graph(X, knn, dense_threshold):
    """
    Build a cosine-similarity graph over embeddings X.

    Returns either:
      - a dense (N, N) numpy array (small N), or
      - a sparse CSR (N, N) matrix restricted to each point's top-`knn`
        neighbors, symmetrized (large N).
    """
    n = X.shape[0]

    if n <= dense_threshold:
        if sparse.issparse(X):
            X = X.toarray()
        sim = X @ X.T
        sim = np.clip(sim, 0.0, 1.0)
        np.fill_diagonal(sim, 1.0)
        return sim, "dense"

    # Sparse k-NN graph for scalability on large benchmarks.
    knn = min(knn, n - 1)
    nn = NearestNeighbors(n_neighbors=knn + 1, metric="cosine", algorithm="brute")
    nn.fit(X)
    dist, idx = nn.kneighbors(X)  # includes self at column 0
    sim_vals = 1.0 - dist
    sim_vals = np.clip(sim_vals, 0.0, 1.0)

    rows = np.repeat(np.arange(n), knn + 1)
    cols = idx.ravel()
    vals = sim_vals.ravel()
    sim = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    # symmetrize: keep the max of sim(i,j), sim(j,i)
    sim = sim.maximum(sim.T)
    sim.setdiag(1.0)
    sim = sim.tocsr()
    return sim, "sparse"


def get_column_dense(sim, kind, j):
    """Return column j of the similarity graph as a dense 1-D numpy array."""
    if kind == "dense":
        return sim[:, j]
    col = sim.getcol(j)
    return np.asarray(col.todense()).ravel()


def column_sum_all(sim, kind, n):
    """Initial marginal gain of every candidate before any point is selected."""
    if kind == "dense":
        return sim.sum(axis=0)
    return np.asarray(sim.sum(axis=0)).ravel()


# --------------------------------------------------------------------------
# Lazy greedy submodular coverage maximization
# --------------------------------------------------------------------------

def greedy_facility_location(
    sim, kind, n, target_coverage=None, target_size=None,
    min_size=1, max_size=None, verbose=True,
):
    """
    Lazy greedy maximization of F(S) = (1/n) * sum_i max_{j in S} sim(i,j).

    Stops as soon as EITHER target_coverage OR target_size is reached
    (whichever triggers first), subject to min_size/max_size bounds.

    Returns:
        selected   : list[int]      -- indices in selection order
        coverage   : float          -- final F(S)
        current_max: np.ndarray[n]  -- best similarity of each point to S
    """
    max_size = max_size or n
    current_max = np.zeros(n, dtype=np.float64)
    remaining = np.ones(n, dtype=bool)

    init_gain = column_sum_all(sim, kind, n)
    heap = [(-float(init_gain[j]), j, 0) for j in range(n)]
    heapq.heapify(heap)

    stamp = np.zeros(n, dtype=np.int64)  # iteration each candidate's gain reflects
    iteration = 0
    selected = []
    coverage = 0.0

    t0 = time.time()
    while heap and len(selected) < max_size:
        neg_gain, j, ver = heapq.heappop(heap)
        if not remaining[j]:
            continue
        if ver != iteration:
            col = get_column_dense(sim, kind, j)
            gain = float(np.maximum(0.0, col - current_max).sum())
            heapq.heappush(heap, (-gain, j, iteration))
            continue

        # accept j: its gain was computed against the CURRENT current_max
        col = get_column_dense(sim, kind, j)
        selected.append(j)
        remaining[j] = False
        current_max = np.maximum(current_max, col)
        coverage = float(current_max.mean())
        iteration += 1

        if verbose and (len(selected) % max(1, n // 20) == 0 or len(selected) <= 5):
            print(
                f"[greedy] selected={len(selected):5d}  coverage={coverage:.4f}"
                f"  elapsed={time.time()-t0:.1f}s",
                file=sys.stderr,
            )

        if len(selected) >= min_size:
            if target_size is not None and len(selected) >= target_size:
                break
            if target_coverage is not None and coverage >= target_coverage:
                break

    return selected, coverage, current_max


def greedy_k_center(sim, kind, n, target_size, min_size=1, verbose=True):
    """
    Gonzalez farthest-point greedy k-center: alternative, radius-based
    selection with a classic 2-approximation guarantee on the covering
    radius (Gonzalez, 1985). Useful if you prefer a "no point is left more
    than 2x-optimal-radius away from its representative" guarantee instead
    of the average-coverage guarantee of facility location.
    """
    rng_state = np.random.RandomState(0)
    first = int(rng_state.randint(n))
    selected = [first]
    min_dist_to_S = 1.0 - get_column_dense(sim, kind, first)  # cosine distance

    while len(selected) < max(target_size, min_size):
        j = int(np.argmax(min_dist_to_S))
        selected.append(j)
        col = get_column_dense(sim, kind, j)
        min_dist_to_S = np.minimum(min_dist_to_S, 1.0 - col)
        if verbose and len(selected) % max(1, n // 20) == 0:
            radius = float(min_dist_to_S.max())
            print(
                f"[k-center] selected={len(selected):5d}  covering_radius={radius:.4f}",
                file=sys.stderr,
            )
    coverage = float((1.0 - min_dist_to_S).mean())
    return selected, coverage, (1.0 - min_dist_to_S)


# --------------------------------------------------------------------------
# Voronoi reweighting
# --------------------------------------------------------------------------

def assign_weights(sim, kind, n, selected):
    """
    Assign every original point to its nearest selected representative and
    return per-representative weights (cluster sizes) -- the multiplier
    needed for unbiased reweighted evaluation on the compressed set.
    """
    S = len(selected)
    best_sim = np.full(n, -1.0)
    best_rep = np.full(n, -1, dtype=np.int64)

    for rank, j in enumerate(selected):
        col = get_column_dense(sim, kind, j)
        better = col > best_sim
        best_sim[better] = col[better]
        best_rep[better] = rank

    weights = np.bincount(best_rep[best_rep >= 0], minlength=S)
    assignment_sim = best_sim
    return weights, assignment_sim, best_rep


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Provable submodular coreset compression for RAG question benchmarks."
    )
    ap.add_argument("--input", required=True, help="Input JSON file (list of question objects).")
    ap.add_argument("--output", required=True, help="Output compressed JSON file.")
    ap.add_argument("--report", default=None, help="Optional JSON report of the compression run.")
    ap.add_argument("--coverage", type=float, default=None,
                     help="Target coverage in (0,1]. Stop once mean nearest-representative "
                          "similarity reaches this value. e.g. 0.95")
    ap.add_argument("--ratio", type=float, default=None,
                     help="Target compression ratio in (0,1]. Alternative/joint size control: "
                          "1.0 = full dataset, 0.5 = half size, etc.")
    ap.add_argument("--min-ratio", type=float, default=0.01,
                     help="Lower bound on compressed size as a fraction of N (safety floor).")
    ap.add_argument("--max-ratio", type=float, default=1.0,
                     help="Upper bound on compressed size as a fraction of N.")
    ap.add_argument("--method", choices=["facility", "kcenter"], default="facility",
                     help="facility = submodular coverage greedy (1-1/e guarantee, default). "
                          "kcenter = Gonzalez farthest-point greedy (2-approx radius guarantee).")
    ap.add_argument("--embedding", choices=["tfidf", "sbert"], default="tfidf")
    ap.add_argument("--sbert-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--knn", type=int, default=50,
                     help="Neighbors per node in the sparse similarity graph (large datasets only).")
    ap.add_argument("--dense-threshold", type=int, default=3000,
                     help="Use a full dense similarity matrix below this many questions; "
                          "use a sparse k-NN graph above it.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.coverage is None and args.ratio is None:
        ap.error("Provide at least one of --coverage or --ratio to control output size.")
    if args.coverage is not None and not (0.0 < args.coverage <= 1.0):
        ap.error("--coverage must be in (0, 1].")
    if args.ratio is not None and not (0.0 < args.ratio <= 1.0):
        ap.error("--ratio must be in (0, 1].")

    verbose = not args.quiet

    # ---- load ----
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) == 0:
        ap.error("Input JSON must be a non-empty list of question objects.")

    n = len(data)
    questions = []
    for i, item in enumerate(data):
        q = item.get("question", "")
        if not isinstance(q, str) or not q.strip():
            print(f"[warn] item id={item.get('id', i)} has empty/missing 'question'; using placeholder.",
                  file=sys.stderr)
            q = " "
        questions.append(q)

    min_size = max(1, math.ceil(args.min_ratio * n))
    max_size = max(min_size, math.floor(args.max_ratio * n))
    target_size = math.ceil(args.ratio * n) if args.ratio is not None else None
    if target_size is not None:
        target_size = min(max(target_size, min_size), max_size)

    if verbose:
        print(f"[info] loaded {n} questions from {args.input}", file=sys.stderr)
        print(f"[info] embedding method: {args.embedding}", file=sys.stderr)

    # ---- embed + build graph ----
    t0 = time.time()
    X = embed_questions(questions, method=args.embedding, sbert_model=args.sbert_model)
    sim, kind = build_similarity_graph(X, knn=args.knn, dense_threshold=args.dense_threshold)
    if verbose:
        print(f"[info] similarity graph ({kind}) built in {time.time()-t0:.1f}s", file=sys.stderr)

    # ---- select coreset ----
    t0 = time.time()
    if args.method == "facility":
        selected, coverage, current_max = greedy_facility_location(
            sim, kind, n,
            target_coverage=args.coverage,
            target_size=target_size,
            min_size=min_size,
            max_size=max_size,
            verbose=verbose,
        )
        guarantee = (
            "Greedy facility-location coverage maximization is a (1 - 1/e) ~= 0.632 "
            "approximation to the optimal coverage achievable by any subset of the same "
            "size, since F(S) is monotone submodular (Nemhauser, Wolsey & Fisher, 1978)."
        )
    else:
        eff_target = target_size or max(min_size, int(round((args.coverage or 0.9) * n)))
        selected, coverage, current_max = greedy_k_center(
            sim, kind, n, target_size=eff_target, min_size=min_size, verbose=verbose,
        )
        guarantee = (
            "Gonzalez farthest-point greedy k-center achieves a covering radius at most "
            "2x the optimal radius achievable by any subset of the same size (Gonzalez, 1985)."
        )

    if verbose:
        print(f"[info] selection done in {time.time()-t0:.1f}s: "
              f"{len(selected)}/{n} questions, coverage={coverage:.4f}", file=sys.stderr)

    # ---- reweight (Voronoi assignment over the FULL original set) ----
    weights, assignment_sim, best_rep = assign_weights(sim, kind, n, selected)
    total_weight = int(weights.sum())

    # ---- build output ----
    compressed = []
    for rank, idx in enumerate(selected):
        item = dict(data[idx])  # shallow copy, preserves all original fields
        item["coreset_weight"] = int(weights[rank])
        item["coreset_normalized_weight"] = float(weights[rank]) / total_weight if total_weight else 0.0
        item["coreset_selection_rank"] = rank + 1  # 1 = first/most-covering pick
        item["compression_stage"] = "coreset_v1"
        compressed.append(item)

    Path(args.output).write_text(json.dumps(compressed, indent=2, ensure_ascii=False), encoding="utf-8")
    if verbose:
        print(f"[info] wrote {len(compressed)} questions to {args.output}", file=sys.stderr)

    # ---- report ----
    if args.report:
        weight_arr = np.array(weights, dtype=np.float64)
        report = {
            "input_file": args.input,
            "output_file": args.output,
            "original_count": n,
            "compressed_count": len(selected),
            "compression_ratio": len(selected) / n,
            "method": args.method,
            "embedding": args.embedding,
            "similarity_graph": kind,
            "target_coverage": args.coverage,
            "target_ratio": args.ratio,
            "achieved_coverage": coverage,
            "achieved_mean_assignment_similarity": float(assignment_sim.mean()),
            "achieved_min_assignment_similarity": float(assignment_sim.min()),
            "weight_distribution": {
                "min": float(weight_arr.min()) if len(weight_arr) else 0,
                "max": float(weight_arr.max()) if len(weight_arr) else 0,
                "mean": float(weight_arr.mean()) if len(weight_arr) else 0,
                "std": float(weight_arr.std()) if len(weight_arr) else 0,
            },
            "theoretical_guarantee": guarantee,
        }
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        if verbose:
            print(f"[info] wrote report to {args.report}", file=sys.stderr)
            print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()