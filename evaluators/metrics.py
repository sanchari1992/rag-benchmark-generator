from collections import defaultdict

def compute_metrics(results):
    """
    results: list of dicts (detailed results)
    """

    total = len(results)
    correct = sum(1 for r in results if r["verdict"] == "correct")
    incorrect = total - correct

    avg_score = sum(r.get("score", 0) for r in results) / max(total, 1)

    exact = sum(
        1 for r in results
        if r["expected"].strip().lower() == r["predicted"].strip().lower()
    )

    empty = sum(1 for r in results if not r["predicted"].strip())

    # breakdowns
    by_repo = defaultdict(lambda: {"total": 0, "correct": 0})
    by_strategy = defaultdict(lambda: {"total": 0, "correct": 0})
    by_model = defaultdict(lambda: {"total": 0, "correct": 0})

    for r in results:
        repo = r.get("repo_name", "unknown")
        strat = r.get("strategy", "unknown")
        model = r.get("generator_model", "unknown")

        for k, v in [(by_repo, repo), (by_strategy, strat), (by_model, model)]:
            k[v]["total"] += 1
            if r["verdict"] == "correct":
                k[v]["correct"] += 1

    def to_acc(d):
        return {
            k: {
                "total": v["total"],
                "correct": v["correct"],
                "accuracy": (v["correct"] / v["total"]) * 100 if v["total"] else 0
            }
            for k, v in d.items()
        }

    return {
        "summary": {
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": (correct / total) * 100 if total else 0,
            "avg_score": avg_score,
            "exact_matches": exact,
            "empty_answers": empty
        },
        "by_repo": to_acc(by_repo),
        "by_strategy": to_acc(by_strategy),
        "by_model": to_acc(by_model)
    }