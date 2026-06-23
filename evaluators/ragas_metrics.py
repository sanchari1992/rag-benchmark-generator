from collections import defaultdict


def compute_ragas_metrics(results):
    """
    results:
        list of dictionaries containing

        {
            ...
            "faithfulness": 0.82
        }
    """

    total = len(results)

    if total == 0:
        return {
            "summary": {
                "total": 0,
                "average_faithfulness": 0
            },
            "by_repo": {},
            "by_strategy": {},
            "by_model": {}
        }

    avg_faithfulness = (
        sum(r.get("faithfulness", 0) for r in results)
        / total
    )

    by_repo = defaultdict(list)
    by_strategy = defaultdict(list)
    by_model = defaultdict(list)

    for r in results:

        by_repo[r.get("repo_name", "unknown")].append(
            r.get("faithfulness", 0)
        )

        by_strategy[r.get("strategy", "unknown")].append(
            r.get("faithfulness", 0)
        )

        by_model[r.get("generator_model", "unknown")].append(
            r.get("faithfulness", 0)
        )

    def summarize(d):
        return {
            k: {
                "total": len(v),
                "average_faithfulness": sum(v) / len(v)
            }
            for k, v in d.items()
        }

    return {
        "summary": {
            "total": total,
            "average_faithfulness": avg_faithfulness
        },
        "by_repo": summarize(by_repo),
        "by_strategy": summarize(by_strategy),
        "by_model": summarize(by_model)
    }