from all_metrics.ragas import (
    ragas_faithfulness,
    ragas_answer_relevancy,
    ragas_context_relevance,
)

from all_metrics.ares import (
    ares_context_relevance,
)

from all_metrics.datamorgana import (
    dm_placeholder,
)


class MetricRegistry:

    def __init__(self, judge):

        self.metrics = {

            "ragas_faithfulness":
                lambda item: ragas_faithfulness(item, judge),

            "ragas_answer_relevancy":
                lambda item: ragas_answer_relevancy(item, judge),

            "ragas_context_relevance":
                lambda item: ragas_context_relevance(item, judge),

            # placeholders
            "ares_context_relevance":
                lambda item: ares_context_relevance(item, judge),

            "datamorgana_placeholder":
                lambda item: dm_placeholder(item, judge),
        }

    def get(self, name):

        if name not in self.metrics:
            raise ValueError(f"Unknown metric {name}")

        return self.metrics[name]