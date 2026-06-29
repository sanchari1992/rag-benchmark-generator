import os

from benchmark_generation.strategies.base import BenchmarkStrategy

from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import StylingConfig, EvolutionConfig


class DeepEvalStrategy(BenchmarkStrategy):
    """
    Minimal stable DeepEval adapter using OpenAI.
    """

    def __init__(self):

        self.synthesizer = Synthesizer(
            styling_config=StylingConfig(),
            evolution_config=EvolutionConfig(
                num_evolutions=0  # keep it stable and fast
            )
        )

        self.last_metadata = {}

    @property
    def name(self):
        return "DeepEval"

    def generate_pair(self, llm, model, chunk):

        goldens = self.synthesizer.generate_goldens_from_contexts(
            contexts=[[chunk["text"]]]
        )

        golden = goldens[0]

        self.last_metadata = {
            "style": getattr(golden, "style", None),
            "evolution": getattr(golden, "evolution", None)
        }

        return golden.input, golden.expected_output

    def generate_question(self, llm, model, chunk):
        q, _ = self.generate_pair(llm, model, chunk)
        return q

    def generate_answer(self, llm, model, question, chunk):
        _, a = self.generate_pair(llm, model, chunk)
        return a