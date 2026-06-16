from benchmark_generation.strategies.base import BenchmarkStrategy
from benchmark_generation.prompts.ragas_prompts import (
    RAGAS_QUESTION_PROMPT,
    RAGAS_ANSWER_PROMPT
)
from benchmark_generation.core.context_utils import prepare_context


class RAGASStrategy(BenchmarkStrategy):
    @property
    def name(self):
        return "RAGAS"

    def generate_question(self, llm, model, chunk):

        prompt = RAGAS_QUESTION_PROMPT.format(
            context = prepare_context(chunk["text"])
        )

        messages = [
            {"role": "system", "content": "You are a dataset generator."},
            {"role": "user", "content": prompt}
        ]

        return llm.generate(model, messages)

    def generate_answer(self, llm, model, question, chunk):

        prompt = RAGAS_ANSWER_PROMPT.format(
            context=chunk["text"],
            question=question
        )

        messages = [
            {"role": "system", "content": "You are a QA system."},
            {"role": "user", "content": prompt}
        ]

        return llm.generate(model, messages)