from benchmark_generation.strategies.base import (
    BenchmarkStrategy
)

from benchmark_generation.core.context_utils import (
    prepare_context
)

from benchmark_generation.prompts.ares.prompts_original import (
    ARES_QUERY_PROMPT,
    ARES_ANSWER_PROMPT,
    ARES_CONTRADICTORY_ANSWER_PROMPT
)

from benchmark_generation.prompts.ares.fewshot_examples import (
    ARES_QUERY_EXAMPLES,
    ARES_ANSWER_EXAMPLES
)


class ARESStrategy(BenchmarkStrategy):

    @property
    def name(self):
        return "ARES"

    def _build_query_examples(self):

        text = ""

        for i, ex in enumerate(
            ARES_QUERY_EXAMPLES,
            start=1
        ):

            text += (
                f"Example #{i}\n"
                f"Document: {ex['document']}\n"
                f"Query: {ex['query']}\n\n"
            )

        return text

    def _build_answer_examples(self):

        text = ""

        for i, ex in enumerate(
            ARES_ANSWER_EXAMPLES,
            start=1
        ):

            text += (
                f"Example #{i}\n"
                f"Query: {ex['query']}\n"
                f"Document: {ex['document']}\n"
                f"Answer: {ex['answer']}\n\n"
            )

        return text

    def generate_question(
        self,
        llm,
        model,
        chunk
    ):

        prompt = ARES_QUERY_PROMPT.format(
            few_shot_examples=
                self._build_query_examples(),
            document=
                prepare_context(
                    chunk["text"]
                )
        )

        messages = [
            {
                "role": "system",
                "content":
                    "You generate synthetic search queries."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        return llm.generate(
            model,
            messages
        )

    def generate_answer(
        self,
        llm,
        model,
        question,
        chunk
    ):

        prompt = ARES_ANSWER_PROMPT.format(
            few_shot_examples=
                self._build_answer_examples(),

            query=question,

            document=
                prepare_context(
                    chunk["text"]
                )
        )

        messages = [
            {
                "role": "system",
                "content":
                    "You answer questions from documents."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        return llm.generate(
            model,
            messages
        )

    def generate_contradictory_answer(
        self,
        llm,
        model,
        question,
        chunk
    ):

        prompt = (
            ARES_CONTRADICTORY_ANSWER_PROMPT
            .format(
                query=question,
                document=
                prepare_context(
                    chunk["text"]
                )
            )
        )

        messages = [
            {
                "role": "system",
                "content":
                    "Generate contradictory answers."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        return llm.generate(
            model,
            messages
        )