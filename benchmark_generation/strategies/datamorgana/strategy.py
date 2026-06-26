import os

from benchmark_generation.strategies.base import BenchmarkStrategy

from benchmark_generation.prompts.datamorgana.prompts import (
    DATAMORGANA_QUESTION_PROMPT,
    DATAMORGANA_ANSWER_PROMPT
)

from benchmark_generation.core.context_utils import prepare_context

from benchmark_generation.utils.datamorgana_utils import (
    load_datamorgana_config,
    sample_user_categories,
    sample_question_categories
)


class DataMorganaStrategy(BenchmarkStrategy):

    def __init__(self):

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "datamorgana",
            "datamorgana_config.json"
        )

        self.config = load_datamorgana_config(config_path)

        self.last_metadata = {}

    @property
    def name(self):

        return "DataMorgana"

    def generate_question(self, llm, model, chunk):

        user_categories = sample_user_categories(
            self.config
        )

        question_categories = sample_question_categories(
            self.config
        )

        user_descriptions = "\n".join(
            [
                f"- {c['description']}"
                for c in user_categories
            ]
        )

        question_descriptions = "\n".join(
            [
                f"- {c['description']}"
                for c in question_categories
            ]
        )

        prompt = DATAMORGANA_QUESTION_PROMPT.format(
            context=prepare_context(chunk["text"]),
            user_characteristics=user_descriptions,
            question_characteristics=question_descriptions
        )

        messages = [
            {
                "role": "system",
                "content": "You are a benchmark generation system."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        question = llm.generate(
            model,
            messages
        )

        self.last_metadata = {
            "user_categories":
                [c["name"] for c in user_categories],

            "question_categories":
                [c["name"] for c in question_categories]
        }

        return question

    def generate_answer(
        self,
        llm,
        model,
        question,
        chunk
    ):

        prompt = DATAMORGANA_ANSWER_PROMPT.format(
            question=question,
            context=chunk["text"]
        )

        messages = [
            {
                "role": "system",
                "content": "You are a QA system."
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