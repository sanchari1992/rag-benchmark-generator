from prompts.ragas import (
    faithfulness_prompt,
    answer_relevancy_prompt,
    context_relevance_prompt,
)


def ragas_faithfulness(item, judge):

    prompt = faithfulness_prompt(
        context=item["retrieved_context"],
        answer=item["generated_answer"]
    )

    return judge.score(prompt)


def ragas_answer_relevancy(item, judge):

    prompt = answer_relevancy_prompt(
        question=item["question"],
        answer=item["generated_answer"]
    )

    return judge.score(prompt)


def ragas_context_relevance(item, judge):

    prompt = context_relevance_prompt(
        question=item["question"],
        context=item["retrieved_context"]
    )

    return judge.score(prompt)