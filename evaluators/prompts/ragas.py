def faithfulness_prompt(context, answer):

    return f"""
You are evaluating Faithfulness.

Faithfulness measures whether the answer is fully supported by the provided context.

Any statement not supported by the context should reduce the score.

Return JSON:

{{
    "score": float,
    "reason": ""
}}

Context

{context}

Answer

{answer}
"""

def answer_relevancy_prompt(question, answer):

    return f"""
You are evaluating Answer Relevancy.

Question

{question}

Answer

{answer}

Score from 0-10.

Return JSON.

{{
    "score": float,
    "reason": ""
}}
"""

def context_relevance_prompt(question, context):

    return f"""
Determine how relevant the retrieved context is for answering the question.

Question

{question}

Retrieved Context

{context}

Return

{{
    "score": float,
    "reason":""
}}
"""