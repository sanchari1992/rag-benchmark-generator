RAGAS_QUESTION_PROMPT = """
Your task is to formulate a question from given context satisfying the rules given below:

1. The question should be fully answered from the given context.
2. The question should be framed from a part that contains non-trivial information.
3. The answer should not contain any links.
4. The question should be of moderate difficulty.
5. The question must be reasonable and must be understood and responded to by humans.
6. Do not use phrases such as 'provided context', 'given context', etc. in the question.

Context:
{context}

Return only the question.
"""


RAGAS_ANSWER_PROMPT = """
Answer the question using the information from the given context.

Question:
{question}

Context:
{context}

Return only the answer.
"""