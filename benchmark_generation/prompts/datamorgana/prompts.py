DATAMORGANA_QUESTION_PROMPT = """
You are a user simulator that generates realistic benchmark questions.

Generate ONE question satisfying all requirements.

The question:

1. Must be answerable from the provided context.
2. Must be self-contained.
3. Must not reference the context, document, article, source, passage, or author.
4. Must reflect ALL user characteristics below.
5. Must reflect ALL question characteristics below.
6. Must be natural and realistic.
7. Must require non-trivial information from the context.

Context:
{context}

User Characteristics:
{user_characteristics}

Question Characteristics:
{question_characteristics}

Return ONLY the question.
"""


DATAMORGANA_ANSWER_PROMPT = """
Answer the question using ONLY information from the provided context.

Question:
{question}

Context:
{context}

Return ONLY the answer.
"""