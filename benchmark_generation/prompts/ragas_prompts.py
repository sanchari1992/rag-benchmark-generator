RAGAS_QUESTION_PROMPT = """
Generate a question from the context.

Rules:
- Must be answerable from context
- No mention of "context"
- No links
- Moderate difficulty
- Non-trivial information only

Context:
{context}

Return ONLY the question.
"""


RAGAS_ANSWER_PROMPT = """
Answer ONLY using the context.

Rules:
- No hallucination
- If unknown say "Not enough information"

Context:
{context}

Question:
{question}

Return ONLY the answer.
"""