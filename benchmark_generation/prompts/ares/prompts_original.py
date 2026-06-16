ARES_QUERY_PROMPT = """
You are generating a synthetic search query.

Generate a realistic user query that can be answered
using the document below.

The query should:

1. Be answerable from the document.
2. Require understanding of the document.
3. Be realistic and human-like.
4. Not mention the document itself.

{few_shot_examples}

Document:
{document}

Query:
"""

ARES_ANSWER_PROMPT = """
Answer the query using only the information
contained in the document.

{few_shot_examples}

Query:
{query}

Document:
{document}

Answer:
"""

ARES_CONTRADICTORY_ANSWER_PROMPT = """
Generate an answer that contradicts
the information in the document.

Query:
{query}

Document:
{document}

Contradictory Answer:
"""