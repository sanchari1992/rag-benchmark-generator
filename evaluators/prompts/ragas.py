def statement_extraction_prompt(question, answer):
    return f"""Given a question and answer, break the answer down into one or more standalone statements, one per factual claim in each sentence.
Question: {question}
Answer: {answer}
Return JSON: {{"statements": ["...", "..."]}}
"""

def statement_verification_prompt(context, statements):
    stmt_block = "\n".join(f"- {s}" for s in statements)
    return f"""Given the context and the following statements, determine whether each statement is supported by the information in the context. Give a brief explanation, then a final verdict (Yes/No) for each.
Context: {context}
Statements:
{stmt_block}
Return JSON: {{"verdicts": [{{"statement": "...", "explanation": "...", "verdict": "Yes"}}, ...]}}
"""

# Faithfulness score = |supported| / |total statements| — computed, not scored by the LLM
def compute_faithfulness(verdicts):
    supported = sum(1 for v in verdicts if v["verdict"].lower() == "yes")
    return supported / len(verdicts)

def reverse_question_prompt(answer):
    return f"""Generate a question that this answer would be a response to.
Answer: {answer}
Return JSON: {{"question": "..."}}
"""

# Generate n=3-5 reverse questions per answer, embed them + the original question
# (e.g. text-embedding-3-small), then:
def compute_answer_relevance(orig_q_emb, generated_q_embs):
    import numpy as np
    sims = [np.dot(orig_q_emb, qe) / (np.linalg.norm(orig_q_emb) * np.linalg.norm(qe))
            for qe in generated_q_embs]
    return sum(sims) / len(sims)


def context_extraction_prompt(question, context):
    return f"""Extract only the sentences from the context below that are needed to answer the question. If no sentences are relevant, or the question can't be answered from this context, return "Insufficient Information". Do not alter the wording of any extracted sentence.
Question: {question}
Context: {context}
Return JSON: {{"extracted_sentences": ["...", "..."]}}
"""

def compute_context_relevance(extracted_sentences, context):
    total_sentences = len([s for s in context.split(".") if s.strip()])
    return len(extracted_sentences) / total_sentences