import numpy as np


class MetricRegistry:
    def __init__(self, judge, embed_fn=None):
        self.judge = judge
        self.embed_fn = embed_fn  # callable: str -> np.ndarray
        self._metrics = {
            "ragas_faithfulness": self.faithfulness,
            "ragas_answer_relevancy": self.answer_relevancy,
            "ragas_context_relevance": self.context_relevance,
        }

    def get(self, name):
        if name not in self._metrics:
            raise ValueError(f"Unknown metric: {name}")
        return self._metrics[name]

    # ---------- Faithfulness ----------
    def faithfulness(self, item):
        answer = item["generated_answer"]
        question = item["question"]
        context = item["retrieved_context"]

        extraction_prompt = f"""Given a question and answer, break the answer down into one or more standalone factual statements, one per claim.
Question: {question}
Answer: {answer}
Return ONLY valid JSON: {{"statements": ["...", "..."]}}"""

        extraction = self.judge.score(extraction_prompt)
        statements = extraction.get("statements", [])

        if not statements:
            return {"score": None, "reason": "No statements extracted", "verdicts": []}

        stmt_block = "\n".join(f"- {s}" for s in statements)
        verification_prompt = f"""Given the context and the following statements, determine whether each statement is supported by the information in the context. Give a brief explanation, then a final verdict (Yes/No) for each statement, in order.
Context: {context}
Statements:
{stmt_block}
Return ONLY valid JSON: {{"verdicts": [{{"statement": "...", "explanation": "...", "verdict": "Yes"}}, ...]}}"""

        verification = self.judge.score(verification_prompt)
        verdicts = verification.get("verdicts", [])

        if not verdicts:
            return {"score": None, "reason": "No verdicts returned", "statements": statements}

        supported = sum(1 for v in verdicts if str(v.get("verdict", "")).strip().lower() == "yes")
        score = supported / len(verdicts)

        return {
            "score": score,
            "reason": f"{supported}/{len(verdicts)} statements supported by context",
            "statements": statements,
            "verdicts": verdicts,
        }

    # ---------- Answer Relevancy ----------
    def answer_relevancy(self, item, n=3):
        answer = item["generated_answer"]
        question = item["question"]

        if self.embed_fn is None:
            raise RuntimeError(
                "answer_relevancy requires an embed_fn (e.g. OpenAI embeddings or "
                "sentence-transformers). Pass one to MetricRegistry(judge, embed_fn=...)."
            )

        gen_questions = []
        for _ in range(n):
            prompt = f"""Generate a question that the following answer would be a direct response to.
Answer: {answer}
Return ONLY valid JSON: {{"question": "..."}}"""
            result = self.judge.score(prompt)
            gq = result.get("question")
            if gq:
                gen_questions.append(gq)

        if not gen_questions:
            return {"score": None, "reason": "No reverse-questions generated", "generated_questions": []}

        orig_emb = self.embed_fn(question)
        sims = []
        for gq in gen_questions:
            gq_emb = self.embed_fn(gq)
            sim = np.dot(orig_emb, gq_emb) / (
                np.linalg.norm(orig_emb) * np.linalg.norm(gq_emb) + 1e-8
            )
            sims.append(float(sim))

        score = sum(sims) / len(sims)
        return {
            "score": score,
            "reason": f"Mean cosine similarity across {len(sims)} generated questions",
            "generated_questions": gen_questions,
            "similarities": sims,
        }

    # ---------- Context Relevance ----------
    def context_relevance(self, item):
        question = item["question"]
        context = item["retrieved_context"]

        prompt = f"""Extract only the sentences from the context below that are needed to answer the question. Copy them verbatim, do not alter wording. If no relevant sentences exist, or the question can't be answered from this context, return an empty list.
Question: {question}
Context: {context}
Return ONLY valid JSON: {{"extracted_sentences": ["...", "..."]}}"""

        result = self.judge.score(prompt)
        extracted = result.get("extracted_sentences", [])

        total_sentences = len([s for s in context.split(".") if s.strip()])
        if total_sentences == 0:
            return {"score": None, "reason": "Empty context", "extracted_sentences": extracted}

        score = min(len(extracted) / total_sentences, 1.0)
        return {
            "score": score,
            "reason": f"{len(extracted)}/{total_sentences} sentences judged relevant",
            "extracted_sentences": extracted,
        }