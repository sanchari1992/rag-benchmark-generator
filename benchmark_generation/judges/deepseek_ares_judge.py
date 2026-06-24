import json
import time


class DeepSeekARESJudge:

    def __init__(self, llm_client, model="deepseek-chat"):
        self.llm = llm_client
        self.model = model

    def _call(self, messages, retries=3):

        for i in range(retries):
            try:
                return self.llm.generate(self.model, messages)
            except Exception as e:
                print(f"retry {i+1}: {e}")
                time.sleep(1.5 * (i + 1))

        raise RuntimeError("LLM failed")

    def _parse(self, text):

        try:
            return json.loads(text)
        except:
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end])

    # ---------------- faithfulness ----------------
    def faithfulness(self, q, ctx, ans):

        prompt = f"""
Return JSON only.

Is the answer fully supported by the context?

Question: {q}
Context: {ctx}
Answer: {ans}

Return:
{{"faithfulness": true/false}}
"""

        msg = [
            {"role": "system", "content": "AIS judge"},
            {"role": "user", "content": prompt}
        ]

        return self._parse(self._call(msg))

    def context_relevance(self, q, ctx):

        prompt = f"""
Is context relevant to question?

Q: {q}
Context: {ctx}

Return JSON:
{{"context_relevance": true/false}}
"""

        msg = [
            {"role": "system", "content": "judge"},
            {"role": "user", "content": prompt}
        ]

        return self._parse(self._call(msg))

    def answer_relevance(self, q, ans):

        prompt = f"""
Does answer address question?

Q: {q}
A: {ans}

Return JSON:
{{"answer_relevance": true/false}}
"""

        msg = [
            {"role": "system", "content": "judge"},
            {"role": "user", "content": prompt}
        ]

        return self._parse(self._call(msg))

    def evaluate(self, q, ctx, ans):

        return {
            **self.context_relevance(q, ctx),
            **self.answer_relevance(q, ans),
            **self.faithfulness(q, ctx, ans)
        }