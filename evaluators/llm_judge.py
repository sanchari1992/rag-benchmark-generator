import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class DeepSeekJudge:
    """
    LLM-based evaluator using DeepSeek API (OpenAI-compatible).
    """

    def __init__(self, model="deepseek-chat"):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        self.model = model

    def judge(self, question, expected, predicted):
        """
        Returns structured JSON judgment.
        """

        prompt = f"""
You are an evaluation system.

Compare the expected answer and predicted answer.

Rules:
- If meaning is correct → mark correct
- Allow minor formatting differences
- Be strict about factual mismatch

Return ONLY valid JSON:

{{
  "verdict": "correct" or "incorrect",
  "score": float between 0 and 1,
  "reason": "short explanation"
}}

Question:
{question}

Expected Answer:
{expected}

Predicted Answer:
{predicted}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        content = response.choices[0].message.content.strip()

        # safe JSON parsing
        import json
        try:
            return json.loads(content)
        except:
            return {
                "verdict": "incorrect",
                "score": 0.0,
                "reason": "Failed to parse LLM output"
            }