import os
import json

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


    def score(self, prompt):
        """
        Generic LLM scoring interface.

        Used by:
        - RAGAS metrics
        - ARES metrics
        - DataMorgana metrics
        - Custom metrics

        Returns JSON.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )


        content = response.choices[0].message.content.strip()


        try:
            return json.loads(content)

        except Exception:

            return {
                "score": 0.0,
                "reason": "Failed to parse LLM output",
                "raw_output": content
            }



    def judge(self, question, expected, predicted):
        """
        Existing correctness evaluation.
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


        return self.score(prompt)