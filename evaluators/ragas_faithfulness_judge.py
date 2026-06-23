import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class FaithfulnessJudge:

    def __init__(self, model="deepseek-chat"):

        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

        self.model = model

    def judge(self, question, context, answer):

        prompt = f"""
You are implementing the RAGAS Faithfulness metric.

Faithfulness measures whether every factual statement
made in the answer can be inferred from the provided context.

Step 1.
Extract one or more independent factual statements from
the answer.

Step 2.
For each statement determine whether it is supported
by the context.

Return ONLY valid JSON.

{{
  "statements":[
      {{
          "statement":"...",
          "supported":true
      }}
  ],
  "faithfulness":0.0
}}

where

faithfulness =
(number of supported statements) /
(total statements)

Question:
{question}

Context:
{context}

Answer:
{answer}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        content = response.choices[0].message.content.strip()

        try:
            return json.loads(content)
        except Exception:

            return {
                "statements": [],
                "faithfulness": 0
            }