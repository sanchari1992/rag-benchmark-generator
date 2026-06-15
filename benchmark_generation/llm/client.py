from openai import OpenAI


class LLMClient:

    def __init__(self, api_key, base_url="https://api.deepseek.com"):

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def generate(self, model, messages, temperature=0.2):

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )

        return response.choices[0].message.content