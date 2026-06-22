import json
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# RAG Adapter
# ----------------------------
from adapters.factory import load_adapter

# ----------------------------
# LLM Client
# ----------------------------
from openai import OpenAI


# =========================================================
# LLM WRAPPER (DeepSeek)
# =========================================================
class DeepSeekRAGGenerator:
    def __init__(self, model="deepseek-chat"):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        self.model = model

    def answer(self, question, context_chunks):
        context = "\n\n".join(context_chunks[:5])  # top-k context

        system_prompt = (
            "You are a precise QA system using retrieved context.\n"
            "Rules:\n"
            "- Answer ONLY using the provided context\n"
            "- If answer is not in context, say: unknown\n"
            "- No explanation\n"
            "- No extra words\n"
            "- Keep answer extremely concise\n"
        )

        user_prompt = f"""
CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )

        return response.choices[0].message.content.strip()


# =========================================================
# LOAD BENCHMARK
# =========================================================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# MAIN PIPELINE
# =========================================================
def generate_rag_answers(
    benchmark_path,
    rag_name,
    output_path="results.json"
):
    data = load_json(benchmark_path)

    # ----------------------------
    # Load RAG adapter
    # ----------------------------
    print("\n[1] Loading RAG adapter...")
    adapter = load_adapter(rag_name)

    print("✔ RAG loaded")

    # ----------------------------
    # Load LLM
    # ----------------------------
    print("\n[2] Loading LLM...")
    llm = DeepSeekRAGGenerator()

    print("✔ LLM ready")

    results = []

    # ----------------------------
    # Inference loop
    # ----------------------------
    print("\n[3] Running RAG inference...\n")

    for item in tqdm(data):
        qid = item["id"]
        question = item["question"]

        # ----------------------------
        # STEP 1: Retrieve context from RAG
        # ----------------------------
        try:
            retrieved = adapter.retrieve(question)

            # normalize
            if isinstance(retrieved, list):
                context_chunks = [
                    r["text"] if isinstance(r, dict) and "text" in r else str(r)
                    for r in retrieved
                ]
            else:
                context_chunks = [str(retrieved)]

        except Exception as e:
            context_chunks = []

        # ----------------------------
        # STEP 2: LLM generates answer using context
        # ----------------------------
        try:
            response = llm.answer(question, context_chunks)
        except Exception:
            response = ""

        # ----------------------------
        # STORE OUTPUT
        # ----------------------------
        results.append({
            "id": item["id"],
            "repo_name": item.get("repo_name"),
            "strategy": item.get("strategy"),
            "generator_model": "deepseek-chat",
            "question": question,
            "response": response,
            "retrieved_context": context_chunks[:5]  # keep traceability
        })

    # ----------------------------
    # SAVE RESULTS
    # ----------------------------
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n✔ Saved results to {output_path}")


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("benchmark_path")
    parser.add_argument("--rag", required=True, help="RAG name (neural / ragify / etc)")
    parser.add_argument("--output", default="results.json")

    args = parser.parse_args()

    generate_rag_answers(
        args.benchmark_path,
        args.rag,
        args.output
    )