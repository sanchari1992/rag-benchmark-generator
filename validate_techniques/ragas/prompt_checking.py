import json
import os
import time
import requests
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# =========================
# CONFIG
# =========================

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

MAX_PAGES = 50
OUTPUT_FILE = "ragas_test_benchmark.json"


# =========================
# RAGAS PROMPTS
# =========================

RAGAS_QUESTION_PROMPT = """
Your task is to formulate a question from given context satisfying the rules given below:

1. The question should be fully answered from the given context.
2. The question should be framed from a part that contains non-trivial information.
3. The question should not contain links.
4. The question should be of moderate difficulty.
5. The question must be reasonable and understandable by humans.
6. Do not use phrases such as 'provided context', 'given context', etc. in the question.

Context:
{context}

Return only the question.
"""

RAGAS_ANSWER_PROMPT = """
Answer the question using the information from the given context.

Question:
{question}

Context:
{context}

Return only the answer.
"""


# =========================
# STEP 1: LOAD DATASET
# =========================

def load_wikieval_sources():
    ds = load_dataset("vibrantlabsai/WikiEval")
    data = ds["train"]

    sources = data["source"]

    seen = set()
    pages = []

    for s in sources:
        if s and s not in seen:
            seen.add(s)
            pages.append(s)

    return pages, data


# =========================
# STEP 2: WIKIPEDIA FETCH
# =========================

def fetch_wikipedia_intro(title: str):
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + title.replace(" ", "_")

    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None

        data = r.json()

        return {
            "title": title,
            "context": data.get("extract", ""),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", "")
        }

    except Exception:
        return None


# =========================
# STEP 3: LLM GENERATION
# =========================

def generate_question(context: str):
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{
            "role": "user",
            "content": RAGAS_QUESTION_PROMPT.format(context=context)
        }],
        temperature=0.7
    )

    return resp.choices[0].message.content.strip()


def generate_answer(context: str, question: str):
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{
            "role": "user",
            "content": RAGAS_ANSWER_PROMPT.format(
                context=context,
                question=question
            )
        }],
        temperature=0.2
    )

    return resp.choices[0].message.content.strip()


# =========================
# STEP 4: BUILD BENCHMARK
# =========================

def build_benchmark(pages, max_pages=50):
    benchmark = []

    for i, title in enumerate(tqdm(pages[:max_pages])):

        page_data = fetch_wikipedia_intro(title)
        if not page_data:
            continue

        context = page_data["context"]

        if not context or len(context) < 50:
            continue

        try:
            question = generate_question(context)
            answer = generate_answer(context, question)

        except Exception as e:
            print(f"[SKIP] {title}: {e}")
            continue

        benchmark.append({
            "id": i,
            "repo_name": "wikieval_recreated",
            "strategy": "RAGAS",
            "generator_model": "deepseek-chat",

            "source_chunk": 0,
            "context_length": len(context),

            "chapter_name": title,
            "wiki_url": page_data.get("url", ""),

            "question": question,
            "answer": answer,

            "source_text": context,
            "compression_stage": "none"
        })

        # avoid rate limits
        time.sleep(1)

    return benchmark


# =========================
# STEP 5: SAVE OUTPUT
# =========================

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    print("Loading WikiEval dataset...")
    pages, dataset = load_wikieval_sources()

    print(f"Unique Wikipedia pages found: {len(pages)}")

    print("Generating benchmark...")

    benchmark = build_benchmark(pages, MAX_PAGES)

    print(f"Generated samples: {len(benchmark)}")

    save_json(benchmark, OUTPUT_FILE)

    print(f"Saved to {OUTPUT_FILE}")