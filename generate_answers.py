import json
import os
import time
import datetime
import platform

import psutil
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

        context = "\n\n".join(context_chunks[:5])

        system_prompt = (
            "You are a helpful AI assistant.\n"
            "Answer the user's question as accurately and clearly as possible.\n"
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
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0
        )


        answer_text = response.choices[0].message.content.strip()


        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }


        return answer_text, usage



# =========================================================
# LOAD BENCHMARK
# =========================================================
def load_json(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



# =========================================================
# APPEND METRICS LOG
# =========================================================
def append_run_metrics(log_file, metrics):

    with open(log_file, "a", encoding="utf-8") as f:

        f.write("\n")
        f.write("=" * 80)
        f.write("\n")

        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

        f.write("=" * 80)
        f.write("\n")



# =========================================================
# MAIN PIPELINE
# =========================================================
def generate_rag_answers(
        benchmark_path,
        rag_name,
        output_path="results.json",
        metrics_file="run_metrics.txt"
):


    data = load_json(benchmark_path)


    # ----------------------------
    # Start timer
    # ----------------------------
    run_start = time.time()


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


    # Metrics trackers

    retrieval_times = []
    generation_times = []

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    error_count = 0



    # ----------------------------
    # Inference loop
    # ----------------------------
    print("\n[3] Running RAG inference...\n")


    for item in tqdm(data):

        qid = item["id"]

        question = item["question"]



        # ==================================
        # STEP 1: RETRIEVE CONTEXT
        # ==================================

        try:

            retrieval_start = time.time()


            retrieved = adapter.retrieve(question)


            retrieval_times.append(
                time.time() - retrieval_start
            )


            if isinstance(retrieved, list):

                context_chunks = [
                    r["text"]
                    if isinstance(r, dict)
                    and "text" in r
                    else str(r)

                    for r in retrieved
                ]

            else:

                context_chunks = [str(retrieved)]


        except Exception:

            context_chunks = []

            error_count += 1



        # ==================================
        # STEP 2: GENERATE ANSWER
        # ==================================

        try:

            generation_start = time.time()


            response, usage = llm.answer(
                question,
                context_chunks
            )


            generation_times.append(
                time.time() - generation_start
            )


            total_input_tokens += usage["input_tokens"]

            total_output_tokens += usage["output_tokens"]

            total_tokens += usage["total_tokens"]



        except Exception:

            response = ""

            error_count += 1



        # ==================================
        # SAVE RESULT
        # ==================================

        results.append(
            {
                "id": qid,

                "repo_name":
                    item.get("repo_name"),

                "strategy":
                    item.get("strategy"),

                "generator_model":
                    llm.model,

                "question":
                    question,

                "response":
                    response,

                "retrieved_context":
                    context_chunks[:5]
            }
        )



    # =================================================
    # SAVE ANSWERS
    # =================================================

    with open(output_path, "w", encoding="utf-8") as f:

        json.dump(
            results,
            f,
            indent=2
        )


    # =================================================
    # COMPUTE RUN METRICS
    # =================================================

    total_runtime = time.time() - run_start

    num_questions = len(data)



    metrics = {

        "timestamp":
            datetime.datetime.now().isoformat(),


        "benchmark_file":
            benchmark_path,


        "output_file":
            output_path,


        "rag_system":
            rag_name,


        "llm_model":
            llm.model,


        "questions_processed":
            num_questions,



        # -------------------------
        # Timing
        # -------------------------

        "total_runtime_seconds":
            round(total_runtime, 3),


        "average_runtime_per_question_seconds":
            round(
                total_runtime /
                max(num_questions, 1),
                3
            ),


        "average_retrieval_time_seconds":
            round(
                sum(retrieval_times) /
                max(len(retrieval_times), 1),
                3
            ),


        "average_generation_time_seconds":
            round(
                sum(generation_times) /
                max(len(generation_times), 1),
                3
            ),



        # -------------------------
        # Token Usage
        # -------------------------

        "total_input_tokens":
            total_input_tokens,


        "total_output_tokens":
            total_output_tokens,


        "total_tokens":
            total_tokens,


        "average_input_tokens_per_question":
            round(
                total_input_tokens /
                max(num_questions,1),
                2
            ),


        "average_output_tokens_per_question":
            round(
                total_output_tokens /
                max(num_questions,1),
                2
            ),



        # -------------------------
        # Errors
        # -------------------------

        "errors":
            error_count,



        # -------------------------
        # System Info
        # -------------------------

        "cpu_count":
            psutil.cpu_count(),


        "memory_gb":
            round(
                psutil.virtual_memory().total /
                (1024**3),
                2
            ),


        "python_version":
            platform.python_version(),


        "operating_system":
            platform.platform()

    }



    append_run_metrics(
        metrics_file,
        metrics
    )



    print(
        f"\n✔ Saved answers to {output_path}"
    )

    print(
        f"✔ Metrics appended to {metrics_file}"
    )




# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":

    import argparse


    parser = argparse.ArgumentParser()


    parser.add_argument(
        "benchmark_path"
    )


    parser.add_argument(
        "--rag",
        required=True,
        help="RAG name (neural / ragify / etc)"
    )


    parser.add_argument(
        "--output",
        default="results.json"
    )


    parser.add_argument(
        "--metrics",
        default="run_metrics.txt"
    )


    args = parser.parse_args()



    generate_rag_answers(
        args.benchmark_path,
        args.rag,
        args.output,
        args.metrics
    )