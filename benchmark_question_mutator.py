"""
benchmark_question_mutator.py

Generate metric-specific harder benchmark questions.

For each selected missing metric:
    - mutate every benchmark question once
    - append generated questions

Example:
    Benchmark size = 100
    Metrics selected = 5

    Output size:
        100 + (100 * 5)
        = 600


Usage:

python benchmark_question_mutator.py \
    --input benchmark.json \
    --metrics metrics.json \
    --output new_benchmark.json \
    --report report.json \
    --top_metrics 5
"""

import argparse
import json
import os

from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI


############################################################
# Environment
############################################################

load_dotenv()


client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


############################################################
# JSON Utilities
############################################################

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as f:
        return json.load(f)



def save_json(data, path):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


############################################################
# Metric Selection
############################################################

def rank_metrics(metric_summary, top_k=5):
    """
    Select metrics where benchmark coverage is low.

    Lower normalized average means fewer examples
    with that property.
    """

    candidates = []


    for metric, stats in metric_summary.items():

        if not isinstance(stats, dict):
            continue


        avg = stats.get("average")
        minimum = stats.get("minimum")
        maximum = stats.get("maximum")


        if (
            avg is None
            or minimum is None
            or maximum is None
        ):
            continue


        if maximum == minimum:
            continue


        normalized_position = (
            (avg - minimum)
            /
            (maximum - minimum)
        )


        missing_score = (
            1 - normalized_position
        )


        candidates.append(
            {
                "metric": metric,
                "average": avg,
                "minimum": minimum,
                "maximum": maximum,
                "normalized_position":
                    normalized_position,
                "missing_score":
                    missing_score
            }
        )


    candidates.sort(
        key=lambda x: x["missing_score"],
        reverse=True
    )


    return candidates[:top_k]



############################################################
# DeepSeek Mutation
############################################################

def build_prompt(question, target_metric):

    return f"""
You are an expert RAG benchmark generation system.

Modify the following benchmark question to increase
the value of this specific metric:

Target metric:
{target_metric}


Original question:
{question}


Rules:
1. Preserve the original information need.
2. Preserve the expected answer.
3. Do not introduce unsupported facts.
4. Modify only the question.
5. Make the question naturally harder.
6. Increase the target metric through meaningful reasoning.
7. Avoid changing the topic.

Return ONLY the modified question.
"""



def mutate_question(question, target_metric):

    prompt = build_prompt(
        question,
        target_metric
    )


    response = client.chat.completions.create(
        model="deepseek-chat",

        messages=[
            {
                "role": "system",
                "content":
                    "You generate challenging RAG benchmark questions."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.7,

        max_tokens=256
    )


    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )



############################################################
# Benchmark Mutation
############################################################

def generate_mutated_benchmark(
        benchmark,
        selected_metrics
):
    """
    Generate one mutation per:
        question × selected_metric
    """

    generated_records = []

    report = []


    for metric_info in selected_metrics:


        target_metric = (
            metric_info["metric"]
        )


        print(
            f"\nGenerating for metric: "
            f"{target_metric}"
        )


        for record in tqdm(
            benchmark,
            desc=target_metric
        ):


            question = record.get(
                "question",
                ""
            )


            try:

                modified_question = (
                    mutate_question(
                        question,
                        target_metric
                    )
                )


                new_record = record.copy()


                new_record["id"] = (
                    f"{record['id']}_"
                    f"{target_metric}_mutation"
                )


                new_record["parent_id"] = (
                    record["id"]
                )


                new_record["original_question"] = (
                    question
                )


                new_record["question"] = (
                    modified_question
                )


                new_record["mutation_metric"] = (
                    target_metric
                )


                new_record["compression_stage"] = (
                    "metric_mutation"
                )


                generated_records.append(
                    new_record
                )


                report.append(
                    {
                        "parent_id":
                            record["id"],

                        "metric":
                            target_metric,

                        "original_question":
                            question,

                        "modified_question":
                            modified_question
                    }
                )


            except Exception as e:


                report.append(
                    {
                        "parent_id":
                            record["id"],

                        "metric":
                            target_metric,

                        "error":
                            str(e)
                    }
                )


    return generated_records, report



############################################################
# Main
############################################################

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--input",
        required=True,
        help="Input benchmark JSON"
    )


    parser.add_argument(
        "--metrics",
        required=True,
        help="Metric summary JSON"
    )


    parser.add_argument(
        "--output",
        required=True,
        help="Output benchmark JSON"
    )


    parser.add_argument(
        "--report",
        required=True,
        help="Mutation report JSON"
    )


    parser.add_argument(
        "--top_metrics",
        type=int,
        default=5
    )


    args = parser.parse_args()



    ################################################
    # Load
    ################################################

    benchmark = load_json(
        args.input
    )


    metric_summary = load_json(
        args.metrics
    )



    ################################################
    # Select metrics
    ################################################

    selected_metrics = rank_metrics(
        metric_summary,
        args.top_metrics
    )


    print("\nSelected metrics:")

    for metric in selected_metrics:

        print(
            f"{metric['metric']} "
            f"(missing score="
            f"{metric['missing_score']:.3f})"
        )



    ################################################
    # Generate
    ################################################

    mutated_records, report = (
        generate_mutated_benchmark(
            benchmark,
            selected_metrics
        )
    )


    final_dataset = (
        benchmark +
        mutated_records
    )



    ################################################
    # Save
    ################################################

    save_json(
        final_dataset,
        args.output
    )


    save_json(
        {
            "selected_metrics":
                selected_metrics,

            "original_size":
                len(benchmark),

            "generated_size":
                len(mutated_records),

            "final_size":
                len(final_dataset),

            "mutations":
                report
        },
        args.report
    )


    print("\nCompleted")
    print(
        "Original questions:",
        len(benchmark)
    )

    print(
        "Generated questions:",
        len(mutated_records)
    )

    print(
        "Final benchmark size:",
        len(final_dataset)
    )



if __name__ == "__main__":
    main()