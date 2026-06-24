import json
from tqdm import tqdm
from benchmark_generation.judges.deepseek_ares_judge import DeepSeekARESJudge


class ARESLabeler:

    def __init__(self, llm, model):
        self.judge = DeepSeekARESJudge(llm, model)

    def label_dataset(self, input_path, output_path):

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        labeled = []

        for ex in tqdm(data):

            try:
                q = ex["question"]
                ctx = ex["source_text"]

                pos = ex["answer"]
                neg = ex.get("contradictory_answer")

                ex["positive_labels"] = self.judge.evaluate(q, ctx, pos)

                if neg:
                    ex["negative_labels"] = self.judge.evaluate(q, ctx, neg)

                labeled.append(ex)

            except Exception as e:
                print("error:", e)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(labeled, f, indent=2)

        return labeled