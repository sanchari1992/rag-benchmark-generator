import os
from dotenv import load_dotenv

from benchmark_generation.labelers.ares_labeler import ARESLabeler
from benchmark_generation.llm.client import LLMClient   # IMPORTANT

load_dotenv()

# -------------------------
# FIX 1: define model
# -------------------------
MODEL_NAME = "deepseek-chat"

# -------------------------
# FIX 2: create llm client
# -------------------------
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("Missing API key")

llm = LLMClient(api_key=api_key)

# -------------------------
# create labeler
# -------------------------
labeler = ARESLabeler(llm, MODEL_NAME)

# -------------------------
# FIX 3: safe path
# -------------------------
input_path = r"exports\ares\wizard\run_20260623_210133\benchmark.json"
output_path = r"exports\ares\wizard\run_20260623_210133\labeled_run2.json"

labeler.label_dataset(input_path, output_path)