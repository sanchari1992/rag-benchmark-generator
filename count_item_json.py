import json

with open("repositories/COVID-QA/data/question-answering/200423_covidQA.json", "r", encoding="utf-8") as f:
    data = json.load(f)

num_articles = len(data["data"])
num_paragraphs = 0
num_questions = 0

for article in data["data"]:
    num_paragraphs += len(article["paragraphs"])
    for paragraph in article["paragraphs"]:
        num_questions += len(paragraph["qas"])

print(f"Articles   : {num_articles}")
print(f"Paragraphs : {num_paragraphs}")
print(f"Questions  : {num_questions}")