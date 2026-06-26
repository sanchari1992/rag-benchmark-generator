import json
import random


def load_datamorgana_config(config_path):

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_category(categories):

    probs = [c["probability"] for c in categories]

    return random.choices(
        categories,
        weights=probs,
        k=1
    )[0]


def sample_user_categories(config):

    selected = []

    for categorization in config["user_categorizations"]:

        category = sample_category(
            categorization["categories"]
        )

        selected.append(category)

    return selected


def sample_question_categories(config):

    selected = []

    for categorization in config["question_categorizations"]:

        category = sample_category(
            categorization["categories"]
        )

        selected.append(category)

    return selected