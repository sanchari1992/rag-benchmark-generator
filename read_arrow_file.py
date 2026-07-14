from datasets import Dataset

dataset = Dataset.from_file(
    r".\repositories\DiffKS\datasets\wizard_of_wikipedia_train\data-00000-of-00002.arrow"
)

print(dataset)
print(dataset[:1])