import re
from datasets import load_dataset

dataset = load_dataset("Anthropic/model-written-evals")

def preprocess(data):
    questions = [re.search("Question: (.*?) Choices", item).group(1) for item in data]
    a = [re.search("\(A\) (.*?) \(B\)", item).group(1) for item in data]
    b = [re.search("\(B\) (.*)", item).group(1) for item in data]
    labels = [0 if x.strip() == "(A)" else 1 for x in data["answer_matching_behavior"]]
    return questions, a, b, labels

dataset = dataset.map(preprocess, batched=True)