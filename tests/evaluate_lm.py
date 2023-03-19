"""
Evaluates a Causal LM.
"""

# pylint: disable=missing-parentheses-for-call-in-test
# pylint: disable=using-constant-test

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm


def qnli(num_fewshot = 3):
    """Returns List[(int, str)] dataset"""
    full = load_dataset("glue", "qnli", split="validation")
    prompt = "Output 0 if Sentence answers Question, otherwise Output 1.\n"
    for i in range(num_fewshot):
        question = full[i]['question']
        answer = full[i]['sentence']
        label = full[i]['label']
        prompt += f'Question: {question}\n' + f'Sentence: {answer}\n' + f'Output: {label}\n'

    data = []
    for i in range(num_fewshot, len(full)):
        few_shot_example = prompt
        question = full[i]['question']
        answer = full[i]['sentence']
        label = full[i]['label']
        few_shot_example += f'Question: {question}\n' + f'Sentence: {answer}\n' + 'Output: '
        data.append((few_shot_example, label))
    return data

def wnli(num_fewshot = 3):
    """Returns List[(int, str)] dataset"""
    full = load_dataset("glue", "wnli", split="validation")
    prompt = "Output 1 if Sentence1 implies Sentence2, otherwise Output 0.\n"
    for i in range(num_fewshot):
        sentence1 = full[i]['sentence1']
        sentence2 = full[i]['sentence2']
        label = full[i]['label']
        prompt += f'Sentence1: {sentence1}\n' + f'Sentence2: {sentence2}\n' + f'Output: {label}\n'

    data = []
    for i in range(num_fewshot, len(full)):
        few_shot_example = prompt
        sentence1 = full[i]['sentence1']
        sentence2 = full[i]['sentence2']
        label = full[i]['label']
        few_shot_example += f'Sentence1: {sentence1}\n' + f'Sentence2: {sentence2}\n' + 'Output: '
        data.append((few_shot_example, label))
    return data

def sentiment(num_fewshot = 3):
    """Returns List[(int, str)] dataset"""
    full = load_dataset("carblacac/twitter-sentiment-analysis", "None", split="validation")
    prompt = 'Output 1 if Text displays positive sentiment, else Output 0 for negative sentiment.\n'
    for i in range(num_fewshot):
        text = full[i]['text']
        feeling = full[i]['feeling']
        prompt += f'Text: {text}\n' 'Output: {feeling}\n'

    data = []
    for i in range(num_fewshot, len(full)):
        few_shot_example = prompt
        text = full[i]['text']
        feeling = full[i]['feeling']
        few_shot_example += f'Text: {text}\n' + 'Output: '
        data.append((few_shot_example, feeling))
    return data



DATASETS = {
    'qnli' : qnli,
    'wnli' : wnli,
    'sentiment' : sentiment,
}

def evaluate_model(model, tokenizer, dataset, device, batch_size=8):
    """Evaluates model on dataset."""
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-instance-attributes


    token_id_0 = tokenizer.encode('0', add_special_tokens=False)[0]
    token_id_1 = tokenizer.encode('1', add_special_tokens=False)[0]

    num_correct = 0
    num_evaluated = 0

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i + batch_size]
        prompts, labels = zip(*batch)

        input_ids = tokenizer(prompts,
                              return_tensors='pt',
                              padding=True,
                              truncation=True).input_ids.to(device)
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        logits_0 = logits[:, -1, token_id_0]
        logits_1 = logits[:, -1, token_id_1]

        predictions = (logits_1 > logits_0).int().tolist()
        num_correct += sum(p == l for p, l in zip(predictions, labels))
        num_evaluated += batch_size

    return num_correct / num_evaluated

if __name__ == '__main__':
    EVAL_SIZE = 1000
    for model_name in (
        "EleutherAI/pythia-1b",
        "theblackcat102/pythia-1b-deduped-sft",
        "gmukobi/pythia-1b-superhf-v1.0"):
        eval_device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        eval_model = AutoModelForCausalLM.from_pretrained(model_name).to(eval_device)
        eval_tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"\n\n\nEvaluating model: {model_name}")
        for d in ('sentiment', 'qnli', 'wnli'):
            print(f'--------------------------\nEvaluating on dataset: {d}')
            eval_dataset = DATASETS[d]()
            accuracy = evaluate_model(
                eval_model, eval_tokenizer, eval_dataset[:EVAL_SIZE], eval_device
                )
            print(f'Accuracy: {accuracy}\n')
