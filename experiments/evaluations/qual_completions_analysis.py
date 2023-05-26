"""
A file for graphing the usage of particular words by a language model.
Currently graphs mean number of completions that use 'sorry' in the completion.
"""
import json
import os
import plotly.graph_objects as go

COMPLETION_FOLDER = "experiments/evaluations/test_completions"

MODELS = [
    "alpaca_7b",
    "llama-7b",
    # "sft-on-preferences-v1",
    "rlhf-v3-lr-5.0e-6-batch-16@gold-run",
    "test-save-alpaca@model-2048-prompts-batch-size-8",
    # "shf-7b-gold-v1",
    # "shf-7b-gold-v1@step-0064",
    # "shf-7b-gold-v1@step-1024",
    # "shf-7b-gold-v1@step-8192",
    "shf-pythia-12B@v3",
    "pythia-12B-deduped",
    "shf-7b-default",
    # "pythia-6.9B-deduped",
]


def main():
    """
    Main function. Creates a graph of the mean number of completions with "sorry" grouped
    by dataset.
    """
    dataset_means = {}

    # pylint: disable=duplicate-code
    for model in MODELS:
        # load completions
        with open(
            os.path.join(COMPLETION_FOLDER, f"{model}.json"), encoding="utf-8"
        ) as file:
            completions_per_dataset = json.load(file)

        # calculate the mean number of completions containing "sorry" per dataset
        for dataset, completions in completions_per_dataset.items():
            sorry_completions = 0
            for completion in completions:
                if "sorry" in completion or "Sorry" in completion:
                    sorry_completions += 1
            total = len(completions)
            mean = sorry_completions / total

            if dataset not in dataset_means:
                dataset_means[dataset] = []
            dataset_means[dataset].append(mean)

    # create the graph
    fig = go.Figure()
    for dataset, means in dataset_means.items():
        fig.add_trace(go.Bar(x=MODELS, y=means, name=dataset))

    fig.update_layout(
        title="Mean Number of Completions with 'Sorry'",
        xaxis_title="Language Model",
        yaxis_title="Mean Number of Completions",
        barmode="group",
    )

    fig.show()


if __name__ == "__main__":
    main()
