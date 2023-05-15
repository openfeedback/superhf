"""
Re-attaches the prompts to openai completions
"""
import os
import json

from superhf.data import get_superhf_prompts


def fix_dataset_completions(dataset_name, completions):
    """
    Given a dataset and a list of completions, remove the prompt by splitting on
    '\n\nAssistant: ' and then re-attaching the prompt to the completion
    """
    prompts = get_superhf_prompts(dataset_name, split="test")
    fixed_completions = []
    for i, completion in enumerate(completions):
        prefix = "\n\nHuman: "
        assert (
            completion[: len(prefix)] == prefix
        ), "Completion doesn't start with human prompt"
        completion = completion.split("\n\nAssistant: ")[1]
        prompt = prompts[i]
        fixed_completions.append(prompt + " " + completion)
    return fixed_completions


def main():
    """
    Main function
    """
    script_path_dir = os.path.dirname(os.path.abspath(__file__))

    path_to_completions = os.path.join(script_path_dir, "openai_completions/")
    path_to_fixed_completions = os.path.join(
        script_path_dir, "openai_completions_fixed/"
    )
    if not os.path.exists(path_to_fixed_completions):
        os.makedirs(path_to_fixed_completions)
    # get all the json files in the directory
    openai_completion_files = [
        file for file in os.listdir(path_to_completions) if file.endswith(".json")
    ]
    # load all the completions
    for file in openai_completion_files:
        with open(path_to_completions + file, "r", encoding="utf-8") as infile:
            print(f"Fixing completions for {file}")
            completions_dict = json.load(infile)
            for dataset_name in completions_dict.keys():
                completions_dict[dataset_name] = fix_dataset_completions(
                    dataset_name, completions_dict[dataset_name]
                )
            with open(
                path_to_fixed_completions + file, "w", encoding="utf-8"
            ) as outfile:
                json.dump(completions_dict, outfile, indent=4)
                outfile.write("\n")
                print(f"Fixed completions wrote to {path_to_fixed_completions + file}")
    print("Done!")


if __name__ == "__main__":
    main()
