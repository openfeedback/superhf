"""How often we observed train reward improving."""

import numpy as np
import pandas as pd


RUNNING_AVG_WINDOW_SHF = 100
RUNNING_AVG_WINDOW_RLHF = 8 * RUNNING_AVG_WINDOW_SHF


def main() -> None:
    """Main function."""

    shf_train_reward_file = "eval_results/training_rewards/shf_training_rewards.csv"
    rlhf_train_reward_file = "eval_results/training_rewards/rlhf_training_rewards.csv"

    for reward_file in [shf_train_reward_file, rlhf_train_reward_file]:
        # Read the CSV files
        rewards_file = pd.read_csv(reward_file)
        # Drop columns that have __MAX or __MIN
        rewards_file = rewards_file.loc[:, ~rewards_file.columns.str.contains("__")]

        # Drop the step column
        rewards_file = rewards_file.drop(columns=["Step"])

        window = (
            RUNNING_AVG_WINDOW_SHF if "shf" in reward_file else RUNNING_AVG_WINDOW_RLHF
        )
        sum_improvement = 0
        num_improving = 0
        num_total = 0

        # Iterate over the columns
        for column_name in rewards_file:
            # Get all the non-blank values
            column = rewards_file[column_name].dropna()

            # Skip the column if it's empty
            if len(column) == 0:
                continue

            # Calculate the starting reward
            start_reward = np.mean(column[:window])

            # Calculate the ending reward
            end_reward = np.mean(column[-window:])

            # Calculate the improvement and whether it improved
            improvement = end_reward - start_reward
            sum_improvement += improvement
            num_improving += 1 if improvement > 0 else 0

            num_total += 1

        print(f"{reward_file} improvement: {sum_improvement / num_total:.3f}")
        print(f"{reward_file} improving: {num_improving / num_total:.3f}")
        print(f"{reward_file} total: {num_total}")


if __name__ == "__main__":
    main()
